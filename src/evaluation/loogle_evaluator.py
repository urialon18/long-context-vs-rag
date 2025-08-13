"""LooGLE-specific evaluator for RAG comparison."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from data.loogle_loader import LooGLEDatasetLoader
from evaluation.metrics import MetricsCalculator, RAGMetrics
from long_context_rag.pipeline import LongContextRAGPipeline
from regular_rag.pipeline import RegularRAGPipeline
from utils.config import config


class LooGLEEvaluator:
    """Evaluator for LooGLE dataset using both RAG approaches."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the evaluator.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.metrics_calculator = MetricsCalculator()
        self.dataset_loader = LooGLEDatasetLoader()

    def run_full_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation of both RAG approaches on LooGLE.

        Args:
            save_results: Whether to save results to files

        Returns:
            Complete evaluation results
        """
        print("Starting LooGLE RAG Comparison Evaluation...")
        print("=" * 60)

        # Step 1: Load dataset and create test questions
        print("\n1. Loading LooGLE dataset and creating test questions...")
        test_questions = self.dataset_loader.create_test_questions()

        # Print dataset statistics
        stats = self.dataset_loader.get_question_stats()
        print("Dataset statistics:")
        print(f"- Total questions: {stats['total_questions']}")
        print(
            f"- Context length: {stats['context_stats']['min_chars']:,} - {stats['context_stats']['max_chars']:,} chars"
        )
        print(f"- Avg context length: {stats['context_stats']['avg_chars']:,} chars")
        print(
            f"- Estimated total tokens: ~{stats['context_stats']['estimated_tokens']:,}"
        )

        # Step 2: Evaluate Regular RAG
        print("\n2. Evaluating Regular RAG...")
        regular_rag_results = self._evaluate_regular_rag(test_questions)

        # Step 3: Evaluate Long Context RAG
        print("\n3. Evaluating Long Context RAG...")
        # Remove initial long sleep to avoid excessive waiting; per-question sleeps remain
        print("✅ Starting Long Context RAG evaluation...")
        long_context_results = self._evaluate_long_context_rag(test_questions)

        # Step 4: Calculate metrics (LLM judge removed; compute directly from predictions)
        print("\n4. Calculating metrics...")
        ground_truth = [{"answer": q["answer"]} for q in test_questions]
        regular_metrics = self.metrics_calculator.calculate_metrics(
            regular_rag_results["predictions"], ground_truth
        )
        long_context_metrics = self.metrics_calculator.calculate_metrics(
            long_context_results["predictions"], ground_truth
        )

        # Step 5: Compare results
        print("\n5. Comparing results...")
        comparison = self._compare_loogle_results(regular_metrics, long_context_metrics)

        # Sanitize predictions for output (remove answer/options/predicted_answer; add ground truth)
        regular_rag_results["predictions"] = self._sanitize_predictions(
            regular_rag_results["predictions"], test_questions
        )
        long_context_results["predictions"] = self._sanitize_predictions(
            long_context_results["predictions"], test_questions
        )

        # Compile complete results
        complete_results = {
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "n_questions": config.n_questions,
                    "max_context_length": config.max_context_length,
                    "random_seed": config.random_seed,
                    "llm_model": config.llm_model,
                    "embedding_model": config.embedding_model,
                    "dataset_name": config.dataset_name,
                    "dataset_subset": config.dataset_subset,
                },
                "dataset_stats": stats,
            },
            "regular_rag": {
                "results": regular_rag_results,
                "metrics": regular_metrics.to_dict(),
            },
            "long_context_rag": {
                "results": long_context_results,
                "metrics": long_context_metrics.to_dict(),
            },
            "comparison": comparison,
        }

        # Step 6: Skip console summary; saving locations are printed in _save_results

        # Step 7: Save results
        if save_results:
            self._save_results(complete_results)

        print("\nEvaluation completed!")
        return complete_results

    def _sanitize_predictions(
        self, predictions: List[Dict[str, Any]], test_questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove unwanted fields and attach ground truth answers for output files."""
        sanitized: List[Dict[str, Any]] = []
        for i, pred in enumerate(predictions):
            pred_copy = dict(pred)
            # Remove fields not needed in output
            pred_copy.pop("answer", None)
            pred_copy.pop("predicted_answer", None)
            pred_copy.pop("options", None)
            # Attach ground truth from dataset
            if i < len(test_questions):
                pred_copy["ground_truth_answer"] = test_questions[i]["answer"]
            sanitized.append(pred_copy)
        return sanitized

    def _evaluate_regular_rag(
        self, test_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate Regular RAG pipeline on LooGLE questions."""
        print("Initializing Regular RAG pipeline...")
        pipeline = RegularRAGPipeline()

        # Answer questions with per-question indexing
        predictions = []

        for i, question in enumerate(test_questions):
            print(f"  Question {i + 1}/{len(test_questions)}")

            # For fair comparison: Index only this question's document
            single_document = [
                {
                    "id": 0,
                    "content": question["context"],
                    "is_relevant": True,
                    "kb_id": 0,
                    "title": question.get("title", f"Document {i}"),
                    "question_id": i,  # Add question_id for chunker
                }
            ]

            # Clear previous index and index only this document
            pipeline.clear_index()
            indexing_stats = pipeline.index_documents(single_document)
            print(
                f"    📋 Indexed {indexing_stats['total_chunks']} chunks from 1 document"
            )

            # Get prediction from Regular RAG
            result = pipeline.answer_question(
                question["question"],
                options=None,  # No multiple choice options
                question_id=i,
            )

            predictions.append(result)

            # No LLM judge: rely on direct comparison metrics later

            # Rate limiting
            if i < len(test_questions) - 1:  # Don't sleep after last question
                print(
                    f"   ⏳ Waiting {config.regular_rag_sleep} seconds to avoid quota limits..."
                )
                time.sleep(config.regular_rag_sleep)

        return {
            "pipeline_type": "Regular RAG",
            "indexing_stats": indexing_stats,
            "predictions": predictions,
            "pipeline_stats": pipeline.get_pipeline_stats(),
        }

    def _evaluate_long_context_rag(
        self, test_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate Long Context RAG pipeline on LooGLE questions."""
        print("Initializing Long Context RAG pipeline...")
        pipeline = LongContextRAGPipeline()

        # For Long Context RAG, we pass the full context directly
        # No need to create separate documents since each question is self-contained

        predictions = []

        for i, question in enumerate(test_questions):
            print(f"  Question {i + 1}/{len(test_questions)}")

            # Get prediction from Long Context RAG
            # We pass the context as a single document
            documents = [
                {
                    "id": 0,
                    "content": question["context"],
                    "is_relevant": True,
                    "kb_id": 0,
                    "title": question.get("title", "Document"),
                }
            ]

            # Load the single document for this question
            _ = pipeline.load_documents(documents)

            result = pipeline.answer_question(
                question["question"],
                options=None,  # No multiple choice options
                question_id=i,
            )

            predictions.append(result)

            # No LLM judge: rely on direct comparison metrics later

            # Rate limiting
            if i < len(test_questions) - 1:  # Don't sleep after last question
                print(
                    f"   ⏳ Waiting {config.long_context_rag_sleep} seconds to avoid quota limits..."
                )
                time.sleep(config.long_context_rag_sleep)

        return {
            "pipeline_type": "Long Context RAG",
            "predictions": predictions,
            "pipeline_stats": pipeline.get_pipeline_stats(),
        }

    # Legacy placeholder retained for backward compatibility; not used now
    def _calculate_loogle_metrics(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {"deprecated": True}

    def _compare_loogle_results(
        self,
        regular_metrics: RAGMetrics,
        lc_metrics: RAGMetrics,
    ) -> Dict[str, Any]:
        """Compare Regular RAG vs Long Context RAG results using RAGMetrics."""

        reg = regular_metrics
        lc = lc_metrics

        accuracy_winner = (
            "Long Context RAG" if lc.accuracy > reg.accuracy else "Regular RAG"
        )
        if lc.accuracy == reg.accuracy:
            accuracy_winner = "Tie"

        speed_winner = (
            "Regular RAG"
            if reg.avg_response_time < lc.avg_response_time
            else "Long Context RAG"
        )
        cost_winner = (
            "Regular RAG"
            if reg.total_cost_estimate < lc.total_cost_estimate
            else "Long Context RAG"
        )

        accuracy_diff = lc.accuracy - reg.accuracy
        speed_ratio = (
            lc.avg_response_time / reg.avg_response_time
            if reg.avg_response_time > 0
            else float("inf")
        )
        cost_ratio = (
            lc.total_cost_estimate / reg.total_cost_estimate
            if reg.total_cost_estimate > 0
            else float("inf")
        )

        return {
            "pipelines": {
                "Regular RAG": reg.to_dict(),
                "Long Context RAG": lc.to_dict(),
            },
            "winner": {
                "accuracy": accuracy_winner,
                "speed": speed_winner,
                "cost": cost_winner,
                "overall": accuracy_winner,
            },
            "differences": {
                "accuracy_diff": accuracy_diff,
                "speed_ratio": speed_ratio,
                "cost_ratio": cost_ratio,
            },
        }

    def _print_summary(self, comparison: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("LOOGLE EVALUATION SUMMARY")
        print("=" * 60)

        reg_metrics = comparison["pipelines"]["Regular RAG"]
        lc_metrics = comparison["pipelines"]["Long Context RAG"]

        print("\nACCURACY:")
        print(f"  Regular RAG:      {reg_metrics['accuracy']:.1%}")
        print(f"  Long Context RAG: {lc_metrics['accuracy']:.1%}")
        print(f"  Winner: {comparison['winner']['accuracy']}")

        print("\nSUCCESS RATE:")
        print(f"  Regular RAG:      {reg_metrics['success_rate']:.1%}")
        print(f"  Long Context RAG: {lc_metrics['success_rate']:.1%}")

        print("\nSPEED:")
        print(f"  Regular RAG:      {reg_metrics['avg_response_time']:.2f}s")
        print(f"  Long Context RAG: {lc_metrics['avg_response_time']:.2f}s")
        print(f"  Winner: {comparison['winner']['speed']}")

        print("\nCOST:")
        print(f"  Regular RAG:      ${reg_metrics['total_cost_estimate']:.4f}")
        print(f"  Long Context RAG: ${lc_metrics['total_cost_estimate']:.4f}")
        print(f"  Winner: {comparison['winner']['cost']}")

        print(f"\nOVERALL WINNER: {comparison['winner']['overall']}")

        print("\n" + "=" * 60)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save complete results
        results_file = self.results_dir / f"loogle_rag_comparison_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save individual pipeline results
        reg_file = self.results_dir / f"loogle_regular_rag_{timestamp}.json"
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump(results["regular_rag"], f, indent=2, ensure_ascii=False)

        lc_file = self.results_dir / f"loogle_long_context_rag_{timestamp}.json"
        with open(lc_file, "w", encoding="utf-8") as f:
            json.dump(results["long_context_rag"], f, indent=2, ensure_ascii=False)

        print("\nResults saved:")
        print(f"- Complete results: {results_file}")
        print(f"- Regular RAG: {reg_file}")
        print(f"- Long Context RAG: {lc_file}")
