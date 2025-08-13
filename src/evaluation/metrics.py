"""Evaluation metrics for RAG comparison."""

from dataclasses import dataclass
from typing import Any, Dict, List

from utils.config import config


@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics."""

    accuracy: float
    avg_response_time: float
    avg_tokens_input: float
    avg_tokens_output: float
    avg_tokens_total: float
    total_cost_estimate: float
    success_rate: float

    # Additional metrics
    questions_answered: int
    questions_total: int
    avg_context_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "avg_response_time": self.avg_response_time,
            "avg_tokens_input": self.avg_tokens_input,
            "avg_tokens_output": self.avg_tokens_output,
            "avg_tokens_total": self.avg_tokens_total,
            "total_cost_estimate": self.total_cost_estimate,
            "success_rate": self.success_rate,
            "questions_answered": self.questions_answered,
            "questions_total": self.questions_total,
            "avg_context_length": self.avg_context_length,
        }


class MetricsCalculator:
    """Calculator for RAG evaluation metrics."""

    def calculate_metrics(
        self, results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> RAGMetrics:
        """
        Calculate evaluation metrics from results.

        Args:
            results: List of prediction results
            ground_truth: List of ground truth answers

        Returns:
            RAGMetrics object with calculated metrics
        """
        if len(results) != len(ground_truth):
            raise ValueError("Results and ground truth must have same length")

        # Calculate accuracy
        correct_predictions = 0
        successful_predictions = 0

        total_response_time = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_context_length = 0

        for result, gt in zip(results, ground_truth):
            # Accuracy calculation
            if result["success"] and result["predicted_answer"] is not None:
                successful_predictions += 1
                if result["predicted_answer"] == gt["answer"]:
                    correct_predictions += 1

            # Timing and token metrics
            if result["success"]:
                total_response_time += result["timing"]["total_time"]
                total_input_tokens += result["tokens"]["input"]
                total_output_tokens += result["tokens"]["output"]

                # Context length (different for each pipeline type)
                if "context_length" in result:
                    total_context_length += result["context_length"]
                elif "retrieved_chunks" in result:
                    # Estimate context length for regular RAG
                    total_context_length += (
                        result["retrieved_chunks"] * 200
                    )  # Rough estimate

        # Calculate averages
        n_results = len(results)
        n_successful = successful_predictions

        accuracy = correct_predictions / n_successful if n_successful > 0 else 0.0
        success_rate = n_successful / n_results

        avg_response_time = (
            total_response_time / n_successful if n_successful > 0 else 0.0
        )
        avg_input_tokens = (
            total_input_tokens / n_successful if n_successful > 0 else 0.0
        )
        avg_output_tokens = (
            total_output_tokens / n_successful if n_successful > 0 else 0.0
        )
        avg_total_tokens = avg_input_tokens + avg_output_tokens
        avg_context_length = (
            total_context_length / n_successful if n_successful > 0 else 0.0
        )

        # Cost estimation
        input_cost = (total_input_tokens / 1_000_000) * getattr(
            config, "gemini_flash_input_cost", 0.075
        )
        output_cost = (total_output_tokens / 1_000_000) * getattr(
            config, "gemini_flash_output_cost", 0.30
        )
        total_cost = input_cost + output_cost

        return RAGMetrics(
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            avg_tokens_input=avg_input_tokens,
            avg_tokens_output=avg_output_tokens,
            avg_tokens_total=avg_total_tokens,
            total_cost_estimate=total_cost,
            success_rate=success_rate,
            questions_answered=n_successful,
            questions_total=n_results,
            avg_context_length=avg_context_length,
        )

    def compare_metrics(
        self,
        metrics1: RAGMetrics,
        metrics2: RAGMetrics,
        name1: str = "Pipeline 1",
        name2: str = "Pipeline 2",
    ) -> Dict[str, Any]:
        """
        Compare two sets of metrics.

        Args:
            metrics1: First pipeline metrics
            metrics2: Second pipeline metrics
            name1: Name of first pipeline
            name2: Name of second pipeline

        Returns:
            Comparison results
        """
        comparison = {
            "pipelines": {name1: metrics1.to_dict(), name2: metrics2.to_dict()},
            "differences": {
                "accuracy_diff": metrics2.accuracy - metrics1.accuracy,
                "speed_ratio": metrics1.avg_response_time / metrics2.avg_response_time
                if metrics2.avg_response_time > 0
                else float("inf"),
                "cost_ratio": metrics1.total_cost_estimate
                / metrics2.total_cost_estimate
                if metrics2.total_cost_estimate > 0
                else float("inf"),
                "token_ratio": metrics1.avg_tokens_total / metrics2.avg_tokens_total
                if metrics2.avg_tokens_total > 0
                else float("inf"),
            },
            "winner": {
                "accuracy": name1 if metrics1.accuracy > metrics2.accuracy else name2,
                "speed": name1
                if metrics1.avg_response_time < metrics2.avg_response_time
                else name2,
                "cost": name1
                if metrics1.total_cost_estimate < metrics2.total_cost_estimate
                else name2,
                "overall": self._determine_overall_winner(
                    metrics1, metrics2, name1, name2
                ),
            },
        }

        return comparison

    def _determine_overall_winner(
        self, metrics1: RAGMetrics, metrics2: RAGMetrics, name1: str, name2: str
    ) -> str:
        """Determine overall winner based on weighted criteria."""
        score1 = 0
        score2 = 0

        # Accuracy (weight: 0.5)
        if metrics1.accuracy > metrics2.accuracy:
            score1 += 0.5
        elif metrics2.accuracy > metrics1.accuracy:
            score2 += 0.5

        # Speed (weight: 0.2)
        if metrics1.avg_response_time < metrics2.avg_response_time:
            score1 += 0.2
        elif metrics2.avg_response_time < metrics1.avg_response_time:
            score2 += 0.2

        # Cost (weight: 0.2)
        if metrics1.total_cost_estimate < metrics2.total_cost_estimate:
            score1 += 0.2
        elif metrics2.total_cost_estimate < metrics1.total_cost_estimate:
            score2 += 0.2

        # Success rate (weight: 0.1)
        if metrics1.success_rate > metrics2.success_rate:
            score1 += 0.1
        elif metrics2.success_rate > metrics1.success_rate:
            score2 += 0.1

        if score1 > score2:
            return name1
        elif score2 > score1:
            return name2
        else:
            return "Tie"
