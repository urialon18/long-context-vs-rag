"""Long Context RAG pipeline implementation."""

import time
from typing import Any, Dict, List, Optional

from utils.config import config
from utils.llm_client import GeminiClient


class LongContextRAGPipeline:
    """Long Context RAG pipeline: dump all documents into LLM context."""

    def __init__(self):
        """Initialize the Long Context RAG pipeline."""
        self.llm_client = GeminiClient()
        self.documents = None
        self.all_documents = None
        self.relevant_docs = None
        self.distractor_docs = None
        self.is_loaded = False

    def load_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load documents into memory and prepare smart sampling for context.

        Args:
            documents: List of documents

        Returns:
            Loading statistics
        """
        print("Loading documents for Long Context RAG...")
        start_time = time.time()

        self.documents = documents
        self.all_documents = documents

        self.relevant_docs = [doc for doc in documents if doc.get("is_relevant", True)]
        self.distractor_docs = []

        self.is_loaded = True
        end_time = time.time()

        if len(documents) == 1:
            sample_size = 1
            sample_tokens = len(documents[0]["content"].split()) * 1.3
        else:
            sample_size = len(self.relevant_docs)
            sample_tokens = sum(
                len(doc["content"].split()) * 1.3 for doc in self.relevant_docs
            )

        loading_stats = {
            "total_documents_available": len(documents),
            "relevant_documents": len(self.relevant_docs),
            "distractor_documents": len(self.distractor_docs),
            "sample_size_per_query": sample_size,
            "estimated_tokens_per_query": int(sample_tokens),
            "loading_time": end_time - start_time,
            "context_utilization": min(1.0, sample_tokens / 2000000),
        }

        print(
            f"Long Context RAG documents loaded in {loading_stats['loading_time']:.2f}s"
        )
        print(
            f"- {loading_stats['total_documents_available']} total documents available"
        )
        print(f"- {loading_stats['relevant_documents']} relevant documents")
        if len(self.documents) != 1:
            print(f"- Will use {sample_size} documents per query")
        print(
            f"- ~{loading_stats['estimated_tokens_per_query']:,} estimated tokens per query"
        )
        print(
            f"- {loading_stats['context_utilization']:.1%} of 2M context window utilized per query"
        )

        return loading_stats

    def answer_question(
        self, question: str, options: Optional[List[str]] = None, question_id: int = -1
    ) -> Dict[str, Any]:
        """
        Answer a question using Long Context RAG.

        Args:
            question: Question to answer
            options: List of answer options (optional, for multiple choice)
            question_id: ID of the question for evaluation

        Returns:
            Answer result with metadata
        """
        if not self.is_loaded:
            raise ValueError("Documents must be loaded before answering questions")

        start_time = time.time()

        if len(self.documents) == 1:
            context = self.documents[0]["content"]
            sample_size = 1
        else:
            context = "\n\n".join(doc["content"] for doc in self.relevant_docs)
            sample_size = len(self.relevant_docs)

        if options:
            prompt = self.llm_client.create_multiple_choice_prompt(
                question, options, context
            )
        else:
            prompt = self.llm_client.create_open_ended_prompt(question, context)
        llm_response = self.llm_client.generate_response(prompt)

        end_time = time.time()

        predicted_answer = self._parse_answer(llm_response["response"])

        context_analysis = self._analyze_context_usage(question_id)

        return {
            "question": question,
            "options": options,
            "predicted_answer": predicted_answer,
            "llm_response": llm_response["response"],
            "context_length": len(context.split()),
            "total_documents_in_context": sample_size,
            "context_analysis": context_analysis,
            "tokens": {
                "input": llm_response["input_tokens"],
                "output": llm_response["output_tokens"],
                "total": llm_response["total_tokens"],
            },
            "timing": {
                "total_time": end_time - start_time,
                "llm_time": llm_response["response_time"],
            },
            "success": llm_response["success"],
            "error": llm_response.get("error", None),
        }

    # Smart context creation removed for LooGLE (single-doc per query)

    def _parse_answer(self, llm_response: str) -> Optional[int]:
        """Parse the LLM response to extract the answer choice."""
        response = llm_response.strip().lower()

        import re

        numbers = re.findall(r"\b([0-3])\b", response)

        if numbers:
            return int(numbers[0])

        answer_patterns = [
            r"answer[:\s]+([0-3])",
            r"option[:\s]+([0-3])",
            r"choice[:\s]+([0-3])",
            r"^([0-3])$",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return int(match.group(1))

        return None

    def _analyze_context_usage(self, question_id: int) -> Dict[str, Any]:
        """Analyze how the context was used for this question."""
        if len(self.documents) == 1:
            sample_size = 1
        else:
            sample_size = len(self.relevant_docs)
        target_relevant_docs = []

        return {
            "total_documents_in_context": sample_size,
            "relevant_documents_in_context": sample_size,
            "distractor_documents_in_context": 0,
            "target_relevant_documents": 0,
            "signal_to_noise_ratio": 1.0 if sample_size > 0 else 0.0,
            "has_target_document": False,
            "context_strategy": "single_document"
            if sample_size == 1
            else "multi_document",
            "total_documents_available": len(self.all_documents),
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        if not self.is_loaded:
            return {"pipeline_type": "Long Context RAG", "is_loaded": False}

        total_tokens = sum(
            len(doc["content"].split()) * 1.3 for doc in self.all_documents
        )

        return {
            "pipeline_type": "Long Context RAG",
            "is_loaded": self.is_loaded,
            "total_documents": len(self.all_documents),
            "estimated_total_tokens": int(total_tokens),
            "context_window_utilization": min(1.0, total_tokens / 2000000),
            "config": {
                "llm_model": config.llm_model,
                "temperature": config.llm_temperature,
                "context_window": "2M tokens",
            },
        }

    def clear_documents(self) -> None:
        """Clear loaded documents."""
        self.documents = None
        self.is_loaded = False
        print("Long Context RAG documents cleared")
