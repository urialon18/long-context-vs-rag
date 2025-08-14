"""Complete Regular RAG pipeline implementation."""

import time
from typing import Any, Dict, List, Optional

from utils.config import config
from utils.llm_client import GeminiClient

from .chunker import DocumentChunker
from .embedder import DocumentEmbedder
from .retriever import DocumentRetriever
from .vector_store import ChromaVectorStore


class RegularRAGPipeline:
    """Complete Regular RAG pipeline: chunk -> embed -> store -> retrieve -> generate."""

    def __init__(self, persist_directory: Optional[str] = "chroma_db"):
        """
        Initialize the Regular RAG pipeline.

        Args:
            persist_directory: Directory to persist ChromaDB (default: "chroma_db")
        """
        self.chunker = DocumentChunker()
        self.embedder = DocumentEmbedder()
        self.vector_store = ChromaVectorStore(
            collection_name="regular_rag_chunks", persist_directory=persist_directory
        )
        self.retriever = DocumentRetriever(self.vector_store, self.embedder)
        self.llm_client = GeminiClient()

        self.is_indexed = self._check_if_indexed()

    def _check_if_indexed(self) -> bool:
        """Check if the vector store already contains indexed documents."""
        try:
            stats = self.vector_store.get_collection_stats()
            chunk_count = stats.get("total_chunks", 0)
            if chunk_count > 0:
                print(f"📋 Found existing index with {chunk_count} chunks")
                return True
            return False
        except Exception:
            return False

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index documents for retrieval. Always clears existing index to prevent duplicates.

        Args:
            documents: List of documents to index

        Returns:
            Indexing statistics
        """
        # Always clear the index to prevent duplicates and ensure fresh indexing
        if self.is_indexed:
            print("🗑️  Clearing existing index to prevent duplicates...")
            self.clear_index()

        # Check if we have the exact same documents already indexed
        # This is a more robust check but for now, we'll always re-index to be safe

        print("🔄 Starting Regular RAG indexing pipeline...")
        start_time = time.time()

        chunks = self.chunker.chunk_documents(documents)

        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        self.vector_store.add_chunks(chunks_with_embeddings)

        end_time = time.time()
        self.is_indexed = True

        total_tokens = sum(chunk["token_count"] for chunk in chunks)

        indexing_stats = {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_chunks_per_doc": len(chunks) / len(documents),
            "avg_tokens_per_chunk": total_tokens / len(chunks),
            "indexing_time": end_time - start_time,
            "chunks_per_second": len(chunks) / (end_time - start_time),
            "cached": False,
        }

        print(
            f"✅ Regular RAG indexing completed in {indexing_stats['indexing_time']:.2f}s"
        )
        print(
            f"- {indexing_stats['total_chunks']} chunks from {indexing_stats['total_documents']} documents"
        )
        print(f"- {indexing_stats['avg_chunks_per_doc']:.1f} chunks per document")
        print(f"- {indexing_stats['avg_tokens_per_chunk']:.1f} tokens per chunk")

        return indexing_stats

    def answer_question(
        self, question: str, options: Optional[List[str]] = None, question_id: int = -1
    ) -> Dict[str, Any]:
        """
        Answer a question using Regular RAG.

        Args:
            question: Question to answer
            options: List of answer options (optional, for multiple choice)
            question_id: ID of the question for evaluation

        Returns:
            Answer result with metadata
        """
        if not self.is_indexed:
            raise ValueError("Documents must be indexed before answering questions")

        start_time = time.time()

        retrieval_result = self.retriever.retrieve(question, k=config.retrieval_k)
        retrieved_chunks = retrieval_result["chunks"]
        query_rewrite_cost = retrieval_result["query_rewrite_cost"]

        context = self._create_context(retrieved_chunks)

        if options:
            prompt = self.llm_client.create_multiple_choice_prompt(
                question, options, context
            )
        else:
            prompt = self.llm_client.create_open_ended_prompt(question, context)
        llm_response = self.llm_client.generate_response(prompt)

        end_time = time.time()

        predicted_answer = self._parse_answer(llm_response["response"])

        total_input_tokens = (
            llm_response["input_tokens"] + query_rewrite_cost["input_tokens"]
        )
        total_output_tokens = (
            llm_response["output_tokens"] + query_rewrite_cost["output_tokens"]
        )
        total_tokens = total_input_tokens + total_output_tokens

        return {
            "question": question,
            "options": options,
            "predicted_answer": predicted_answer,
            "llm_response": llm_response["response"],
            "context": context,
            "retrieved_chunks": len(retrieved_chunks),
            "query_rewrite_info": {
                "original_query": retrieval_result["original_query"],
                "rewritten_query": retrieval_result["rewritten_query"],
                "cost": query_rewrite_cost,
            },
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
                "breakdown": {
                    "query_rewrite": {
                        "input": query_rewrite_cost["input_tokens"],
                        "output": query_rewrite_cost["output_tokens"],
                        "total": query_rewrite_cost["total_tokens"],
                    },
                    "answer_generation": {
                        "input": llm_response["input_tokens"],
                        "output": llm_response["output_tokens"],
                        "total": llm_response["total_tokens"],
                    },
                },
            },
            "timing": {
                "total_time": end_time - start_time,
                "llm_time": llm_response["response_time"],
            },
            "success": llm_response["success"] and query_rewrite_cost["success"],
            "error": llm_response.get("error", None)
            or query_rewrite_cost.get("error", None),
        }

    def _create_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"--- Document {i + 1} (Similarity: {chunk['similarity']:.3f}) ---"
            )
            context_parts.append(chunk["content"])
            context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    def _parse_answer(self, llm_response: str) -> Optional[int]:
        """Parse the LLM response to extract the answer choice."""
        response = llm_response.strip().lower()

        # Look for patterns like "0", "1", "2", "3" or "answer: 0", etc.
        import re

        # Try to find a number (0-3) in the response
        numbers = re.findall(r"\b([0-3])\b", response)

        if numbers:
            return int(numbers[0])

        # Try to find answer patterns
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

        # If no clear number found, return None
        return None

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "pipeline_type": "Regular RAG",
            "is_indexed": self.is_indexed,
            "vector_store": vector_stats,
            "config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retrieval_k": config.retrieval_k,
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
            },
        }

    def clear_index(self) -> None:
        """Clear the vector store index."""
        self.vector_store.clear_collection()
        self.is_indexed = False
        print("🗑️  Regular RAG index cleared")

    def force_reindex(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Force re-indexing even if index exists."""
        print("🔄 Forcing re-index...")
        self.clear_index()
        return self.index_documents(documents)
