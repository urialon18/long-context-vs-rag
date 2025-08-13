"""Retrieval and reranking for Regular RAG pipeline."""

from typing import Any, Dict, List

from utils.config import config
from utils.llm_client import GeminiClient

from .embedder import DocumentEmbedder
from .vector_store import ChromaVectorStore


class DocumentRetriever:
    """Handles document retrieval and reranking for Regular RAG."""

    def __init__(self, vector_store: ChromaVectorStore, embedder: DocumentEmbedder):
        """
        Initialize the retriever.

        Args:
            vector_store: ChromaDB vector store
            embedder: Document embedder for query embedding
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = GeminiClient()

    def retrieve(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Simple retrieve without query rewriting.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Dictionary containing retrieved chunks
        """
        k = k or config.retrieval_k

        query_embedding = self.embedder.embed_query(query)

        retrieved_chunks = self.vector_store.search(
            query_embedding,
            k=k * 2,  # Retrieve more for potential reranking
        )

        final_chunks = retrieved_chunks[:k]
        for chunk in final_chunks:
            chunk["rerank_score"] = chunk["similarity"]

        return {
            "chunks": final_chunks,
            "original_query": query,
            "rewritten_query": query,
            "query_rewrite_cost": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "success": True,
                "error": None,
            },
        }

    def retrieve_and_rerank(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Retrieve and rerank documents for a query with LLM-based query rewriting.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Dictionary containing retrieved chunks and query rewriting cost info
        """
        k = k or config.retrieval_k

        # Use LLM to rewrite query for better retrieval
        rewrite_result = self._rewrite_query_with_llm(query)
        rewritten_query = rewrite_result["rewritten_query"]
        query_rewrite_cost = rewrite_result["cost_info"]

        # Step 1: Generate query embedding
        query_embedding = self.embedder.embed_query(rewritten_query)

        # Step 2: Retrieve similar chunks
        retrieved_chunks = self.vector_store.search(
            query_embedding, k=k * 2
        )  # Retrieve more for reranking

        # Step 3: Rerank chunks (simple scoring)
        reranked_chunks = self._rerank_chunks(rewritten_query, retrieved_chunks)

        # Return top k results with actual query rewriting cost info
        return {
            "chunks": reranked_chunks[:k],
            "original_query": query,
            "rewritten_query": rewritten_query,
            "query_rewrite_cost": query_rewrite_cost,
        }

    def _rewrite_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Rewrite query using LLM to improve retrieval quality.

        Args:
            query: Original query

        Returns:
            Dictionary with rewritten query and cost info
        """
        rewrite_prompt = f"""You are an expert at improving search queries for document retrieval. Your task is to rewrite the given question to make it better for semantic search in a collection of science fiction short stories.

Original question: {query}

Please rewrite this question to:
1. Include relevant synonyms and related terms
2. Be more specific about what information is needed
3. Use terms that would likely appear in the relevant documents
4. Maintain the core meaning and intent

Respond with ONLY the improved query, nothing else.

Improved query:"""

        llm_response = self.llm_client.generate_response(rewrite_prompt)

        if llm_response["success"]:
            rewritten_query = llm_response["response"].strip()
        else:
            # Fallback to original query if rewriting fails
            rewritten_query = query

        return {
            "rewritten_query": rewritten_query,
            "cost_info": {
                "input_tokens": llm_response["input_tokens"],
                "output_tokens": llm_response["output_tokens"],
                "total_tokens": llm_response["total_tokens"],
                "success": llm_response["success"],
                "error": llm_response.get("error", None),
            },
        }

    def _expand_query(self, query: str) -> str:
        """
        Expand query with additional context (simple implementation).

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        # Simple query expansion - add context keywords
        expansion_terms = [
            "story",
            "character",
            "plot",
            "narrative",
            "scene",
            "chapter",
            "book",
            "novel",
            "text",
            "passage",
        ]

        # Add relevant expansion terms that aren't already in the query
        query_lower = query.lower()
        expanded_terms = [term for term in expansion_terms if term not in query_lower]

        if expanded_terms:
            # Add a few relevant terms
            expanded_query = f"{query} {' '.join(expanded_terms[:3])}"
        else:
            expanded_query = query

        return expanded_query

    def _rerank_chunks(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks using additional scoring.

        Args:
            query: Original query
            chunks: Retrieved chunks from vector search

        Returns:
            Reranked chunks
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Score each chunk
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            content_terms = set(content_lower.split())

            # Base similarity score from vector search
            base_score = chunk["similarity"]

            # Lexical overlap score
            overlap_score = (
                len(query_terms & content_terms) / len(query_terms | content_terms)
                if query_terms | content_terms
                else 0
            )

            # Length normalization (prefer chunks that aren't too short or too long)
            content_length = len(chunk["content"].split())
            length_score = (
                min(1.0, content_length / 200) * min(1.0, 400 / content_length)
                if content_length > 0
                else 0
            )

            # Combined score
            final_score = (
                0.7 * base_score  # Vector similarity
                + 0.2 * overlap_score  # Lexical overlap
                + 0.1 * length_score  # Length normalization
            )

            chunk["rerank_score"] = final_score

        # Sort by rerank score
        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_chunks
