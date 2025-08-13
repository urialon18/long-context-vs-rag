"""Embedding generation for Regular RAG pipeline."""

from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import config


class DocumentEmbedder:
    """Generates embeddings for documents and queries."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or config.embedding_model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(
            f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunks with 'content' field

        Returns:
            Chunks with added 'embedding' field
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract text content
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        print(f"Generated embeddings with dimension {embeddings.shape[1]}")
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding as numpy array
        """
        return self.model.encode([query], convert_to_numpy=True)[0]

    def compute_similarity(
        self, query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and chunk embeddings.

        Args:
            query_embedding: Query embedding
            chunk_embeddings: List of chunk embeddings

        Returns:
            Similarity scores
        """
        chunk_embeddings_matrix = np.array(chunk_embeddings)

        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings_matrix, query_embedding) / (
            np.linalg.norm(chunk_embeddings_matrix, axis=1)
            * np.linalg.norm(query_embedding)
        )

        return similarities
