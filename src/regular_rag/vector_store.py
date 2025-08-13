"""Vector storage using ChromaDB for Regular RAG pipeline."""

from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from utils.config import config


class ChromaVectorStore:
    """Vector store using ChromaDB for document retrieval."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        print(f"ChromaDB collection '{collection_name}' initialized")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunks with embeddings and metadata
        """
        if not chunks:
            return

        print(f"Adding {len(chunks)} chunks to vector store...")

        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk['doc_id']}_{chunk['chunk_index']}" for chunk in chunks]
        embeddings = [chunk["embedding"].tolist() for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                "doc_id": str(chunk["doc_id"]),
                "kb_id": str(chunk["kb_id"]),
                "chunk_index": chunk["chunk_index"],
                "is_relevant": chunk["is_relevant"],
                "question_id": chunk["question_id"],
                "token_count": chunk["token_count"],
            }
            metadatas.append(metadata)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))

            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        print(f"Added {len(chunks)} chunks to vector store")

    def search(
        self, query_embedding: np.ndarray, k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of similar chunks with scores
        """
        k = k or config.retrieval_k

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            chunk = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1
                - results["distances"][0][i],  # Convert distance to similarity
            }
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_dimension": None,  # ChromaDB doesn't expose this directly
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection '{self.collection_name}'")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection '{self.collection_name}'")
