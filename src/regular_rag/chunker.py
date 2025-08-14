"""Document chunking for Regular RAG pipeline."""

import re
from typing import Any, Dict, List

from utils.config import config


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding and retrieval."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents.

        Args:
            documents: List of documents with 'content' field

        Returns:
            List of chunks with metadata
        """
        all_chunks = []

        for doc in documents:
            doc_chunks = self.chunk_single_document(doc)
            all_chunks.extend(doc_chunks)

        print(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def chunk_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document into smaller pieces.

        Args:
            document: Document with 'content' field and metadata

        Returns:
            List of chunks with metadata
        """
        content = document["content"]

        # Split into sentences first to preserve sentence boundaries
        sentences = self._split_into_sentences(content)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(document, chunk_text, len(chunks)))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk if it exists
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(document, chunk_text, len(chunks)))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving sentence boundaries."""
        # Simple sentence splitting - could be improved with spacy/nltk
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count."""
        if not sentences:
            return []

        overlap_sentences = []
        overlap_tokens = 0

        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence.split())
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def _create_chunk(
        self, document: Dict[str, Any], chunk_text: str, chunk_index: int
    ) -> Dict[str, Any]:
        """Create a chunk with metadata."""
        chunk = {
            "content": chunk_text,
            "chunk_index": chunk_index,
            "doc_id": document["id"],
            "token_count": len(chunk_text.split()),
        }

        # Add optional fields if they exist (for backwards compatibility)
        if "kb_id" in document:
            chunk["kb_id"] = document["kb_id"]
        if "is_relevant" in document:
            chunk["is_relevant"] = document["is_relevant"]
        if "question_id" in document:
            chunk["question_id"] = document["question_id"]

        return chunk
