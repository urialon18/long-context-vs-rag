"""Configuration management for the RAG comparison project."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG comparison experiment."""

    # API Configuration
    google_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configuration
    llm_model: str = "gemini-2.5-flash"  # Gemini 2.5 Flash
    llm_temperature: float = 0.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_k: int = 5  # Number of chunks to retrieve

    # Experiment Configuration
    n_questions: int = 2  # Reduce to 1 question for debugging
    random_seed: int = 42

    # Dataset Configuration
    dataset_name: str = "bigai-nlco/LooGLE"
    dataset_subset: str = "longdep_qa"
    dataset_split: str = "test"

    # LooGLE specific - context length limits to stay under 240K chars
    max_context_length: int = 240000  # characters

    # Rate limiting for API calls
    regular_rag_sleep: int = 10  # seconds between calls
    long_context_rag_sleep: int = 20  # seconds between calls (reduced for faster runs)

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")


# Global configuration instance
config = Config()
