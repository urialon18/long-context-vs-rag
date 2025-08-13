#!/usr/bin/env python3
"""Test script for LooGLE RAG comparison."""

import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.loogle_evaluator import LooGLEEvaluator
from utils.config import config


def main():
    """Run LooGLE RAG evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.validate()

    logging.info("🔍 Testing LooGLE RAG Pipeline")
    logging.info("%s", "=" * 50)
    logging.info("Dataset: %s (%s)", config.dataset_name, config.dataset_subset)
    logging.info("Questions: %s", config.n_questions)
    logging.info("Max context length: %s chars", f"{config.max_context_length:,}")
    logging.info(
        "Rate limiting: %ss (Regular), %ss (Long Context)",
        config.regular_rag_sleep,
        config.long_context_rag_sleep,
    )

    try:
        evaluator = LooGLEEvaluator()

        results = evaluator.run_full_evaluation(save_results=True)
        return results

    except Exception as e:
        logging.error("❌ Error during evaluation: %s", e)
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
