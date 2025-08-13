"""Dataset loading for LooGLE long dependency QA dataset."""

import random
from typing import Any, Dict, List

from datasets import load_dataset

from utils.config import config


class LooGLEDatasetLoader:
    """Loader for the LooGLE longdep_qa dataset."""

    def __init__(self):
        """Initialize the dataset loader."""
        self.dataset = None
        self.test_questions = None

    def load_dataset(self) -> None:
        """Load the LooGLE dataset."""
        print(
            f"Loading {config.dataset_name} dataset (subset: {config.dataset_subset})..."
        )
        self.dataset = load_dataset(config.dataset_name, config.dataset_subset)
        print(
            f"Dataset loaded: {len(self.dataset[config.dataset_split])} examples in {config.dataset_split} split"
        )

    def create_test_questions(
        self, cache_file: str = "loogle_cache.json"
    ) -> List[Dict[str, Any]]:
        """
        Create test questions from LooGLE dataset with context length filtering.
        Uses caching to ensure same questions across runs.

        Args:
            cache_file: Path to cache file for persistence

        Returns:
            List of test questions with context, question, answer, and metadata
        """
        import os

        # Try to load from cache first
        if os.path.exists(cache_file):
            print(f"Loading cached test questions from {cache_file}")
            try:
                return self.load_test_questions(cache_file)
            except Exception as e:
                print(
                    f"Warning: Failed to load cache ({e}), creating new test questions"
                )

        # Create new test questions if cache doesn't exist or failed to load
        print("Creating new test questions and caching them...")

        if self.dataset is None:
            self.load_dataset()

        # Set random seed for reproducibility
        random.seed(config.random_seed)

        # Get test split
        test_data = self.dataset[config.dataset_split]

        # Filter examples by context length and sensitive content
        valid_examples = []
        for i, example in enumerate(test_data):
            context_length = len(example["context"])
            if (
                context_length <= config.max_context_length
                and not self._is_sensitive_content(example)
            ):
                valid_examples.append((i, example, context_length))

        # Count filtered examples for reporting
        total_examples = len(test_data)
        length_filtered = sum(
            1
            for example in test_data
            if len(example["context"]) > config.max_context_length
        )
        military_filtered = sum(
            1
            for example in test_data
            if len(example["context"]) <= config.max_context_length
            and self._is_sensitive_content(example)
        )

        print("Dataset filtering results:")
        print(f"- Original dataset size: {total_examples}")
        print(
            f"- Filtered by length (>{config.max_context_length:,} chars): {length_filtered}"
        )
        print(f"- Filtered by sensitive content: {military_filtered}")
        print(f"- Valid non-sensitive examples: {len(valid_examples)}")

        if len(valid_examples) < config.n_questions:
            raise ValueError(
                f"Not enough valid examples! Found {len(valid_examples)}, need {config.n_questions}"
            )

        # Sort by context length and sample from different ranges for diversity
        valid_examples.sort(key=lambda x: x[2])  # Sort by context length

        # Sample diverse examples across context lengths
        n_samples = config.n_questions
        sample_indices = []

        # Divide into thirds and sample from each
        third_size = len(valid_examples) // 3
        ranges = [
            (0, third_size),  # Short contexts
            (third_size, 2 * third_size),  # Medium contexts
            (2 * third_size, len(valid_examples)),  # Long contexts
        ]

        samples_per_range = n_samples // 3
        remaining_samples = n_samples % 3

        for i, (start, end) in enumerate(ranges):
            range_samples = samples_per_range + (1 if i < remaining_samples else 0)
            range_indices = list(range(start, min(end, len(valid_examples))))
            selected = random.sample(
                range_indices, min(range_samples, len(range_indices))
            )
            sample_indices.extend(selected)

        # Create test questions
        test_questions = []
        for idx in sample_indices:
            original_idx, example, context_length = valid_examples[idx]

            test_questions.append(
                {
                    "id": example["id"],
                    "doc_id": example["doc_id"],
                    "context": example["context"],
                    "question": example["question"],
                    "answer": example["answer"],
                    "evidence": example["evidence"],
                    "title": example["title"],
                    "context_length": context_length,
                    "original_index": original_idx,
                }
            )

        self.test_questions = test_questions

        # Save to cache for future runs
        self.save_test_questions(cache_file)

        print(f"Selected {len(test_questions)} test questions")
        print("Context length distribution:")
        lengths = [q["context_length"] for q in test_questions]
        print(f"  - Min: {min(lengths):,} chars")
        print(f"  - Max: {max(lengths):,} chars")
        print(f"  - Avg: {sum(lengths) // len(lengths):,} chars")
        print(f"Cached to {cache_file} for future runs")

        return test_questions

    def get_question_stats(self) -> Dict[str, Any]:
        """Get statistics about the test questions."""
        if self.test_questions is None:
            raise ValueError(
                "Test questions not created yet. Call create_test_questions() first."
            )

        context_lengths = [len(q["context"]) for q in self.test_questions]
        question_lengths = [len(q["question"]) for q in self.test_questions]
        answer_lengths = [len(q["answer"]) for q in self.test_questions]

        return {
            "total_questions": len(self.test_questions),
            "context_stats": {
                "min_chars": min(context_lengths),
                "max_chars": max(context_lengths),
                "avg_chars": sum(context_lengths) // len(context_lengths),
                "total_chars": sum(context_lengths),
                "estimated_tokens": sum(context_lengths) // 4,  # Rough estimate
            },
            "question_stats": {
                "min_chars": min(question_lengths),
                "max_chars": max(question_lengths),
                "avg_chars": sum(question_lengths) // len(question_lengths),
            },
            "answer_stats": {
                "min_chars": min(answer_lengths),
                "max_chars": max(answer_lengths),
                "avg_chars": sum(answer_lengths) // len(answer_lengths),
            },
        }

    def save_test_questions(self, filepath: str) -> None:
        """Save the test questions to a JSON file."""
        import json

        if self.test_questions is None:
            raise ValueError("Test questions not created yet.")

        data = {
            "test_questions": self.test_questions,
            "config": {
                "n_questions": config.n_questions,
                "max_context_length": config.max_context_length,
                "random_seed": config.random_seed,
                "dataset_name": config.dataset_name,
                "dataset_subset": config.dataset_subset,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Test questions saved to {filepath}")

    def load_test_questions(self, filepath: str) -> List[Dict[str, Any]]:
        """Load test questions from a JSON file."""
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Respect current config.n_questions even when loading from cache
        cached_questions = data.get("test_questions", [])
        self.test_questions = cached_questions[: config.n_questions]

        print(f"Test questions loaded from {filepath}")
        print(
            f"- {len(self.test_questions)} test questions (trimmed to current n={config.n_questions})"
        )

        return self.test_questions

    def _is_sensitive_content(self, example: Dict[str, Any]) -> bool:
        """Check if the example contains content that might trigger safety filters (military/violence)."""

        # Military/weapons keywords to filter out
        sensitive_keywords = {
            "missile",
            "missiles",
            "weapon",
            "weapons",
            "military",
            "army",
            "navy",
            "air force",
            "drdo",
            "defense",
            "defence",
            "warfare",
            "combat",
            "fighter",
            "bomber",
            "tank",
            "tanks",
            "artillery",
            "ammunition",
            "explosive",
            "explosives",
            "bomb",
            "bombs",
            "nuclear",
            "ballistic",
            "rocket",
            "rockets",
            "torpedo",
            "torpedoes",
            "radar",
            "satellite",
            "surveillance",
            "intelligence",
            "reconnaissance",
            "strategic",
            "tactical",
            "assault",
            "sniper",
            "rifle",
            "rifles",
            "gun",
            "guns",
            "pistol",
            "pistols",
            "grenade",
            "grenades",
            "warhead",
            "warheads",
            "submarine",
            "submarines",
            "destroyer",
            "destroyers",
            "frigate",
            "frigates",
            "aircraft carrier",
            "battleship",
            "battleships",
            "stealth",
            "drone",
            "drones",
            "uav",
            "cruise missile",
            "intercontinental",
            "icbm",
            "anti-aircraft",
            "anti-missile",
            "patriot",
            "aegis",
            "tomahawk",
            "hellfire",
            "javelin",
            "stinger",
            "sidewinder",
            # General violence terms
            "kill",
            "killing",
            "murder",
            "assault",
            "attack",
            "attacks",
            "explosive",
            "explosives",
        }

        # Check question, context, and answer for military keywords
        text_to_check = [
            example.get("question", ""),
            example.get("context", ""),
            example.get("answer", ""),
        ]

        for text in text_to_check:
            text_lower = text.lower()
            for keyword in sensitive_keywords:
                if keyword in text_lower:
                    return True

        return False
