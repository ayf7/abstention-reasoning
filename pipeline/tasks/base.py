"""Base task class with shared functionality."""

import re
from pathlib import Path
from typing import Any
from collections import defaultdict

from pipeline.core.utils import extract_answer


class BaseTask:
    """
    Base class for tasks.

    Subclasses must implement:
        - name: str
        - create_primitives(num_puzzles, seed) -> list[dict]
        - format_prompt(primitive, template) -> list[dict]
        - check_correctness(primitive, generation) -> tuple[bool, dict]
    """

    name: str = "base"

    # === Required methods (must override) ===

    def create_primitives(self, num_puzzles: int, seed: int) -> list[dict]:
        """
        Generate raw puzzle data.

        Returns:
            List of dicts, each with at least 'index' and 'variant' fields.
        """
        raise NotImplementedError

    def format_prompt(self, primitive: dict, template: str) -> list[dict]:
        """
        Format primitive into chat messages using template.

        Returns:
            List of message dicts: [{"role": "...", "content": "..."}, ...]
        """
        raise NotImplementedError

    def check_correctness(self, primitive: dict, generation: str) -> tuple[bool, dict]:
        """
        Check if generation correctly solves the puzzle.

        Returns:
            (is_correct, metadata_dict)
        """
        raise NotImplementedError

    # === Optional methods (can override) ===

    def get_split_indices(
        self, total: int, split: str, seed: int = 42
    ) -> list[int]:
        """
        Return indices for a given split.

        Override this to define custom split logic. Default splits:
        - sft: 30% (indices 0-30%)
        - rl: 40% (indices 30-70%)
        - classifier: 20% (indices 70-90%)
        - eval: 10% (indices 90-100%)

        Args:
            total: Total number of primitives
            split: Split name (sft, rl, classifier, eval)
            seed: Random seed for shuffling

        Returns:
            List of indices belonging to this split
        """
        import random

        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

        # Default split ratios
        splits = {
            "sft": (0.0, 0.3),
            "rl_train": (0.3, 0.65),
            "rl_val": (0.65, 0.7),
            "classifier": (0.7, 0.9),
            "eval": (0.9, 1.0),
        }

        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

        start_ratio, end_ratio = splits[split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)

        return indices[start:end]

    def compute_reward(self, primitive: dict, generation: str) -> tuple[float, dict]:
        """
        Compute reward for RL training.
        Default: 1.0 if correct, 0.0 otherwise.

        Returns:
            (reward, info_dict)
        """
        is_correct, meta = self.check_correctness(primitive, generation)
        return (1.0 if is_correct else 0.0, {"correct": is_correct, **meta})

    def get_ground_truth(self, primitive: dict) -> dict:
        """
        Extract ground truth for embedding in prompts.
        Default: all fields except 'index'.
        """
        return {k: v for k, v in primitive.items() if k != "index"}

    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute aggregate metrics from evaluation results.
        Default: accuracy overall and by variant.
        """
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))

        # Group by variant
        by_variant = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            variant = r.get("variant", "unknown")
            by_variant[variant]["total"] += 1
            if r.get("correct", False):
                by_variant[variant]["correct"] += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "accuracy_by_variant": {
                v: d["correct"] / d["total"] if d["total"] > 0 else 0
                for v, d in by_variant.items()
            },
            "total": total,
            "correct": correct,
            "counts_by_variant": dict(by_variant),
        }

    # === Utility methods ===

    def extract_answer(self, generation: str) -> str | None:
        """Extract answer from generation. Override for custom parsing."""
        return extract_answer(generation)

    def load_template(self, template_path: Path | str) -> str:
        """Load template from file."""
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
