"""ZebraLogic task implementation.

ZebraLogic is a logical reasoning benchmark with constraint satisfaction puzzles
(Zebra puzzles). This task uses the mc_mode (multiple choice) subset from HuggingFace.

Dataset: https://huggingface.co/datasets/WildEval/ZebraLogic
"""

import re

from pipeline.tasks.base import BaseTask


class ZebraLogicTask(BaseTask):
    """
    ZebraLogic multiple-choice reasoning task.

    Goal: Solve constraint satisfaction puzzles by selecting the correct answer
    from multiple choices.
    """

    name = "zebra_logic"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    # Zebra-specific assistant prefix (used in SFT prompts and RL runtime)
    assistant_prefix = "<think>\nLet me analyze this logic puzzle step by step."

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """
        Load ZebraLogic puzzles from HuggingFace.

        Unlike generated tasks, this downloads a fixed dataset.
        num_puzzles controls how many to use (None = all available).
        """
        from datasets import load_dataset

        # Load mc_mode subset
        ds = load_dataset("WildEval/ZebraLogic", name="mc_mode", split="test")

        # Shuffle with seed for reproducibility
        ds = ds.shuffle(seed=seed)

        # Limit to requested number (None means use all)
        if num_puzzles is not None and num_puzzles < len(ds):
            ds = ds.select(range(num_puzzles))

        primitives = []
        for idx, row in enumerate(ds):
            # Extract variant from ID (e.g., "lgp-test-4x4-27" -> "4x4")
            puzzle_id = row["id"]
            variant_match = re.search(r"(\d+x\d+)", puzzle_id)
            variant = variant_match.group(1) if variant_match else "unknown"

            primitives.append({
                "index": idx,
                "variant": variant,
                "puzzle_id": puzzle_id,
                "puzzle": row["puzzle"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": row["answer"],
            })

        return primitives

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format ZebraLogic puzzle into chat messages."""
        # Format choices as lettered list (A, B, C, D, ...)
        choices_formatted = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(primitive["choices"])
        )

        # Substitute variables in template
        content = template.replace("{puzzle}", primitive["puzzle"])
        content = content.replace("{question}", primitive["question"])
        content = content.replace("{choices}", choices_formatted)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]

        if include_assistant_prefix:
            messages.append({
                "role": "assistant",
                "content": self.assistant_prefix,
            })

        return messages

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated answer matches the correct choice.

        Accepts:
        - Exact match with answer text
        - Letter choice (A, B, C, D) matching the correct answer's position
        """
        predicted = self.extract_answer(generation)

        if predicted is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
            }

        predicted = predicted.strip()
        correct_answer = primitive["answer"]
        choices = primitive["choices"]

        # Find correct answer's letter
        try:
            correct_idx = choices.index(correct_answer)
            correct_letter = chr(65 + correct_idx)
        except ValueError:
            correct_letter = None

        # Normalize predicted answer
        predicted_normalized = predicted.lower().strip()

        # Check for exact match (case-insensitive)
        if predicted_normalized == correct_answer.lower():
            return True, {
                "predicted_answer": predicted,
                "correct_answer": correct_answer,
                "match_type": "exact",
            }

        # Check for letter match (e.g., "A" or "A.")
        predicted_letter = predicted.upper().strip().rstrip(".")
        if len(predicted_letter) == 1 and predicted_letter == correct_letter:
            return True, {
                "predicted_answer": predicted,
                "correct_answer": correct_answer,
                "match_type": "letter",
            }

        # Check if predicted matches any choice by letter
        if len(predicted_letter) == 1 and predicted_letter.isalpha():
            letter_idx = ord(predicted_letter) - 65
            if 0 <= letter_idx < len(choices):
                predicted_choice = choices[letter_idx]
                return False, {
                    "predicted_answer": predicted,
                    "predicted_choice": predicted_choice,
                    "correct_answer": correct_answer,
                    "error": "wrong_choice",
                }

        # Check if predicted text matches any choice
        for i, choice in enumerate(choices):
            if predicted_normalized == choice.lower():
                is_correct = (choice == correct_answer)
                return is_correct, {
                    "predicted_answer": predicted,
                    "predicted_choice": choice,
                    "correct_answer": correct_answer,
                    "match_type": "choice_text" if is_correct else None,
                    "error": None if is_correct else "wrong_choice",
                }

        return False, {
            "predicted_answer": predicted,
            "correct_answer": correct_answer,
            "error": "no_match",
        }

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for embedding in prompts."""
        return {
            "puzzle_id": primitive["puzzle_id"],
            "variant": primitive["variant"],
            "question": primitive["question"],
            "choices": primitive["choices"],
            "answer": primitive["answer"],
        }

    def get_split_indices(
        self,
        total: int,
        split: str,
        seed: int = 42,
        primitives: list[dict] | None = None,
    ) -> list[int]:
        """
        Return indices for a given split with custom ratios.

        Split ratios:
        - sft: 35%
        - rl_train: 37%
        - rl_val: 5%
        - classifier: 15%
        - eval: 8% (stratified by variant for even distribution)
        """
        import random
        from collections import defaultdict

        rng = random.Random(seed)

        # Custom split ratios
        splits = {
            "sft": (0.0, 0.35),
            "rl_train": (0.35, 0.72),
            "rl_val": (0.72, 0.77),
            "classifier": (0.77, 0.92),
            "eval": (0.92, 1.0),
        }

        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

        # For eval, use stratified sampling if primitives are provided
        if split == "eval" and primitives is not None:
            # Group indices by variant
            by_variant = defaultdict(list)
            for p in primitives:
                by_variant[p["variant"]].append(p["index"])

            # Shuffle each variant's indices
            for variant in by_variant:
                rng.shuffle(by_variant[variant])

            # Take ~8% from each variant (stratified)
            eval_indices = []
            for variant, indices in by_variant.items():
                n_take = max(1, int(len(indices) * 0.08))
                eval_indices.extend(indices[:n_take])

            rng.shuffle(eval_indices)
            return eval_indices

        # For other splits, use standard approach but exclude eval indices
        indices = list(range(total))
        rng.shuffle(indices)

        # If we need to account for stratified eval, recalculate
        if primitives is not None:
            # Get eval indices first (stratified)
            eval_indices = set(self.get_split_indices(total, "eval", seed, primitives))
            # Remove eval indices from pool
            indices = [i for i in indices if i not in eval_indices]
            # Recalculate ratios for remaining data (92% of total)
            remaining = len(indices)
            adjusted_splits = {
                "sft": (0.0, 0.35 / 0.92),
                "rl_train": (0.35 / 0.92, 0.72 / 0.92),
                "rl_val": (0.72 / 0.92, 0.77 / 0.92),
                "classifier": (0.77 / 0.92, 1.0),
            }
            start_ratio, end_ratio = adjusted_splits[split]
            start = int(remaining * start_ratio)
            end = int(remaining * end_ratio)
            return indices[start:end]

        # Fallback: simple ratio-based split
        start_ratio, end_ratio = splits[split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)
        return indices[start:end]

    def _categorize_result(self, r: dict) -> str:
        """Categorize a single result into one of: correct, abstained, incomplete, wrong."""
        is_abstained = r.get("abstained", False) or r.get("metadata", {}).get("abstained", False)

        if r.get("correct", False):
            return "correct"
        elif is_abstained:
            return "abstained"
        elif r.get("finish_reason") == "length" or r.get("error") == "no_answer_tag":
            return "incomplete"
        else:
            return "wrong"

    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute zebra_logic-specific metrics.

        Groups by number of houses (n) as the difficulty dimension.
        Uses four-way outcome distribution: correct, incomplete, abstained, wrong.
        """
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by houses (n) as the difficulty dimension
        dist_by_variant = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0})

        for r in results:
            variant = r.get("variant", "unknown")
            category = self._categorize_result(r)

            # Extract number of houses from variant (e.g., "4x5" -> 4)
            if "x" in variant:
                try:
                    n, _ = variant.split("x")
                    houses_key = f"{n}_houses"
                except ValueError:
                    houses_key = "unknown"
            else:
                houses_key = "unknown"

            dist_by_variant[houses_key]["count"] += 1
            dist_by_variant[houses_key][category] += 1

        # Compute totals
        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0}
        for v_dist in dist_by_variant.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_variant"] = dict(dist_by_variant)
        metrics["distribution"] = total_dist

        # Keep legacy fields for backwards compatibility
        metrics["abstained"] = total_dist["abstained"]
        metrics["abstention_rate"] = total_dist["abstained"] / total_dist["count"] if total_dist["count"] else 0

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """
        Format zebra_logic metrics as a table with outcome distribution by houses.
        """
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # Table header
        lines.append(f"{'Houses':<14} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 65)

        # By variant rows (sorted by number of houses)
        dist_by_variant = metrics.get("distribution_by_variant", {})
        sorted_variants = sorted(
            dist_by_variant.keys(),
            key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 999
        )

        for variant in sorted_variants:
            d = dist_by_variant[variant]
            # Convert variant name like "4_houses" to "4 houses"
            label = variant.replace("_", " ")
            lines.append(
                f"{label:<14} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['abstained']:>11} {d['wrong']:>8}"
            )

        # Total row
        lines.append("-" * 65)
        d = metrics.get("distribution", {})
        lines.append(
            f"{'Total':<14} {d.get('count', 0):>7} {d.get('correct', 0):>10} {d.get('incomplete', 0):>12} {d.get('abstained', 0):>11} {d.get('wrong', 0):>8}"
        )

        # Accuracy summary
        lines.append("")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

        return "\n".join(lines)
