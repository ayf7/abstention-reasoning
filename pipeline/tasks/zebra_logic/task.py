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
                "content": "<think>\nLet me analyze this logic puzzle step by step.",
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

    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute zebra_logic-specific metrics.

        Includes:
        - Overall accuracy, correct, truncated, wrong counts
        - Breakdown by full variant (nxm)
        - Marginal by n (houses) and m (attributes)
        - Median token count for correct answers
        """
        from collections import defaultdict
        from statistics import median

        total = len(results)
        correct_results = [r for r in results if r.get("correct", False)]
        truncated_results = [r for r in results if r.get("finish_reason") == "length"]
        wrong_results = [r for r in results if not r.get("correct", False) and r.get("finish_reason") != "length"]

        # Median token count for correct answers
        correct_tokens = [r.get("token_count", 0) for r in correct_results]
        median_tokens = median(correct_tokens) if correct_tokens else 0

        # Count by full variant (nxm)
        by_variant = defaultdict(lambda: {"correct": 0, "truncated": 0, "wrong": 0, "total": 0})
        for r in results:
            variant = r.get("variant", "unknown")
            by_variant[variant]["total"] += 1
            if r.get("correct", False):
                by_variant[variant]["correct"] += 1
            elif r.get("finish_reason") == "length":
                by_variant[variant]["truncated"] += 1
            else:
                by_variant[variant]["wrong"] += 1

        # Marginal by n (houses) and m (attributes)
        by_houses = defaultdict(lambda: {"correct": 0, "truncated": 0, "wrong": 0, "total": 0})
        by_attributes = defaultdict(lambda: {"correct": 0, "truncated": 0, "wrong": 0, "total": 0})

        for r in results:
            variant = r.get("variant", "unknown")
            if "x" in variant:
                try:
                    n, m = variant.split("x")
                    n, m = int(n), int(m)

                    by_houses[n]["total"] += 1
                    by_attributes[m]["total"] += 1

                    if r.get("correct", False):
                        by_houses[n]["correct"] += 1
                        by_attributes[m]["correct"] += 1
                    elif r.get("finish_reason") == "length":
                        by_houses[n]["truncated"] += 1
                        by_attributes[m]["truncated"] += 1
                    else:
                        by_houses[n]["wrong"] += 1
                        by_attributes[m]["wrong"] += 1
                except ValueError:
                    pass

        # Compute accuracy for each grouping
        def add_accuracy(counts_dict):
            result = {}
            for key, counts in counts_dict.items():
                non_trunc = counts["correct"] + counts["wrong"]
                result[key] = {
                    **counts,
                    "accuracy": counts["correct"] / non_trunc if non_trunc > 0 else 0.0,
                }
            return result

        return {
            "total": total,
            "correct": len(correct_results),
            "truncated": len(truncated_results),
            "wrong": len(wrong_results),
            "accuracy": len(correct_results) / (len(correct_results) + len(wrong_results)) if (len(correct_results) + len(wrong_results)) > 0 else 0.0,
            "median_tokens_correct": median_tokens,
            "by_variant": add_accuracy(by_variant),
            "by_houses": add_accuracy(by_houses),
            "by_attributes": add_accuracy(by_attributes),
        }

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """
        Format zebra_logic metrics with nxm grid display.

        Shows:
        - Overall stats (total, correct, truncated, wrong, accuracy)
        - Marginals by houses (n) and attributes (m)
        - Full nxm grid
        """
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append(f"Total: {metrics.get('total', 0)}")
        lines.append(f"Correct: {metrics.get('correct', 0)}")
        lines.append(f"Truncated: {metrics.get('truncated', 0)}")
        lines.append(f"Wrong: {metrics.get('wrong', 0)}")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

        if "median_tokens_correct" in metrics:
            lines.append(f"Median tokens (correct): {metrics['median_tokens_correct']:.0f}")

        # Marginal by houses (n)
        if "by_houses" in metrics:
            lines.append("")
            lines.append("By houses (n):")
            sorted_keys = sorted(metrics["by_houses"].keys(), key=lambda x: int(x))
            for n in sorted_keys:
                c = metrics["by_houses"][n]
                complete = c['correct'] + c['wrong']
                lines.append(
                    f"  {n}: {c['correct']}/{complete} ({c['accuracy']:.0%}) "
                    f"+ {c['truncated']} incomplete"
                )

        # Marginal by attributes (m)
        if "by_attributes" in metrics:
            lines.append("")
            lines.append("By attributes (m):")
            sorted_keys = sorted(metrics["by_attributes"].keys(), key=lambda x: int(x))
            for m in sorted_keys:
                c = metrics["by_attributes"][m]
                complete = c['correct'] + c['wrong']
                lines.append(
                    f"  {m}: {c['correct']}/{complete} ({c['accuracy']:.0%}) "
                    f"+ {c['truncated']} incomplete"
                )

        # nxm grid
        if "by_variant" in metrics:
            lines.append(self._format_variant_grid(metrics["by_variant"]))

        return "\n".join(lines)

    def _format_variant_grid(self, by_variant: dict) -> str:
        """Format nxm variant counts as a grid."""
        # Parse variants to find grid dimensions
        grid_variants = {}
        for variant in by_variant:
            if "x" in str(variant):
                try:
                    n, m = map(int, str(variant).split("x"))
                    grid_variants[variant] = (n, m)
                except ValueError:
                    pass

        if not grid_variants:
            # Non-grid variants, return simple list
            lines = ["", "By variant:"]
            for variant, c in sorted(by_variant.items()):
                total = c.get("total", 0)
                correct = c.get("correct", 0)
                acc = correct / total if total > 0 else 0
                lines.append(f"  {variant}: {correct}/{total} ({acc:.0%})")
            return "\n".join(lines)

        # Find grid bounds
        n_values = sorted(set(n for n, m in grid_variants.values()))
        m_values = sorted(set(m for n, m in grid_variants.values()))

        # Build grid
        lines = ["", "Grid (rows=houses, cols=attributes):"]
        header = f"{'n\\m':>6}"
        for m in m_values:
            header += f"{m:>12}"
        lines.append(header)
        lines.append("-" * (6 + 12 * len(m_values)))

        for n in n_values:
            row = f"{n:>6}"
            for m in m_values:
                variant = f"{n}x{m}"
                if variant in by_variant:
                    c = by_variant[variant]
                    cell = f"{c.get('correct', 0)}/{c.get('total', 0)}"
                    row += f"{cell:>12}"
                else:
                    row += f"{'--':>12}"
            lines.append(row)

        lines.append("-" * (6 + 12 * len(m_values)))

        # Totals
        total_correct = sum(c.get("correct", 0) for c in by_variant.values())
        total = sum(c.get("total", 0) for c in by_variant.values())
        overall_acc = total_correct / total if total > 0 else 0
        lines.append(f"\nOverall: {total_correct}/{total} ({overall_acc:.0%})")

        return "\n".join(lines)
