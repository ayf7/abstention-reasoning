"""Countdown task implementation."""

import re

from pipeline.tasks.base import BaseTask
from pipeline.core.utils import safe_eval


class CountdownTask(BaseTask):
    """
    Countdown numbers game task.

    Goal: Use given numbers with +, -, *, / to reach a target value.
    Each number can only be used once.

    Supports abstention: if the model outputs <abstain>, it's tracked
    separately in metrics. The reward function determines partial credit.
    """

    name = "countdown"
    default_template_variant = "simple"

    # Default system message
    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """Generate countdown puzzles."""
        from .generator import generate_puzzles

        if num_puzzles is None:
            raise ValueError("countdown task requires --num-puzzles (it generates puzzles, not downloads)")

        return generate_puzzles(
            num_puzzles=num_puzzles,
            seed=seed,
            operand_distribution={4: 0.33, 5: 0.33, 6: 0.34},
        )

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format countdown puzzle into chat messages."""
        # Substitute variables in template
        content = template.replace("{{target}}", str(primitive["target"]))
        content = content.replace("{target}", str(primitive["target"]))
        content = content.replace("{{numbers}}", str(primitive["numbers"]))
        content = content.replace("{numbers}", str(primitive["numbers"]))

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
        Check if the generated expression correctly solves the puzzle.

        Validates:
        1. Check for abstention (<abstain> tag)
        2. Answer can be parsed from <answer> tags
        3. Expression only uses available numbers
        4. Each number used at most once
        5. Expression evaluates to target
        """
        # Check for <abstain> tag first
        if "<abstain>" in generation:
            return False, {
                "predicted_answer": None,
                "abstained": True,
            }

        answer = self.extract_answer(generation)

        if answer is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
                "abstained": False,
            }

        try:
            # Extract numbers used in expression
            numbers_used = [int(n) for n in re.findall(r'\b\d+\b', answer)]
            available = primitive["numbers"].copy()

            # Check each number is available and used at most once
            for num in numbers_used:
                if num not in available:
                    return False, {
                        "predicted_answer": answer,
                        "error": "invalid_number",
                        "invalid_number": num,
                        "abstained": False,
                    }
                available.remove(num)

            # Evaluate expression
            result = safe_eval(answer)

            # Check if result matches target
            is_correct = (result == primitive["target"])

            return is_correct, {
                "predicted_answer": answer,
                "result": result,
                "target": primitive["target"],
                "abstained": False,
            }

        except Exception as e:
            return False, {
                "predicted_answer": answer,
                "error": str(e),
                "abstained": False,
            }

    def compute_reward(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[float, dict]:
        """
        Compute reward for RL training.

        Currently binary: 1.0 if correct, 0.0 otherwise.
        Could be extended for partial credit (e.g., close to target).
        """
        is_correct, meta = self.check_correctness(primitive, generation)
        reward = 1.0 if is_correct else 0.0
        return reward, {"correct": is_correct, **meta}

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
        """Compute metrics including four-way outcome distribution by variant."""
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by variant
        dist_by_variant = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0})

        for r in results:
            variant = r.get("variant", "unknown")
            category = self._categorize_result(r)

            dist_by_variant[variant]["count"] += 1
            dist_by_variant[variant][category] += 1

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
        Format countdown metrics as a table with outcome distribution by operand count.
        """
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # Table header
        lines.append(f"{'Mode':<14} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 65)

        # By variant rows
        dist_by_variant = metrics.get("distribution_by_variant", {})
        for variant in sorted(dist_by_variant.keys(), key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0):
            d = dist_by_variant[variant]
            # Convert variant name like "4_operands" to "4 operands"
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
