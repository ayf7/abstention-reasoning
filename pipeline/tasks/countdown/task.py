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

        # Format hints_block (for hint_analysis: empty when no hints, block of text when hints present)
        if "{hints_block}" in content:
            hints = primitive.get("hints", [])
            if hints:
                hints_str = "Here are some hints to help you:\n" + "\n".join(f"- {h}" for h in hints) + "\n\n"
            else:
                hints_str = ""
            content = content.replace("{hints_block}", hints_str)

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
        1. Check for abstention_commit format (<answer>...<commit>/<abstain>)
        2. Check for simple abstention (<abstain> tag)
        3. Count hint requests (<request> tags)
        4. Answer can be parsed from <answer> tags
        5. Expression only uses available numbers
        6. Each number used at most once
        7. Expression evaluates to target
        """
        # Count hint requests
        num_hints = generation.count("<request>")

        # Detect abstention_commit format: <answer>...</answer>\n<commit> or <abstain>
        has_commit = bool(re.search(r'</answer>\s*<commit>', generation))
        has_abstain_after = bool(re.search(r'</answer>\s*<abstain>', generation))

        if has_commit or has_abstain_after:
            answer = self.extract_answer(generation)
            if answer is None:
                return False, {
                    "predicted_answer": None,
                    "committed": has_commit,
                    "abstained": has_abstain_after,
                    "error": "no_answer_tag",
                    "num_hints": num_hints,
                }
            is_correct, meta = self._check_expression(primitive, answer)
            meta["committed"] = has_commit
            meta["abstained"] = has_abstain_after
            meta["num_hints"] = num_hints
            return is_correct, meta

        # Check for simple abstention: rollout ends with </think>\n\n<abstain>
        if generation.rstrip().endswith("</think>\n\n<abstain>"):
            return False, {
                "predicted_answer": None,
                "abstained": True,
                "num_hints": num_hints,
            }

        answer = self.extract_answer(generation)

        if answer is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
                "abstained": False,
                "num_hints": num_hints,
            }

        is_correct, meta = self._check_expression(primitive, answer)
        meta["abstained"] = False
        meta["num_hints"] = num_hints
        return is_correct, meta

    def _check_expression(self, primitive: dict, answer: str) -> tuple[bool, dict]:
        """Validate and evaluate a countdown expression against the puzzle."""
        try:
            numbers_used = [int(n) for n in re.findall(r'\b\d+\b', answer)]
            available = primitive["numbers"].copy()

            for num in numbers_used:
                if num not in available:
                    return False, {
                        "predicted_answer": answer,
                        "error": "invalid_number",
                        "invalid_number": num,
                    }
                available.remove(num)

            result = safe_eval(answer)
            is_correct = (result == primitive["target"])

            return is_correct, {
                "predicted_answer": answer,
                "result": result,
                "target": primitive["target"],
            }

        except Exception as e:
            return False, {
                "predicted_answer": answer,
                "error": str(e),
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
        """Categorize a single result into one of: correct, abstained, incomplete, wrong.

        For abstention_commit format, committed results are correct/wrong
        (never incomplete), and abstained results are always "abstained".
        """
        metadata = r.get("metadata", {})

        if r.get("committed", False) or metadata.get("committed", False):
            return "correct" if r.get("correct", False) else "wrong"
        if r.get("abstained", False) or metadata.get("abstained", False):
            return "abstained"

        if r.get("correct", False):
            return "correct"
        elif r.get("finish_reason") == "length" or r.get("error") == "no_answer_tag":
            return "incomplete"
        else:
            return "wrong"

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute metrics including four-way outcome distribution by variant.

        When results use abstention_commit format, also computes precision/recall/F1.
        """
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

        # Commit/abstain precision/recall/F1 (abstention_commit format)
        has_commit_format = any(
            r.get("committed", False) or r.get("abstained", False)
            or r.get("metadata", {}).get("committed", False)
            for r in results
        )
        if has_commit_format:
            committed = [r for r in results if r.get("committed", False) or r.get("metadata", {}).get("committed", False)]
            abstained = [r for r in results if r.get("abstained", False) or r.get("metadata", {}).get("abstained", False)]
            committed_set = set(id(r) for r in committed)
            abstained_set = set(id(r) for r in abstained)
            incomplete = [r for r in results if id(r) not in committed_set and id(r) not in abstained_set]

            committed_correct = sum(1 for r in committed if r.get("correct", False))
            abstained_correct = sum(1 for r in abstained if r.get("correct", False))

            precision = committed_correct / len(committed) if committed else 0.0
            total_correct = committed_correct + abstained_correct
            recall = committed_correct / total_correct if total_correct > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["commit_stats"] = {
                "total": len(results),
                "committed": len(committed),
                "committed_correct": committed_correct,
                "committed_wrong": len(committed) - committed_correct,
                "abstained": len(abstained),
                "abstained_correct": abstained_correct,
                "abstained_wrong": len(abstained) - abstained_correct,
                "incomplete": len(incomplete),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            # Per-variant commit breakdown
            def _is_committed(r):
                return r.get("committed", False) or r.get("metadata", {}).get("committed", False)
            def _is_abstained(r):
                return r.get("abstained", False) or r.get("metadata", {}).get("abstained", False)

            commit_by_variant = defaultdict(lambda: {
                "commit_correct": 0, "commit_wrong": 0,
                "abstain_correct": 0, "abstain_wrong": 0,
                "incomplete": 0, "count": 0,
            })
            for r in results:
                variant = r.get("variant", "unknown")
                commit_by_variant[variant]["count"] += 1
                if _is_committed(r):
                    if r.get("correct", False):
                        commit_by_variant[variant]["commit_correct"] += 1
                    else:
                        commit_by_variant[variant]["commit_wrong"] += 1
                elif _is_abstained(r):
                    if r.get("correct", False):
                        commit_by_variant[variant]["abstain_correct"] += 1
                    else:
                        commit_by_variant[variant]["abstain_wrong"] += 1
                else:
                    commit_by_variant[variant]["incomplete"] += 1

            metrics["commit_by_variant"] = dict(commit_by_variant)

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

        # Commit/abstain stats (abstention_commit format)
        commit_stats = metrics.get("commit_stats")
        if commit_stats:
            lines.extend([
                "",
                "Commit/Abstain Analysis:",
                f"  {'':>18} {'Correct':>10} {'Wrong':>10} {'Total':>10}",
                f"  {'Committed':>18} {commit_stats['committed_correct']:>10} {commit_stats['committed_wrong']:>10} {commit_stats['committed']:>10}",
                f"  {'Abstained':>18} {commit_stats['abstained_correct']:>10} {commit_stats['abstained_wrong']:>10} {commit_stats['abstained']:>10}",
                f"  {'Incomplete':>18} {'':>10} {'':>10} {commit_stats['incomplete']:>10}",
                "",
                f"  Precision: {commit_stats['precision']:.3f}  (committed_correct / committed)",
                f"  Recall:    {commit_stats['recall']:.3f}  (committed_correct / all_correct)",
                f"  F1:        {commit_stats['f1']:.3f}",
            ])

            # Per-variant commit breakdown
            commit_by_variant = metrics.get("commit_by_variant")
            if commit_by_variant:
                lines.extend([
                    "",
                    "Commit/Abstain by Variant:",
                    f"  {'Variant':<14} {'Commit+Cor':>12} {'Commit+Wrg':>12} {'Abst+Wrg':>12} {'Abst+Cor':>12} {'Incomplete':>12}",
                    "  " + "-" * 74,
                ])
                totals = {"commit_correct": 0, "commit_wrong": 0, "abstain_wrong": 0, "abstain_correct": 0, "incomplete": 0}
                for variant in sorted(commit_by_variant.keys(), key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0):
                    d = commit_by_variant[variant]
                    label = variant.replace("_", " ")
                    lines.append(
                        f"  {label:<14} {d['commit_correct']:>12} {d['commit_wrong']:>12} {d['abstain_wrong']:>12} {d['abstain_correct']:>12} {d['incomplete']:>12}"
                    )
                    for k in totals:
                        totals[k] += d[k]
                lines.append("  " + "-" * 74)
                lines.append(
                    f"  {'Total':<14} {totals['commit_correct']:>12} {totals['commit_wrong']:>12} {totals['abstain_wrong']:>12} {totals['abstain_correct']:>12} {totals['incomplete']:>12}"
                )

        return "\n".join(lines)
