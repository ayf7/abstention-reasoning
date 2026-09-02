"""Countdown task implementation."""

import importlib.util
import re
from pathlib import Path

from pipeline.tasks.base import BaseTask
from pipeline.core.utils import safe_eval

# The nested tag grammar is defined once, in verl/recipe/shared, so the SFT
# filter below and the reward function verl scores rollouts with cannot drift
# apart. Loaded by path because recipes are standalone files, not a package.
_GRAMMAR_PATH = (Path(__file__).resolve().parents[3]
                 / "verl" / "recipe" / "shared" / "nested_grammar.py")
_spec = importlib.util.spec_from_file_location("nested_grammar", _GRAMMAR_PATH)
_grammar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grammar)
_has_malformed_structure_nested = _grammar.has_malformed_structure_nested


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

        # Detect cross-verification format: </verify><answer>correct/incorrect</answer>
        # Note: opening <verify> is the assistant prefix and not in generation text
        # Distinguished from abstention_verify by: no <commit>/<abstain> tags
        verify_verdict = re.search(
            r'</verify>\s*<answer>\s*(correct|incorrect)\s*</answer>',
            generation, re.DOTALL | re.IGNORECASE,
        )
        if verify_verdict and not re.search(r'<commit>|<abstain>', generation):
            verdict = verify_verdict.group(1).lower()
            generation_correct = primitive.get("generation_correct", False)
            is_correct = (verdict == "correct") == generation_correct
            return is_correct, {
                "verdict": verdict,
                "generation_correct": generation_correct,
                "verify_format": True,
                "num_hints": num_hints,
            }

        # Detect abstention_verify format: </verify>\s*<commit> or <abstain>
        has_verify_commit = bool(re.search(r'</verify>\s*<commit>', generation))
        has_verify_abstain = bool(re.search(r'</verify>\s*<abstain>', generation))
        has_verify_tag = bool(re.search(r'<verify>', generation))

        if has_verify_commit or has_verify_abstain or has_verify_tag:
            answer = self.extract_answer(generation)
            if answer is None:
                return False, {
                    "predicted_answer": None,
                    "committed": has_verify_commit,
                    "abstained": has_verify_abstain,
                    "error": "no_answer_tag",
                    "num_hints": num_hints,
                }
            is_correct, meta = self._check_expression(primitive, answer)
            meta["committed"] = has_verify_commit
            meta["abstained"] = has_verify_abstain
            meta["num_hints"] = num_hints
            return is_correct, meta

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

            if available:
                return False, {
                    "predicted_answer": answer,
                    "error": "unused_numbers",
                    "unused_numbers": available,
                }

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

    def filter_for_sft(
        self,
        examples: list[dict],
        include_abstained: bool = True,
        include_wrong_valid_format: bool = False,
        nested_request: bool = False,
    ) -> list[dict]:
        """
        Filter examples for SFT training.

        When include_wrong_valid_format is True, includes incorrect examples
        that used hints and gave an answer — valuable for teaching the hint
        request/response protocol.

        nested_request additionally drops generations that are correct but do
        not match the inline-request grammar. check_correctness only parses the
        <answer> expression, so it says nothing about tag structure: in the 14B
        run it kept 156 of 1106 correct generations that closed </think> before
        requesting a hint. Countdown produces far more of these than math does
        (39% of asking generations against 2%) — its ask moment is a give-up
        point in a long search, where the habit of ending reasoning with
        </think> fires — so training on them would teach back the very pattern
        the nested format exists to remove.
        """
        def is_abstained(ex):
            return (ex.get("abstained", False)
                    or ex.get("metadata", {}).get("abstained", False)
                    or "</think>\n\n<abstain>" in ex.get("generation", ""))

        def has_hints_and_answer(ex):
            gen = ex.get("generation", "")
            return "<request></request>" in gen and "<answer>" in gen

        filtered = []
        for ex in examples:
            if ex.get("correct", False):
                filtered.append(ex)
            elif include_abstained and is_abstained(ex):
                filtered.append(ex)
            elif include_wrong_valid_format and has_hints_and_answer(ex):
                filtered.append(ex)

        if nested_request:
            filtered = [ex for ex in filtered
                        if not _has_malformed_structure_nested(ex.get("generation", ""))]

        return filtered

    def _categorize_result(self, r: dict) -> str:
        """Categorize a single result into one of: correct, abstained, incomplete, wrong.

        For commit/abstain formats, committed results are correct/wrong,
        abstained results are always "abstained", and anything else is "incomplete".
        For standard format (no commit/abstain fields), uses correctness directly.
        """
        metadata = r.get("metadata", {})

        has_committed = r.get("committed", False) or metadata.get("committed", False)
        has_abstained = r.get("abstained", False) or metadata.get("abstained", False)

        if has_committed:
            return "correct" if r.get("correct", False) else "wrong"
        if has_abstained:
            return "abstained"

        # If commit/abstain format (committed key exists) but neither committed nor abstained → incomplete
        if "committed" in r or "committed" in metadata:
            return "incomplete"

        # Standard format (no commit/abstain fields)
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

            # Accuracy = committed_correct / total (not raw correctness)
            metrics["accuracy"] = committed_correct / len(results) if results else 0.0
            metrics["correct"] = committed_correct

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

        # Summary line
        total_count = metrics.get("total", 0)
        commit_stats = metrics.get("commit_stats")

        if commit_stats:
            raw_correct = commit_stats["committed_correct"] + commit_stats["abstained_correct"]
            answeracc = raw_correct / total_count if total_count else 0
            abstention = commit_stats["abstained"] / total_count if total_count else 0
            non_abstained = total_count - commit_stats["abstained"]
            precision = commit_stats["committed_correct"] / non_abstained if non_abstained else 0
            raw_acc = raw_correct / total_count if total_count else 0
            lines.append("")
            lines.append(f"RawAcc: {raw_acc:.2%} ({raw_correct}/{total_count})  AnswerAcc: {answeracc:.2%}  Abstention: {abstention:.2%}  Precision: {precision:.2%}")
        else:
            lines.append("")
            lines.append(f"Accuracy: {metrics.get('accuracy', 0):.2%}")

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
