"""Competition Math task implementation.

Uses the MATH dataset (competition_math) from HuggingFace with 12,500 competition
mathematics problems across 7 categories and 5 difficulty levels.

Dataset: https://huggingface.co/datasets/qwedsacf/competition_math

Supports multiple hint systems via hint_source config:
- None (default): prefix_hints (5-hint progressive system)
- "solution_hints": 3-hint solution decomposition
- "mcq_hints": 3-hint multiple-choice narrowing
"""

import json
import re

from pipeline.tasks.base import BaseTask


class CompetitionMathTask(BaseTask):
    """
    Competition Math reasoning task.

    Problems span 7 categories: Algebra, Counting & Probability, Geometry,
    Intermediate Algebra, Number Theory, Prealgebra, Precalculus.

    Difficulty levels: Level 1 (easiest) to Level 5 (hardest).
    """

    name = "competition_math"

    # Set by method config via hint_source field; determines which hint data
    # to map into hint_exprs for RL rollouts.
    # None = use prefix_hints (6-hint system), "solution_hints" = 3-hint from
    # solution splits, "mcq_hints" = 3-hint MCQ choices.
    hint_source = None

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    assistant_prefix = "<think>\nLet me work through this problem step by step."

    # Only include harder problem types (exclude Algebra, Prealgebra, Precalculus)
    ALLOWED_TYPES = {
        "Intermediate Algebra",
        "Geometry",
        "Number Theory",
        "Counting & Probability",
    }

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """
        Load competition math problems from HuggingFace, balanced across types.

        Args:
            num_puzzles: Number of problems to load (None = all matching filter)
            seed: Random seed for shuffling
        """
        import random
        from datasets import load_dataset

        rng = random.Random(seed)

        ds = load_dataset("qwedsacf/competition_math", split="train")

        # Group by type
        by_type: dict[str, list] = {t: [] for t in self.ALLOWED_TYPES}
        for row in ds:
            if row["type"] in self.ALLOWED_TYPES:
                by_type[row["type"]].append(row)

        # Shuffle each type
        for t in by_type:
            rng.shuffle(by_type[t])

        # Determine per-type limit for balanced sampling
        if num_puzzles is not None:
            per_type = num_puzzles // len(self.ALLOWED_TYPES)
        else:
            per_type = None

        # Sample from each type (balanced)
        selected = []
        for t in self.ALLOWED_TYPES:
            pool = by_type[t]
            if per_type is not None:
                pool = pool[:per_type]
            selected.extend(pool)

        # Final shuffle
        rng.shuffle(selected)

        primitives = []
        for idx, row in enumerate(selected):
            # Extract answer from solution's \boxed{} format
            answer = self._extract_boxed_answer(row["solution"])

            primitives.append({
                "index": idx,
                "variant": row["type"],  # Problem category
                "level": row["level"],   # Difficulty level (Level 1-5)
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
            })

        return primitives

    def _extract_boxed_answer(self, solution: str) -> str | None:
        """Extract the answer from \\boxed{...} in the solution."""
        # Handle nested braces by finding matching closing brace
        match = re.search(r'\\boxed\{', solution)
        if not match:
            return None

        start = match.end()
        depth = 1
        pos = start

        while pos < len(solution) and depth > 0:
            if solution[pos] == '{':
                depth += 1
            elif solution[pos] == '}':
                depth -= 1
            pos += 1

        if depth == 0:
            return solution[start:pos - 1]
        return None

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format competition math problem into chat messages."""
        content = template.replace("{problem}", primitive["problem"])
        content = content.replace("{level}", primitive["level"])
        content = content.replace("{type}", primitive["variant"])

        # Format hints if present and template uses {hints}
        if "{hints}" in content:
            hints = primitive.get("hints", [])
            if hints:
                hints_str = "\n".join(f"- {h}" for h in hints)
            else:
                hints_str = "(no hints available)"
            content = content.replace("{hints}", hints_str)

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

    def _verify_answer(self, predicted: str, correct_answer: str) -> bool:
        """Verify predicted answer against correct answer using math-verify."""
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

        # Parse gold (LaTeX from dataset) and predicted (plain symbolic from model)
        try:
            gold_parsed = parse(
                f"${correct_answer}$",
                extraction_config=[LatexExtractionConfig()],
            )
        except Exception:
            gold_parsed = []

        try:
            pred_parsed = parse(
                f"${predicted}$",
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )
        except Exception:
            pred_parsed = []

        # Try verification if both parsed successfully
        is_correct = False
        if gold_parsed and pred_parsed:
            try:
                is_correct = verify(gold_parsed, pred_parsed)
            except Exception:
                pass

        # Fallback: try both as plain expressions (no LaTeX wrapping)
        if not is_correct:
            try:
                gold_plain = parse(
                    correct_answer,
                    extraction_config=[ExprExtractionConfig()],
                )
                pred_plain = parse(
                    predicted,
                    extraction_config=[ExprExtractionConfig()],
                )
                if gold_plain and pred_plain:
                    is_correct = verify(gold_plain, pred_plain)
            except Exception:
                pass

        return is_correct

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated answer matches the correct answer.

        Handles three formats:
        - abstention_commit: <answer>...</answer> followed by <commit> or <abstain>
        - simple_abstention: ends with </think>\\n\\n<abstain>
        - standard: <answer>...</answer>

        Uses math-verify for robust symbolic equivalence checking.
        Gold answers are parsed as LaTeX, predicted answers as plain expressions.
        """
        # Detect abstention_commit format: <answer>...</answer>\n<commit> or <abstain>
        has_commit = bool(re.search(r'</answer>\s*<commit>', generation))
        has_abstain_after = bool(re.search(r'</answer>\s*<abstain>', generation))

        if has_commit or has_abstain_after:
            predicted = self.extract_answer(generation)
            if predicted is None:
                return False, {
                    "predicted_answer": None,
                    "committed": has_commit,
                    "abstained": has_abstain_after,
                    "error": "no_answer_tag",
                }

            predicted = predicted.strip()
            correct_answer = primitive.get("answer", "")

            if correct_answer is None:
                return False, {
                    "predicted_answer": predicted,
                    "committed": has_commit,
                    "abstained": has_abstain_after,
                    "error": "no_ground_truth",
                }

            is_correct = self._verify_answer(predicted, correct_answer)
            return is_correct, {
                "predicted_answer": predicted,
                "correct_answer": correct_answer,
                "committed": has_commit,
                "abstained": has_abstain_after,
            }

        # Check for simple abstention (ends with </think>\n\n<abstain>)
        if generation.rstrip().endswith("</think>\n\n<abstain>"):
            return False, {
                "predicted_answer": None,
                "abstained": True,
            }

        # Standard format
        predicted = self.extract_answer(generation)

        if predicted is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
            }

        predicted = predicted.strip()
        correct_answer = primitive.get("answer", "")

        if correct_answer is None:
            return False, {
                "predicted_answer": predicted,
                "correct_answer": None,
                "error": "no_ground_truth",
            }

        is_correct = self._verify_answer(predicted, correct_answer)
        return is_correct, {
            "predicted_answer": predicted,
            "correct_answer": correct_answer,
        }

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for embedding in prompts and RL interactions.

        Hint mapping depends on hint_source (set from method config):
        - None: prefix_hints (5-hint progressive system)
        - "solution_hints": 3-hint solution decomposition
        - "mcq_hints": 3-hint multiple-choice narrowing
        """
        gt = {
            "level": primitive["level"],
            "variant": primitive["variant"],
            "problem": primitive["problem"],
            "answer": primitive["answer"],
        }

        if self.hint_source == "mcq_hints":
            mcq = primitive.get("mcq_hints")
            if mcq:
                gt["hint_exprs"] = [
                    f'The answer is one of: {json.dumps(mcq["hint_1"])}',
                    f'The answer is one of: {json.dumps(mcq["hint_2"])}',
                    mcq["hint_3"],
                ]
        elif self.hint_source == "solution_hints":
            solution_hints = primitive.get("solution_hints")
            if solution_hints:
                gt["hint_exprs"] = [
                    solution_hints["hint_1"],
                    solution_hints["hint_2"],
                    solution_hints["hint_3"],
                ]
        else:
            # Default: prefix_hints (5-hint progressive system)
            if "prefix_hints" in primitive and primitive["prefix_hints"]:
                prefix_hints = primitive["prefix_hints"]
                hint_exprs = []
                for i in range(1, 7):  # hint_1 through hint_6
                    key = f"hint_{i}"
                    if key in prefix_hints:
                        hint_exprs.append(prefix_hints[key])
                gt["hint_exprs"] = hint_exprs
                gt["prefix_hints"] = prefix_hints  # Keep original for reference

        return gt

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
        - sft: 30%
        - rl_train: 35%
        - rl_val: 5%
        - classifier: 20%
        - eval: 10%
        """
        import random

        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

        splits = {
            "sft": (0.0, 0.30),
            "rl_train": (0.30, 0.65),
            "rl_val": (0.65, 0.70),
            "classifier": (0.70, 0.90),
            "eval": (0.90, 1.0),
            "eval_augmented": (0.70, 1.0),
        }

        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

        start_ratio, end_ratio = splits[split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)

        return indices[start:end]

    def filter_for_sft(
        self,
        examples: list[dict],
        include_abstained: bool = True,
        include_wrong_valid_format: bool = False,
    ) -> list[dict]:
        """
        Filter examples for SFT training.

        When include_wrong_valid_format is True, includes incorrect examples
        that used hints and gave an answer — valuable for teaching the hint
        request/response protocol.
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

        return filtered

    def _categorize_result(self, r: dict) -> str:
        """Categorize a result into: correct, abstained, incomplete, wrong.

        For abstention_commit format, committed results are correct/wrong
        (never incomplete), and abstained results are always "abstained".
        """
        metadata = r.get("metadata", {})

        # abstention_commit format: committed overrides other categorization
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
        """
        Compute competition_math-specific metrics.

        Groups by both problem type and difficulty level.
        When results use abstention_commit format, also computes precision/recall/F1.
        """
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by level
        dist_by_level = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0})
        # Track distribution by type
        dist_by_type = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0})

        for r in results:
            level = r.get("level", "unknown")
            ptype = r.get("variant", "unknown")
            category = self._categorize_result(r)

            dist_by_level[level]["count"] += 1
            dist_by_level[level][category] += 1

            dist_by_type[ptype]["count"] += 1
            dist_by_type[ptype][category] += 1

        # Compute totals
        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0}
        for v_dist in dist_by_level.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_level"] = dict(dist_by_level)
        metrics["distribution_by_type"] = dict(dist_by_type)
        metrics["distribution"] = total_dist

        # Commit/abstain precision/recall/F1 (abstention_commit format)
        has_commit_format = any(
            r.get("committed", False) or r.get("abstained", False)
            or r.get("metadata", {}).get("committed", False)
            for r in results
        )
        if has_commit_format:
            def _is_committed(r):
                return r.get("committed", False) or r.get("metadata", {}).get("committed", False)
            def _is_abstained(r):
                return r.get("abstained", False) or r.get("metadata", {}).get("abstained", False)

            committed = [r for r in results if _is_committed(r)]
            abstained = [r for r in results if _is_abstained(r)]
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

            # Per-level commit breakdown
            commit_by_level = defaultdict(lambda: {
                "commit_correct": 0, "commit_wrong": 0,
                "abstain_correct": 0, "abstain_wrong": 0,
                "incomplete": 0, "count": 0,
            })
            for r in results:
                level = r.get("level", "unknown")
                commit_by_level[level]["count"] += 1
                if _is_committed(r):
                    if r.get("correct", False):
                        commit_by_level[level]["commit_correct"] += 1
                    else:
                        commit_by_level[level]["commit_wrong"] += 1
                elif _is_abstained(r):
                    if r.get("correct", False):
                        commit_by_level[level]["abstain_correct"] += 1
                    else:
                        commit_by_level[level]["abstain_wrong"] += 1
                else:
                    commit_by_level[level]["incomplete"] += 1

            metrics["commit_by_level"] = dict(commit_by_level)

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """Format competition_math metrics as tables by level and type."""
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # By difficulty level
        lines.append("By Difficulty Level:")
        lines.append(f"{'Level':<12} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 65)

        dist_by_level = metrics.get("distribution_by_level", {})
        for level in sorted(dist_by_level.keys()):
            d = dist_by_level[level]
            lines.append(
                f"{level:<12} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['abstained']:>11} {d['wrong']:>8}"
            )

        lines.append("")

        # By problem type
        lines.append("By Problem Type:")
        lines.append(f"{'Type':<24} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 77)

        dist_by_type = metrics.get("distribution_by_type", {})
        for ptype in sorted(dist_by_type.keys()):
            d = dist_by_type[ptype]
            lines.append(
                f"{ptype:<24} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['abstained']:>11} {d['wrong']:>8}"
            )

        # Summary
        lines.append("")
        d = metrics.get("distribution", {})
        lines.append(f"Total: {d.get('count', 0)} | Correct: {d.get('correct', 0)} | Accuracy: {metrics.get('accuracy', 0):.1%}")

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

            # Per-level commit breakdown
            commit_by_level = metrics.get("commit_by_level")
            if commit_by_level:
                lines.extend([
                    "",
                    "Commit/Abstain by Level:",
                    f"  {'Level':<12} {'Commit+Cor':>12} {'Commit+Wrg':>12} {'Abst+Wrg':>12} {'Abst+Cor':>12} {'Incomplete':>12}",
                    "  " + "-" * 72,
                ])
                totals = {"commit_correct": 0, "commit_wrong": 0, "abstain_wrong": 0, "abstain_correct": 0, "incomplete": 0}
                for level in sorted(commit_by_level.keys()):
                    d = commit_by_level[level]
                    lines.append(
                        f"  {level:<12} {d['commit_correct']:>12} {d['commit_wrong']:>12} {d['abstain_wrong']:>12} {d['abstain_correct']:>12} {d['incomplete']:>12}"
                    )
                    for k in totals:
                        totals[k] += d[k]
                lines.append("  " + "-" * 72)
                lines.append(
                    f"  {'Total':<12} {totals['commit_correct']:>12} {totals['commit_wrong']:>12} {totals['abstain_wrong']:>12} {totals['abstain_correct']:>12} {totals['incomplete']:>12}"
                )

        return "\n".join(lines)
