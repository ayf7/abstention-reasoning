"""Competition Math task implementation.

Uses the MATH dataset (competition_math) from HuggingFace with 12,500 competition
mathematics problems across 7 categories and 5 difficulty levels.

Dataset: https://huggingface.co/datasets/qwedsacf/competition_math
"""

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

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated answer matches the correct answer.

        Handles LaTeX math expressions with some normalization.
        """
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

        # Normalize both answers for comparison
        pred_normalized = self._normalize_math(predicted)
        correct_normalized = self._normalize_math(correct_answer)

        is_correct = pred_normalized == correct_normalized

        # Numeric fallback: try evaluating as numbers if string comparison fails
        if not is_correct:
            numeric_result = self._try_numeric_equal(pred_normalized, correct_normalized)
            if numeric_result is True:
                is_correct = True

        return is_correct, {
            "predicted_answer": predicted,
            "correct_answer": correct_answer,
            "predicted_normalized": pred_normalized,
            "correct_normalized": correct_normalized,
        }

    def _normalize_math(self, expr: str) -> str:
        """Normalize math expression for comparison."""
        if not expr:
            return ""

        s = expr.strip()

        # Remove common LaTeX wrappers
        s = re.sub(r'^\$+|\$+$', '', s)
        s = re.sub(r'^\\text\{(.+)\}$', r'\1', s)
        s = re.sub(r'^\\textbf\{(.+)\}$', r'\1', s)
        s = re.sub(r'^\\mathrm\{(.+)\}$', r'\1', s)

        # Remove \boxed{...} wrapper (handle nested braces)
        boxed = re.search(r'\\boxed\{', s)
        if boxed:
            start = boxed.end()
            depth = 1
            pos = start
            while pos < len(s) and depth > 0:
                if s[pos] == '{':
                    depth += 1
                elif s[pos] == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                s = s[:boxed.start()] + s[start:pos - 1] + s[pos:]

        # Remove trailing period if present
        s = s.rstrip('.')

        # Normalize common LaTeX commands
        s = s.replace('\\left', '').replace('\\right', '')
        s = s.replace('\\!', '').replace('\\,', '').replace('\\;', '').replace('\\:', '')
        s = s.replace('\\cdot', '*').replace('\\times', '*')
        s = s.replace('\\infty', 'inf')
        s = s.replace('\\le', '<=').replace('\\ge', '>=')

        # Normalize degrees: ^\circ -> empty (just keep the number)
        s = re.sub(r'\^\\circ', '', s)
        s = re.sub(r'°', '', s)

        # Normalize sqrt: \sqrt{x} -> sqrt(x)
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
        # \sqrt x (single char, no braces)
        s = re.sub(r'\\sqrt\s*([a-zA-Z0-9])', r'sqrt(\1)', s)

        # Normalize pi
        s = re.sub(r'\\pi', 'pi', s)

        # Normalize fractions: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b} -> a/b
        # Two-brace form: \frac{num}{den}
        s = re.sub(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
        # One-brace shorthand: \frac{num}d or \frac n{den}
        s = re.sub(r'\\[dt]?frac\{([^{}]+)\}([a-zA-Z0-9])', r'(\1)/(\2)', s)
        s = re.sub(r'\\[dt]?frac([a-zA-Z0-9])\{([^{}]+)\}', r'(\1)/(\2)', s)
        # No-brace shorthand: \frac ab (single char args)
        s = re.sub(r'\\[dt]?frac([a-zA-Z0-9])([a-zA-Z0-9])', r'(\1)/(\2)', s)

        # Simplify parenthesized fractions: (a)/(b) -> a/b when args are simple
        s = re.sub(r'\(([a-zA-Z0-9.]+)\)/\(([a-zA-Z0-9.]+)\)', r'\1/\2', s)

        # Remove thousands separators: comma followed by exactly 3 digits
        while re.search(r'(\d),(\d{3})(?!\d)', s):
            s = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', s)

        # Normalize "and" to comma for lists (e.g., "1 and 2" -> "1,2")
        s = re.sub(r'\s+and\s+', ',', s, flags=re.IGNORECASE)

        # Remove any remaining LaTeX backslash commands (cleanup)
        s = re.sub(r'\\[a-zA-Z]+', '', s)

        # Remove all whitespace for final comparison
        s = re.sub(r'\s+', '', s)

        # Insert implicit multiplication: 2sqrt(3) -> 2*sqrt(3), 8pi -> 8*pi
        s = re.sub(r'(\d)(sqrt|pi)', r'\1*\2', s)
        # Also: pi*2 or sqrt(3)2 are less common but handle )digit
        s = re.sub(r'\)(\d)', r')*\1', s)

        return s.lower()

    def _try_numeric_equal(self, a: str, b: str) -> bool | None:
        """Try numeric comparison as fallback. Returns None if not possible."""
        try:
            # Safe eval for simple arithmetic: digits, /, *, +, -, ., (, ), sqrt, pi
            allowed = set('0123456789.+-*/()sqrtpi')
            for s in (a, b):
                if not all(c in allowed for c in s):
                    return None

            import math
            ns = {"sqrt": math.sqrt, "pi": math.pi}
            va = eval(a, {"__builtins__": {}}, ns)
            vb = eval(b, {"__builtins__": {}}, ns)
            return abs(va - vb) < 1e-9
        except Exception:
            return None

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for embedding in prompts and RL interactions."""
        gt = {
            "level": primitive["level"],
            "variant": primitive["variant"],
            "problem": primitive["problem"],
            "answer": primitive["answer"],
        }
        # Convert prefix_hints dict to hint_exprs list for rollout compatibility
        # The rollout code expects hint_exprs as a list
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
            return ex.get("abstained", False) or ex.get("metadata", {}).get("abstained", False)

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
        """Categorize a result into: correct, abstained, incomplete, wrong."""
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
        Compute competition_math-specific metrics.

        Groups by both problem type and difficulty level.
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

        return "\n".join(lines)
