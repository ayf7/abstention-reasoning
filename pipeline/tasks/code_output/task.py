"""Code Output task implementation.

Uses the rStar-Coder dataset (microsoft/rStar-Coder, seed_sft config) —
competitive programming problems with stdin/stdout I/O.

Each problem is expanded into up to 3 (problem, test_case) primitives with
anonymized identifiers. The task: given a Python program and stdin input,
predict the stdout output.

Dataset: https://huggingface.co/datasets/microsoft/rStar-Coder
"""

import ast
import re
import subprocess
import sys

from pipeline.tasks.base import BaseTask
from pipeline.tasks.code_output.tracer import trace_execution
from pipeline.tasks.code_output.tracer_uniform import trace_execution_uniform, count_true_steps


# Regex to extract example I/O blocks from question text.
# Handles formats like:
#   Example-----\nInput:\n...\nOutput:\n...
#   Sample Input\n...\nSample Output\n...
#   **Input**\n...\n**Output**\n...
#   Example 1\nInput:\n...\nOutput:\n...\nExample 2\n...
_EXAMPLE_IO_PATTERN = re.compile(
    r'(?:\*{0,2})(?:Example|Sample)\s*(?:\d+)?\s*(?:\*{0,2})\s*(?:-----+\s*)?\n?'
    r'[ \t]*(?:\*{0,2})[ \t]*Input[ \t]*(?:\d+)?[ \t]*(?:\*{0,2})[ \t]*[:\-]*[ \t]*\n(.*?)'
    r'\n[ \t]*(?:\*{0,2})[ \t]*(?:Sample[ \t]+)?Output[ \t]*(?:\d+)?[ \t]*(?:\*{0,2})[ \t]*[:\-]*[ \t]*\n(.*?)'
    r'(?=\n[ \t]*-----|\n[ \t]*Explanation|\n[ \t]*Note\b|\n[ \t]*(?:\*{0,2})(?:Example|Sample)|\n[ \t]*\*\*[A-Z]|$)',
    re.DOTALL | re.IGNORECASE,
)

# Minimum lines for a solution to be interesting
MIN_CODE_LINES = 10
# Maximum chars per I/O string
MAX_IO_CHARS = 500
# Maximum test cases per question
MAX_TEST_CASES = 3


def _strip_trailing_sections(text: str) -> str:
    """Strip trailing explanation/detail sections from captured I/O text.

    USACO-style problems embed sections like 'INPUT DETAILS:', 'OUTPUT DETAILS:',
    'OUTPUT FORMAT:', 'CONSTRAINTS:' etc. after the actual I/O data.
    """
    m = re.search(
        r'\n\s*(?:INPUT|OUTPUT)\s+(?:DETAILS?|FORMAT)\s*:|'
        r'\n\s*CONSTRAINTS?\s*:|'
        r'\n\s*SCORING\s*:|'
        r'\n\s*HINT\s*:',
        text, re.IGNORECASE,
    )
    if m:
        text = text[:m.start()]
    return text.strip()


def _parse_example_ios(question: str) -> list[tuple[str, str]]:
    """Parse all example I/O pairs from a question's text.

    Tries three strategies:
    1. Inline format: "Example Input ... Output ..." or "Sample Input ... Sample Output ..."
    2. Codeforces-style: "Examples" section header, then standalone "Input"/"Output" sub-headers
    3. CodeChef-style: "----- Sample Input 1 ------" / "----- Sample Output 1 ------"

    Returns list of (stdin_input, expected_stdout) tuples.
    """
    # Strategy 1: inline "Example/Sample Input ... Output ..."
    pairs = []
    for m in _EXAMPLE_IO_PATTERN.finditer(question):
        inp = _strip_trailing_sections(m.group(1))
        out = _strip_trailing_sections(m.group(2))
        if inp and out:
            pairs.append((inp, out))
    if pairs:
        return pairs

    # Strategy 2: Codeforces-style "Examples" section with standalone Input/Output headers
    # Matches "Examples\n", "-----Examples-----\n", etc.
    examples_match = re.search(r'-*\s*\bExamples?\b\s*-*\s*\n', question, re.IGNORECASE)
    if examples_match:
        section = question[examples_match.end():]

        # Stop at "Note" section
        note_match = re.search(r'\n-*\s*Note\s*-*\s*\n', section)
        if note_match:
            section = section[:note_match.start()]

        # Split by "Input" headers, then split each part by "Output"
        # Use [ \t]* instead of \s* to avoid consuming newlines + data lines
        parts = re.split(r'\n?[ \t]*Input[ \t]*(?:\d+)?[ \t]*\n', section)
        for part in parts[1:]:  # skip text before first Input
            output_split = re.split(r'\n[ \t]*Output[ \t]*(?:\d+)?[ \t]*\n', part, maxsplit=1)
            if len(output_split) == 2:
                inp = _strip_trailing_sections(output_split[0])
                out = _strip_trailing_sections(output_split[1])
                if inp and out:
                    pairs.append((inp, out))

        if pairs:
            return pairs

    # Strategy 3: CodeChef-style "----- Sample Input 1 ------" / "----- Sample Output 1 ------"
    input_parts = re.split(r'-+\s*Sample\s+Input\s*:?\s*\d*\s*-+\s*\n', question, flags=re.IGNORECASE)
    if len(input_parts) > 1:
        for part in input_parts[1:]:
            output_split = re.split(r'-+\s*Sample\s+Output\s*:?\s*\d*\s*-+\s*\n', part, maxsplit=1, flags=re.IGNORECASE)
            if len(output_split) == 2:
                inp = _strip_trailing_sections(output_split[0])
                # Stop output at next section (explanation, constraints, etc.)
                out_text = output_split[1]
                end_match = re.search(r'\n-+\s*(?:explanation|sample|constraint)', out_text, re.IGNORECASE)
                if end_match:
                    out_text = out_text[:end_match.start()]
                out = _strip_trailing_sections(out_text)
                if inp and out:
                    pairs.append((inp, out))

    return pairs


def _code_line_count(code: str) -> int:
    """Count non-empty, non-comment lines."""
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _variant_from_lines(n_lines: int) -> str:
    """Assign variant bucket by code line count."""
    if n_lines <= 20:
        return "short"
    elif n_lines <= 40:
        return "medium"
    else:
        return "long"


def _normalize_stdout(text: str) -> str:
    """Normalize stdout for comparison: strip trailing whitespace per line, strip trailing blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    # Strip trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _validate_code(code: str, stdin_input: str, expected_stdout: str, timeout: float = 10.0) -> bool:
    """Run code with stdin and check if stdout matches expected output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    if result.returncode != 0:
        return False

    return _normalize_stdout(result.stdout) == _normalize_stdout(expected_stdout)


def _process_one_primitive(item, tracer_name):
    """Process a single (question, test_case) pair. Runs in worker process."""
    qid, code, anonymized, variant, tc_idx, stdin_input, expected_stdout = item

    if not _validate_code(code, stdin_input, expected_stdout):
        return {"status": "original_fail"}

    if not _validate_code(anonymized, stdin_input, expected_stdout):
        return {"status": "anonymized_fail"}

    if tracer_name == "uniform":
        hint_exprs = trace_execution_uniform(anonymized, stdin_input)
    else:
        hint_exprs = trace_execution(anonymized, stdin_input)

    total_steps = count_true_steps(anonymized, stdin_input)

    return {
        "status": "ok",
        "primitive": {
            "index": -1,
            "variant": variant,
            "question_id": qid,
            "test_case_idx": tc_idx,
            "code": anonymized,
            "original_code": code,
            "stdin_input": stdin_input,
            "expected_stdout": expected_stdout,
            "hint_exprs": hint_exprs,
            "total_steps": total_steps,
        },
    }


class CodeOutputTask(BaseTask):
    """
    Code Output prediction task (stdin/stdout).

    Given a Python program and stdin input, predict the stdout output.
    Solutions are anonymized to prevent memorization of known problems.
    """

    name = "code_output"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    assistant_prefix = "<think>\nLet me answer this step by step."

    def create_primitives(self, num_puzzles: int | None, seed: int = 42, **kwargs) -> list[dict]:
        """
        Load rStar-Coder problems, filter, anonymize, and expand test cases.

        Streams the full dataset, keeping the shortest passed solution per
        question_id. Then parses example I/O from question text and creates
        one primitive per (question, test_case).

        Args:
            num_puzzles: Max unique questions to collect (None = all available).
                         Total primitives will be higher due to test case expansion.
            seed: Random seed
            **kwargs: Additional options. Supports:
                tracer: "original" (default) or "uniform" — which execution
                    tracer to use for generating hint_exprs.
        """
        import random
        from datasets import load_dataset
        from .anonymizer import anonymize_code

        tracer_name = kwargs.get("tracer", "original")
        if tracer_name == "uniform":
            _trace_fn = trace_execution_uniform
            print("Using uniform execution tracer for hint generation")
        else:
            _trace_fn = trace_execution

        rng = random.Random(seed)

        # Phase 1: Stream dataset, keep shortest passed solution per question_id
        print("Streaming rStar-Coder seed_sft...")
        best_per_question: dict[str, dict] = {}  # question_id -> {code, question}

        ds = load_dataset("microsoft/rStar-Coder", "seed_sft", split="train", streaming=True)
        for row in ds:
            if not row.get("is_passed", False):
                continue

            code = (row.get("code") or "").strip()
            if not code:
                continue

            # Must parse as valid Python
            try:
                ast.parse(code)
            except SyntaxError:
                continue

            qid = row.get("question_id", "")
            if not qid:
                continue

            existing = best_per_question.get(qid)
            if existing is None or len(code) < len(existing["code"]):
                best_per_question[qid] = {
                    "code": code,
                    "question": row.get("question", ""),
                }

            if num_puzzles is not None and len(best_per_question) >= num_puzzles:
                # Keep streaming to find shorter solutions for already-seen questions
                # but stop once we have enough unique questions and have seen a good sample
                if len(best_per_question) >= num_puzzles * 2:
                    break

        print(f"Collected {len(best_per_question)} unique questions with passed solutions")

        # Phase 2: Process each question — parse I/O, anonymize, filter, expand
        # Pre-filter (fast, no subprocess) then parallelize the expensive part
        work_items = []
        skipped = {"no_io": 0, "too_short": 0, "anonymize_fail": 0,
                   "original_fail": 0, "anonymized_fail": 0}
        no_io_samples = []

        for qid, entry in best_per_question.items():
            code = entry["code"]
            question = entry["question"]

            io_pairs = _parse_example_ios(question)
            if not io_pairs:
                skipped["no_io"] += 1
                if len(no_io_samples) < 50:
                    no_io_samples.append((qid, question))
                continue

            n_lines = _code_line_count(code)
            if n_lines < MIN_CODE_LINES:
                skipped["too_short"] += 1
                continue

            try:
                anonymized, _rename_map = anonymize_code(code, fn_name=None)
                ast.parse(anonymized)
            except (SyntaxError, Exception):
                skipped["anonymize_fail"] += 1
                continue

            variant = _variant_from_lines(n_lines)
            valid_pairs = [
                (inp, out) for inp, out in io_pairs
                if len(inp) <= MAX_IO_CHARS and len(out) <= MAX_IO_CHARS
            ][:MAX_TEST_CASES]

            for tc_idx, (stdin_input, expected_stdout) in enumerate(valid_pairs):
                work_items.append((qid, code, anonymized, variant, tc_idx,
                                   stdin_input, expected_stdout))

        print(f"Pre-filtered to {len(work_items)} work items "
              f"(skipped so far: {dict(skipped)})")

        # Parallel validation + tracing
        # Use ProcessPoolExecutor (non-daemonic workers) because the tracer
        # and count_true_steps spawn their own child processes.
        from concurrent.futures import ProcessPoolExecutor
        import os

        num_workers = min(os.cpu_count() or 4, len(work_items))
        print(f"Processing with {num_workers} workers...")

        trace_name = tracer_name

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_one_primitive, item, trace_name)
                for item in work_items
            ]
            results = []
            done = 0
            for future in futures:
                results.append(future.result())
                done += 1
                if done % 500 == 0:
                    print(f"  [{done}/{len(work_items)}] processed")

        primitives = []
        for result in results:
            if result is None:
                continue
            status = result.get("status")
            if status == "original_fail":
                skipped["original_fail"] += 1
            elif status == "anonymized_fail":
                skipped["anonymized_fail"] += 1
            elif status == "ok":
                primitives.append(result["primitive"])

        print(f"Skipped: {skipped}")
        print(f"Generated {len(primitives)} primitives from {len(best_per_question) - sum(skipped.values())} questions")

        # Print sample of unparsed questions for debugging
        if no_io_samples:
            rng_diag = random.Random(seed)
            rng_diag.shuffle(no_io_samples)
            n_show = min(10, len(no_io_samples))
            print(f"\n--- {n_show} sample questions where I/O parsing failed (of {skipped['no_io']}) ---")
            for qid, question in no_io_samples[:n_show]:
                tail = question[-600:] if len(question) > 600 else question
                prefix = "[...] " if len(question) > 600 else ""
                print(f"\n[{qid}]\n{prefix}{tail}")

        # Shuffle and re-index
        rng.shuffle(primitives)
        for i, p in enumerate(primitives):
            p["index"] = i

        return primitives

    @staticmethod
    def _add_line_numbers(code: str) -> str:
        """Add line numbers to code for cross-referencing with hint_exprs."""
        lines = code.splitlines()
        return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))

    def enrich_primitive_for_rl(self, primitive: dict) -> dict:
        """Pre-compute line-numbered code for verl runtime template substitution."""
        enriched = dict(primitive)
        enriched["code"] = self._add_line_numbers(primitive["code"])
        return enriched

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format code output prediction into chat messages."""
        # Add line numbers so the model can cross-reference hint_exprs
        numbered_code = self._add_line_numbers(primitive["code"])
        content = template.replace("{code}", numbered_code)
        content = content.replace("{stdin_input}", primitive["stdin_input"])

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
        Check if predicted stdout matches expected stdout.

        Uses normalized string comparison: strip trailing whitespace per line,
        strip trailing blank lines, compare line-by-line.
        """
        # Check for abstention
        if generation.rstrip().endswith("</think>\n\n<abstain>"):
            return False, {
                "predicted_answer": None,
                "abstained": True,
            }

        predicted = self.extract_answer(generation)

        if predicted is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
                "abstained": False,
            }

        expected = primitive["expected_stdout"]
        is_correct = _normalize_stdout(predicted) == _normalize_stdout(expected)

        return is_correct, {
            "predicted_answer": predicted.strip(),
            "expected_stdout": expected,
            "abstained": False,
        }

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for RL reward computation."""
        return {
            "variant": primitive["variant"],
            "stdin_input": primitive["stdin_input"],
            "expected_stdout": primitive["expected_stdout"],
            "hint_exprs": primitive.get("hint_exprs", []),
        }

    def get_split_indices(
        self,
        total: int,
        split: str,
        seed: int = 42,
        primitives: list[dict] | None = None,
    ) -> list[int]:
        """
        Custom split ratios for code_output.

        - sft: ~1000 samples (19.2%)
        - rl_train: ~2500 samples (48.0%)
        - eval: remaining (~1706 samples, 32.8%)
        """
        import random

        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

        splits = {
            "sft": (0.0, 0.192),
            "rl_train": (0.192, 0.672),
            "eval": (0.672, 1.0),
        }

        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

        start_ratio, end_ratio = splits[split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)

        return indices[start:end]

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
        """Compute metrics with distribution by variant (code length bucket)."""
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        dist_by_variant = defaultdict(
            lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0}
        )

        for r in results:
            variant = r.get("variant", "unknown")
            category = self._categorize_result(r)
            dist_by_variant[variant]["count"] += 1
            dist_by_variant[variant][category] += 1

        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0}
        for v_dist in dist_by_variant.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_variant"] = dict(dist_by_variant)
        metrics["distribution"] = total_dist

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """Format code_output metrics as a table by code length variant."""
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        lines.append(f"{'Variant':<16} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 67)

        dist_by_variant = metrics.get("distribution_by_variant", {})
        variant_order = ["short", "medium", "long"]
        for variant in [v for v in variant_order if v in dist_by_variant] + sorted(set(dist_by_variant) - set(variant_order)):
            d = dist_by_variant[variant]
            lines.append(
                f"{variant:<16} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['abstained']:>11} {d['wrong']:>8}"
            )

        lines.append("-" * 67)
        d = metrics.get("distribution", {})
        lines.append(
            f"{'Total':<16} {d.get('count', 0):>7} {d.get('correct', 0):>10} {d.get('incomplete', 0):>12} {d.get('abstained', 0):>11} {d.get('wrong', 0):>8}"
        )

        lines.append("")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

        return "\n".join(lines)
