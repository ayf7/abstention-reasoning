"""Filter competition math primitives to drop likely-filler prefix hints.

Background: rewrite_solutions_prefix_hints.py always splits a solution into
exactly 5 sequential hints. When the underlying solution is short (e.g. a
single-line computation), the generator has to pad that one real step into 5
micro-steps (e.g. "substitute the values", "simplify the numerator",
"simplify the denominator", "form the quotient", "evaluate it") rather than 5
genuinely distinct reasoning steps.

Manually reviewing examples across all 7 variants and all 5 levels showed
this filler problem does NOT track the problem's `variant` (an earlier
version of this filter kept only Intermediate Algebra / Geometry / Number
Theory / Counting & Probability) or its `level` -- both easy and hard
problems, and both "allowed" and "excluded" variants, contain short,
mechanical solutions that get padded the same way (e.g. Number Theory Level 1
"determine the remainder of 71 (mod 3)" is just as padded as any Algebra
example). What *does* track filler risk is the length of the solution itself:
short solutions consistently get padded, long solutions consistently get 5
genuinely distinct hints.

This module filters by a minimum solution token count (same cl100k_base
tokenizer used elsewhere in this pipeline for hint token stats) instead of
the old variant allow-list.

Usage:
    python -m pipeline.tasks.math.filter_using_heuristics \
        --input artifacts/math/primitives_prefix_hints.json \
        --output artifacts/math/primitives_filtered.json \
        --min-solution-tokens 70
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import tiktoken

from pipeline.core.method import ARTIFACTS_ROOT

# Same encoding used for the hint_stats token counts already stored on each
# primitive, so this cutoff is directly comparable to those numbers.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Chosen by manually inspecting solutions bucketed by token count: below this,
# the large majority of sampled hints were filler/padded; at or above it, the
# large majority were genuinely distinct reasoning steps. See module
# docstring for examples. Keeps ~84% of the dataset (10.5k/12.5k).
DEFAULT_MIN_SOLUTION_TOKENS = 70


def count_tokens(text: str) -> int:
    """Count tokens using the same tokenizer as hint_stats."""
    return len(_ENCODING.encode(text))


def filter_primitives(
    primitives: list[dict],
    min_solution_tokens: int = DEFAULT_MIN_SOLUTION_TOKENS,
) -> list[dict]:
    """Keep primitives whose solution is long enough to support 5 genuinely
    distinct hints, dropping ones likely to have filler/padded hints.

    Args:
        primitives: List of primitive dicts, each with a 'solution' field.
        min_solution_tokens: Minimum token count (cl100k_base) for the
            'solution' field to keep a primitive.

    Returns:
        Filtered list, preserving original order and 'index' values.
    """
    return [
        p for p in primitives
        if count_tokens(p["solution"]) >= min_solution_tokens
    ]


def summarize(primitives: list[dict], kept: list[dict]) -> str:
    """Per-variant before/after counts, as a quick sanity check after filtering."""
    before = defaultdict(int)
    after = defaultdict(int)
    for p in primitives:
        before[p.get("variant", "unknown")] += 1
    for p in kept:
        after[p.get("variant", "unknown")] += 1

    total_before, total_after = len(primitives), len(kept)
    pct = 100 * total_after / total_before if total_before else 0.0
    lines = [f"Kept {total_after}/{total_before} ({pct:.1f}%)", "", "By variant:"]
    for variant in sorted(before):
        b, a = before[variant], after.get(variant, 0)
        lines.append(f"  {variant:25s} {a:5d}/{b:5d} ({100 * a / b:.1f}%)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter competition math primitives by solution length, to drop "
            "problems likely to have filler/padded prefix hints."
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=ARTIFACTS_ROOT / "math" / "primitives_prefix_hints.json",
        help="Input primitives file (with prefix_hints already generated).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=ARTIFACTS_ROOT / "math" / "primitives_filtered.json",
        help="Output path for the filtered primitives.",
    )
    parser.add_argument(
        "--min-solution-tokens", "-t",
        type=int,
        default=DEFAULT_MIN_SOLUTION_TOKENS,
        help=f"Minimum solution token count to keep a primitive (default: {DEFAULT_MIN_SOLUTION_TOKENS}).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reassign sequential 'index' fields in the output (default: keep original indices).",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        primitives = json.load(f)
    print(f"Loaded {len(primitives)} primitives from {args.input}")

    kept = filter_primitives(primitives, args.min_solution_tokens)

    if args.reindex:
        for i, p in enumerate(kept):
            p["index"] = i

    print(summarize(primitives, kept))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"\nWrote {len(kept)} primitives -> {args.output}")


if __name__ == "__main__":
    main()
