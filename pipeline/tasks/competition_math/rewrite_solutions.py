"""Annotate competition math solutions with retrieval/reasoning labels using GPT.

This script analyzes solutions to distinguish between:
- **Retrieval**: Recalling knowledge from memory (theorems, formulas, identities, etc.)
- **Reasoning**: Applying logic/computation using retrieved knowledge and problem state

The goal is to model reasoning as a series of knowledge retrieval + application steps,
making explicit where the model accesses parametric knowledge vs. performs inference.

Usage:
    python -m pipeline.tasks.competition_math.rewrite_solutions \
        --input artifacts/competition_math/primitives_raw.json \
        --output artifacts/competition_math/primitives_idea_1.json \
        --model gpt-5-mini-2025-08-07 \
        --batch-size 20 \
        --max-samples 100
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def extract_annotations(solution: str) -> dict:
    """Extract retrieval and reasoning annotations from a rewritten solution.

    Returns a dict with:
        - retrievals: list of concept names from (retrieval: <concept>) annotations
        - reasoning_count: number of (reasoning) annotations
        - solution_list: list of segments, each ending with an annotation
        - disentangled_count: number of retrieval→reasoning pairs (split sentences)
    """
    retrievals = []
    reasoning_count = 0

    # Find all annotation spans: (start, end, type, concept) positions
    annotation_spans = []

    # Pattern for **(reasoning)**
    reasoning_pattern = re.compile(r'\*\*\(reasoning\)\*\*', re.IGNORECASE)
    for match in reasoning_pattern.finditer(solution):
        annotation_spans.append((match.start(), match.end(), "reasoning", None))
        reasoning_count += 1

    # Find all retrieval annotations (handling nested parens)
    pattern_start = re.compile(r'\*\*\(retrieval:\s*', re.IGNORECASE)
    for match in pattern_start.finditer(solution):
        concept_start = match.end()

        # Find matching closing paren by counting depth
        depth = 1
        pos = concept_start
        while pos < len(solution) and depth > 0:
            if solution[pos] == '(':
                depth += 1
            elif solution[pos] == ')':
                depth -= 1
            pos += 1

        if depth == 0:
            # Check for closing **
            end_pos = pos
            if pos < len(solution) and solution[pos:pos+2] == '**':
                end_pos = pos + 2

            # Extract the concept name
            name = solution[concept_start:pos - 1]
            clean_name = name.strip().replace('**', '').replace('*', '')

            if clean_name:
                retrievals.append(clean_name)
                annotation_spans.append((match.start(), end_pos, "retrieval", clean_name))

    # Sort by start position
    annotation_spans.sort(key=lambda x: x[0])

    # Count disentangled pairs: retrieval immediately followed by reasoning
    disentangled_count = 0
    for i in range(len(annotation_spans) - 1):
        curr_type = annotation_spans[i][2]
        next_type = annotation_spans[i + 1][2]
        if curr_type == "retrieval" and next_type == "reasoning":
            disentangled_count += 1

    # Build solution_list with proper merging of back-to-back annotations
    solution_list = []
    if not annotation_spans:
        if solution.strip():
            solution_list = [solution.strip()]
    else:
        prev_end = 0
        for ann_start, ann_end, ann_type, concept in annotation_spans:
            # Get text before this annotation
            text_before = solution[prev_end:ann_start]

            # Check if there's meaningful content (not just whitespace/punctuation)
            meaningful_before = text_before.strip().rstrip('.,;:')

            if meaningful_before:
                # New segment with real content
                full_segment = solution[prev_end:ann_end].strip()
                solution_list.append(full_segment)
            else:
                # No meaningful content - merge with previous segment
                if solution_list:
                    solution_list[-1] = solution_list[-1] + text_before + solution[ann_start:ann_end]
                else:
                    full_segment = solution[prev_end:ann_end].strip()
                    if full_segment:
                        solution_list.append(full_segment)

            prev_end = ann_end

        # Handle remaining text after last annotation
        remaining = solution[prev_end:].strip()
        if remaining:
            if remaining in '.,;:' and solution_list:
                solution_list[-1] = solution_list[-1] + remaining
            else:
                solution_list.append(remaining)

    return {
        "retrievals": retrievals,
        "reasoning_count": reasoning_count,
        "solution_list": solution_list,
        "disentangled_count": disentangled_count,
    }


REWRITE_SYSTEM_PROMPT = """\
You are a mathematics instructor who analyzes solutions to distinguish knowledge retrieval from reasoning.

## Core Framework

Mathematical reasoning involves two distinct cognitive operations:
1. **Retrieval**: Recalling knowledge from memory - theorems, formulas, identities, strategies, or definitions that exist outside the immediate problem context
2. **Reasoning**: Applying logic to manipulate equations, draw inferences, or compute results using the problem state and retrieved knowledge

Your task: Rewrite the given solution to explicitly label each step as **(retrieval: <concept>)** or **(reasoning)**.

## Annotation Format

- **Retrieval steps**: Label with **(retrieval: <Concept Name>)** where <Concept Name> is the specific theorem, formula, identity, strategy, or definition being recalled
- **Reasoning steps**: Label with **(reasoning)** - these are logical deductions or computations that follow from what's already established

## What Counts as Retrieval

Retrieval occurs when the solver accesses knowledge that:
- Has a name (e.g., "AM-GM Inequality", "Completing the Square", "Vieta's Formulas")
- Is not directly stated in the problem
- Could be taught as a standalone concept
- Would appear in a textbook or reference

Examples:
- Theorems: Pythagorean Theorem, Vieta's Formulas, Fermat's Little Theorem
- Identities: Difference of Squares, Sum of Cubes, $(a+b)^2$ expansion
- Strategies: Casework, Proof by Contradiction, Telescoping
- Definitions: Perfect square, Prime number, Modular arithmetic
- Formulas: Quadratic Formula, Binomial Coefficient

## What Counts as Reasoning

Reasoning occurs when the solver:
- Substitutes values into an equation
- Simplifies or rearranges expressions
- Computes arithmetic
- Draws logical conclusions from established facts
- Applies a retrieved concept to the specific problem

## Critical Rules

1. **Every non-trivial step needs a label.** Either (retrieval: X) or (reasoning).

2. **Retrieval requires a specific name.** The concept must be identifiable.
   - GOOD: (retrieval: Difference of Squares)
   - BAD: (retrieval: algebraic identity)

3. **Reasoning is the default for computation/logic.** When in doubt, if it's not recalling a named concept, it's reasoning.

4. **Sequential application is reasoning.** After retrieving a concept, applying it step-by-step is reasoning.

5. **Preserve all LaTeX.** Keep \\boxed{}, \\frac{}, etc. unchanged.

6. **ONE label per sentence.** Each sentence should have exactly one annotation.
   - If a sentence involves BOTH retrieval AND reasoning, SPLIT it into two or more sentences so that each sentence contains one unit of retrieval or reasoning.
   - First sentence: states the retrieved concept **(retrieval: X)**
   - Second sentence: applies it to get a result **(reasoning)**

   BAD (mixed):
   "By AM-GM, $a^2 + b^2 \\ge 2ab$, so we have $a^2 + b^2 \\ge 2 \\cdot 3 = 6$ **(retrieval: AM-GM Inequality)**"

   GOOD (disentangled):
   "By the AM-GM inequality, $a^2 + b^2 \\ge 2ab$ **(retrieval: AM-GM Inequality)**. Substituting $ab = 3$, we get $a^2 + b^2 \\ge 6$ **(reasoning)**."

## Examples

### Example 1: Pure retrieval (stating a known fact)
BEFORE: "We have $a^3+b^3=(a+b)(a^2-ab+b^2)$."
AFTER: "We have $a^3+b^3=(a+b)(a^2-ab+b^2)$ **(retrieval: Sum of Cubes Identity)**."

### Example 2: Pure reasoning (substitution/computation)
BEFORE: "Since $a + b = 7$, we substitute to get $7 \\cdot (49 - 3ab) = 91$."
AFTER: "Since $a + b = 7$, we substitute to get $7 \\cdot (49 - 3ab) = 91$ **(reasoning)**."

### Example 3: Disentangling retrieval + reasoning
BEFORE: "By AM-GM, $a^2 + \\frac{3}{4a^2} \\ge 2\\sqrt{a^2 \\cdot \\frac{3}{4a^2}} = \\sqrt{3}$."
AFTER: "The AM-GM inequality states that for positive reals, $x + y \\ge 2\\sqrt{xy}$ **(retrieval: AM-GM Inequality)**. Applying this with $x = a^2$ and $y = \\frac{3}{4a^2}$, we get $a^2 + \\frac{3}{4a^2} \\ge 2\\sqrt{\\frac{3}{4}} = \\sqrt{3}$ **(reasoning)**."

### Example 4: Disentangling retrieval + reasoning
BEFORE: "We complete the square with respect to $b^2 + \\frac{b}{a}$, getting $(b + \\frac{1}{2a})^2 - \\frac{1}{4a^2}$."
AFTER: "To handle $b^2 + \\frac{b}{a}$, we use completing the square **(retrieval: Completing the Square)**. This gives $(b + \\frac{1}{2a})^2 - \\frac{1}{4a^2}$ **(reasoning)**."

### Example 5: Pure retrieval (definition/criterion)
BEFORE: "For the product to be a perfect square, all the exponents need to be even."
AFTER: "For the product to be a perfect square, all the exponents need to be even **(retrieval: Perfect Square Criterion)**."

## Output Format

Return ONLY the rewritten solution with labels. Do not include any preamble or explanation.
"""

REWRITE_USER_TEMPLATE = """\
Problem:
{problem}

Solution:
{solution}

Annotate this solution with **(retrieval: <concept>)** for knowledge retrieval steps and **(reasoning)** for logical deductions:"""


async def rewrite_solution(
    client: AsyncOpenAI,
    problem: str,
    solution: str,
    model: str = "gpt-5-mini-2025-08-07",
    max_retries: int = 3,
    semaphore: asyncio.Semaphore | None = None,
    verbose: bool = False,
) -> str | None:
    """Rewrite a single solution using GPT (async)."""
    user_content = REWRITE_USER_TEMPLATE.format(problem=problem, solution=solution)

    async def _call():
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_completion_tokens=8192,
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if verbose or not content:
                    print(f"  Response: content={repr(content)[:100]}, finish_reason={finish_reason}")
                if not content or not content.strip():
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                return content
            except Exception as e:
                print(f"  API error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        return None

    if semaphore:
        async with semaphore:
            return await _call()
    return await _call()


async def process_primitive(
    client: AsyncOpenAI,
    primitive: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
) -> dict:
    """Process a single primitive: rewrite solution and extract annotations."""
    original_solution = primitive.get("solution_original", primitive.get("solution", ""))

    rewritten = await rewrite_solution(
        client=client,
        problem=primitive["problem"],
        solution=original_solution,
        model=model,
        semaphore=semaphore,
        verbose=verbose,
    )

    new_p = primitive.copy()
    new_p["solution_original"] = original_solution
    new_p["solution_annotated"] = rewritten

    if rewritten is not None:
        annotations = extract_annotations(rewritten)
        new_p["retrievals"] = annotations["retrievals"]
        new_p["reasoning_count"] = annotations["reasoning_count"]
        new_p["solution_list"] = annotations["solution_list"]
        new_p["disentangled_count"] = annotations["disentangled_count"]
    else:
        new_p["retrievals"] = []
        new_p["reasoning_count"] = 0
        new_p["solution_list"] = []
        new_p["disentangled_count"] = 0

    # Remove redundant 'solution' field
    if "solution" in new_p:
        del new_p["solution"]

    return new_p


async def rewrite_solutions(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5-mini-2025-08-07",
    max_samples: int | None = None,
    batch_size: int = 20,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    """Rewrite all solutions in a primitives file (async with batching)."""
    # Load input
    with open(input_path, "r") as f:
        primitives = json.load(f)

    print(f"Loaded {len(primitives)} primitives from {input_path}")

    # Limit samples if requested
    if max_samples is not None:
        primitives = primitives[:max_samples]
        print(f"Processing first {max_samples} samples")

    # Load existing progress if resuming
    completed = {}
    if resume and output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        completed = {
            p["index"]: p for p in existing
            if p.get("solution_annotated")
        }
        print(f"Resuming: {len(completed)} already completed successfully")

    # Separate completed from pending
    pending = []
    results = []
    for p in primitives:
        if p["index"] in completed:
            results.append((p["index"], completed[p["index"]]))
        else:
            pending.append(p)

    print(f"Pending: {len(pending)} to process with batch_size={batch_size}")

    if not pending:
        print("Nothing to do!")
        return

    # Initialize async client
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(batch_size)

    # Process all pending items concurrently (semaphore limits concurrency)
    tasks = [
        process_primitive(client, p, model, semaphore, verbose)
        for p in pending
    ]

    processed = await tqdm_asyncio.gather(*tasks, desc="Annotating solutions")

    # Add processed results
    for p, result in zip(pending, processed):
        results.append((p["index"], result))

        # Periodic save (every batch_size items)
        if len(results) % batch_size == 0:
            _save_results(results, output_path)

    # Final save
    _save_results(results, output_path)

    # Summary
    all_results = [r for _, r in sorted(results, key=lambda x: x[0])]
    success = sum(1 for r in all_results if r.get("solution_annotated") is not None)
    total_retrievals = sum(len(r.get("retrievals", [])) for r in all_results)
    total_reasoning = sum(r.get("reasoning_count", 0) for r in all_results)
    total_disentangled = sum(r.get("disentangled_count", 0) for r in all_results)
    avg_retrievals = total_retrievals / len(all_results) if all_results else 0
    avg_reasoning = total_reasoning / len(all_results) if all_results else 0
    avg_disentangled = total_disentangled / len(all_results) if all_results else 0
    print(f"\nDone! {success}/{len(all_results)} solutions annotated successfully")
    print(f"Retrievals: {total_retrievals} total (avg {avg_retrievals:.1f} per problem)")
    print(f"Reasoning steps: {total_reasoning} total (avg {avg_reasoning:.1f} per problem)")
    print(f"Disentangled pairs: {total_disentangled} total (avg {avg_disentangled:.1f} per problem)")
    print(f"Output saved to: {output_path}")


def _save_results(results: list[tuple[int, dict]], output_path: Path) -> None:
    """Save results sorted by index."""
    sorted_results = [r for _, r in sorted(results, key=lambda x: x[0])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate competition math solutions with retrieval/reasoning labels"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("artifacts/competition_math/primitives_raw.json"),
        help="Input primitives file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("artifacts/competition_math/primitives_idea_1.json"),
        help="Output file with annotated solutions",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model to use (default: gpt-5-mini-2025-08-07)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=20,
        help="Number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing output file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose debug output",
    )

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    asyncio.run(rewrite_solutions(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
