"""Generate leveled hints for competition math problems using GPT.

Instead of inline annotations, this generates 4 progressive hint levels:
- Level 1: Vague direction (what area/approach to consider)
- Level 2: More specific guidance (which technique or theorem)
- Level 3: Detailed setup (how to apply the technique)
- Level 4: Near-complete setup (almost does it for you)

Usage:
    python -m pipeline.tasks.competition_math.rewrite_solutions_leveled_hints \
        --input artifacts/competition_math/primitives.json \
        --output artifacts/competition_math/primitives_leveled_hints.json \
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

from pipeline.core.method import ARTIFACTS_ROOT


HINTS_SYSTEM_PROMPT = """\
You are a mathematics tutor who creates progressive hints for competition math problems.

Your task: Given a problem and its solution, generate 4 levels of hints that progressively reveal more information. A student should be able to use these hints one at a time, only moving to the next level if they're still stuck.

## Hint Levels

**Level 1 - Direction** (1-2 sentences)
- Point toward the general area or type of approach
- Don't name specific theorems or techniques yet
- Example: "Think about how the roots of a polynomial relate to its coefficients."

**Level 2 - Technique** (1-2 sentences)
- Name the specific theorem, formula, or strategy to use
- Don't show how to apply it yet
- Example: "Use Vieta's formulas to express the sum and product of roots."

**Level 3 - Setup** (2-4 sentences)
- Show how to set up the problem using the technique
- Include the key equations or relationships
- Stop before the final calculation
- Example: "By Vieta's formulas, if c and d are roots of $x^2 - 10ax - 11b = 0$, then c + d = 10a and cd = -11b. Write similar equations for the other quadratic, then look for a way to combine them."

**Level 4 - Walkthrough** (2-4 sentences)
- Provide a nearly complete setup that makes the answer almost obvious
- Include intermediate results
- **CRITICAL:** Do NOT state the final numerical answer. End with "compute..." or "evaluate..." leaving the last arithmetic step for the student.
- Example: "From Vieta's formulas: c + d = 10a, cd = -11b, a + b = 10c, ab = -11d. Substituting d = 10a - c and b = 10c - a into the product equations and subtracting gives (a+c)(a-c) = 121(a-c). Since $a \\neq c$, we get a + c = 121. Therefore a + b + c + d = 10(a + c). Evaluate this."

## Guidelines

- Each level should be self-contained (don't say "as mentioned above")
- Hints should work for a student who hasn't seen the solution
- **Use plain ASCII and LaTeX only.** Do NOT use Unicode math symbols. Use LaTeX instead: $\\geq$, $\\leq$, $\\sqrt{}$, $\\times$, $\\neq$, $\\to$, $\\pm$, $\\cdot$, etc.

## Full Examples

### Example 1: Number Theory (Simple)
**Problem:** Find the largest integer less than 74 that leaves a remainder of 3 when divided by 7.

{
  "level_1": "Think about numbers that leave remainder 3 when divided by 7. You are looking for the largest such number below 74.",
  "level_2": "Use the form $7q+3$ for numbers with remainder 3 when divided by 7.",
  "level_3": "Let $N$ be the target number. Then $N = 7q + 3$ for some integer $q$, and $N < 74$. From $7q+3<74$ we get $7q<71$, so $q<71/7$. Since $q$ is an integer, $q \\leq \\lfloor 71/7 \\rfloor$.",
  "level_4": "The largest $N$ occurs when $q$ is as large as possible, i.e., $q = \\lfloor 71/7 \\rfloor$. Therefore $N = 7q + 3$. Compute $\\lfloor 71/7 \\rfloor$, then evaluate $7q + 3$."
}

### Example 2: Infinite Series (Complex)
**Problem:** Compute $\\sum_{n=1}^\\infty \\frac{F_{n+1}}{F_n F_{n+2}}$, where $F_n$ is the $n$th Fibonacci number with $F_1 = F_2 = 1$.

{
  "level_1": "Think about rewriting the $n$th term so that the sum telescopes. Look for a relation among $F_n$, $F_{n+1}$, and $F_{n+2}$ that causes cancellation.",
  "level_2": "Use the Fibonacci recurrence $F_{n+1} = F_{n+2} - F_n$ to rewrite the term as a difference of two fractions.",
  "level_3": "Using $F_{n+1} = F_{n+2} - F_n$, we get $\\frac{F_{n+1}}{F_n F_{n+2}} = \\frac{1}{F_n} - \\frac{1}{F_{n+2}}$. The partial sum $S_N = \\sum_{n=1}^N \\left(\\frac{1}{F_n} - \\frac{1}{F_{n+2}}\\right)$ telescopes.",
  "level_4": "The telescoping sum simplifies to $S_N = \\frac{1}{F_1} + \\frac{1}{F_2} - \\frac{1}{F_{N+1}} - \\frac{1}{F_{N+2}}$. As $N \\to \\infty$, the last two terms vanish. Therefore the sum equals $\\frac{1}{F_1} + \\frac{1}{F_2}$. Evaluate this using $F_1 = F_2 = 1$."
}

## Output Format

Return ONLY a JSON object with this exact structure (no markdown code blocks):
{"level_1": "...", "level_2": "...", "level_3": "...", "level_4": "..."}
"""

HINTS_USER_TEMPLATE = """\
Problem:
{problem}

Solution:
{solution}

Generate 4 progressive hint levels for this problem:"""


def parse_hints_response(content: str) -> dict | None:
    """Parse the JSON hints from GPT response."""
    if not content:
        return None

    # Try to extract JSON from the response
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        # Find the end of the code block
        lines = content.split("\n")
        # Skip first line (```json or ```)
        # Find closing ```
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        hints = json.loads(content)
        # Validate structure
        required_keys = {"level_1", "level_2", "level_3", "level_4"}
        if not required_keys.issubset(hints.keys()):
            print(f"  Warning: Missing keys in response. Got: {hints.keys()}")
            return None
        return hints
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON: {e}")
        print(f"  Content was: {content[:200]}...")
        return None


async def generate_hints(
    client: AsyncOpenAI,
    problem: str,
    solution: str,
    model: str = "gpt-5-mini-2025-08-07",
    max_retries: int = 3,
    semaphore: asyncio.Semaphore | None = None,
    verbose: bool = False,
) -> dict | None:
    """Generate leveled hints for a single problem (async)."""
    user_content = HINTS_USER_TEMPLATE.format(problem=problem, solution=solution)

    async def _call():
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": HINTS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_completion_tokens=8192,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if verbose or not content:
                    print(f"  Response: content={repr(content)[:100]}, finish_reason={finish_reason}")
                hints = parse_hints_response(content)
                if hints is None:
                    print(f"  Parse failed (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                return hints
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
    """Process a single primitive: generate leveled hints."""
    solution = primitive.get("solution_original", primitive.get("solution", ""))

    hints = await generate_hints(
        client=client,
        problem=primitive["problem"],
        solution=solution,
        model=model,
        semaphore=semaphore,
        verbose=verbose,
    )

    new_p = primitive.copy()

    # Store hints as structured object
    if hints is not None:
        new_p["hints"] = hints
    else:
        new_p["hints"] = None

    return new_p


async def generate_leveled_hints(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5-nano-2025-08-07",
    max_samples: int | None = None,
    batch_size: int = 20,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    """Generate leveled hints for all problems (async with batching)."""
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
            if p.get("hints") is not None
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

    # Process all pending items concurrently
    tasks = [
        process_primitive(client, p, model, semaphore, verbose)
        for p in pending
    ]

    processed = await tqdm_asyncio.gather(*tasks, desc="Generating hints")

    # Add processed results
    for p, result in zip(pending, processed):
        results.append((p["index"], result))

        # Periodic save
        if len(results) % batch_size == 0:
            _save_results(results, output_path)

    # Final save
    _save_results(results, output_path)

    # Summary
    all_results = [r for _, r in sorted(results, key=lambda x: x[0])]
    success = sum(1 for r in all_results if r.get("hints") is not None)
    print(f"\nDone! {success}/{len(all_results)} problems processed successfully")
    print(f"Output saved to: {output_path}")


def _save_results(results: list[tuple[int, dict]], output_path: Path) -> None:
    """Save results sorted by index."""
    sorted_results = [r for _, r in sorted(results, key=lambda x: x[0])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate leveled hints for competition math problems"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=ARTIFACTS_ROOT / "competition_math" / "primitives.json",
        help="Input primitives file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=ARTIFACTS_ROOT / "competition_math" / "primitives_leveled_hints.json",
        help="Output file with leveled hints",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="OpenAI model to use (default: gpt-5-nano-2025-08-07)",
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

    asyncio.run(generate_leveled_hints(
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
