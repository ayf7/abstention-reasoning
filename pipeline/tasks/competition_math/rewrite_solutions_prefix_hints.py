"""Generate incremental solution hints for competition math problems using GPT.

This script divides solutions into 5 sequential segments (each 1-2 sentences):
- hint_1: First step (initial recognition/setup)
- hint_2: Second step (builds on hint_1 at runtime)
- hint_3: Third step (builds on hint_1 + hint_2 at runtime)
- hint_4: Fourth step (builds on previous at runtime)
- hint_5: Final step (completes the solution path)

Each hint is stored INDEPENDENTLY (not cumulative) for token efficiency.
At runtime, concatenate: hint_1 + hint_2 + ... + hint_k to get the first k/5 of the solution.

Usage:
    python -m pipeline.tasks.competition_math.rewrite_solutions_prefix_hints \
        --input artifacts/competition_math/primitives.json \
        --output artifacts/competition_math/primitives_prefix_hints.json \
        --batch-size 20 \
        --max-samples 100
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm_asyncio

from pipeline.core.method import ARTIFACTS_ROOT

# Initialize tokenizer (cl100k_base is used by GPT-4, GPT-3.5-turbo, and text-embedding-ada-002)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_ENCODING.encode(text))


PREFIX_HINTS_SYSTEM_PROMPT = """\
You are a mathematics tutor who breaks down solutions into 5 sequential steps.

Your task: Given a problem and its solution, divide the solution into exactly 5 sequential segments. Each segment should be 1-2 sentences containing one logical step.

## What You Must Do

1. **Analyze the solution** and identify 5 sequential steps of roughly equal importance.

2. **Generate 5 INDEPENDENT hint segments** where:
   - **hint_1**: First step (initial recognition, setup, or key insight)
   - **hint_2**: Second step (next logical move, building on hint_1)
   - **hint_3**: Third step (continues from hint_2)
   - **hint_4**: Fourth step (continues from hint_3)
   - **hint_5**: Final step (completes the solution path)

   At runtime, hints will be concatenated: to show 3/5 of the solution, we display hint_1 + hint_2 + hint_3.

## Critical Rules

1. **Each hint is INDEPENDENT**: Do NOT repeat content from previous hints. Each hint contains ONLY its own step.

2. **1-2 sentences per hint**: Keep each hint concise. One logical step per hint.

3. **Precise and actionable**: Tell the student EXACTLY what to do. Include specific equations, identities, or substitutions.

4. **Sequential flow**: When concatenated, hints should read as a coherent solution walkthrough.

5. **No final answer**: hint_5 should end with "compute..." or "evaluate..." leaving the final arithmetic.

6. **Use LaTeX**: Use proper LaTeX notation ($...$) for all math. Do NOT use Unicode math symbols.

## Example 1

**Problem:** Two reals $a$ and $b$ are such that $a+b=7$ and $a^3+b^3=91$. Compute $ab$.

**Solution:** We have $91=a^3+b^3=(a+b)(a^2-ab+b^2)=(a+b)((a+b)^2-3ab)=7\\cdot(49-3ab)$, from which $ab=12$.

**Output:**
{
  "hint_1": "Recall the sum of cubes identity: $a^3 + b^3 = (a+b)(a^2 - ab + b^2)$.",
  "hint_2": "Apply the identity to get $91 = 7(a^2 - ab + b^2)$, so $a^2 - ab + b^2 = 13$.",
  "hint_3": "Express $a^2 + b^2$ using $(a+b)^2 = a^2 + 2ab + b^2$, giving $a^2 + b^2 = 49 - 2ab$.",
  "hint_4": "Substitute into $a^2 - ab + b^2 = 13$ to get $(49 - 2ab) - ab = 13$.",
  "hint_5": "Simplify to $49 - 3ab = 13$ and solve for $ab$."
}

## Example 2

**Problem:** Let $a$ and $b$ be nonzero real numbers. Find the minimum value of $a^2 + b^2 + \\frac{1}{a^2} + \\frac{b}{a}$.

**Solution:** We complete the square with respect to $b^2$ and $\\frac{b}{a}$ to get $b^2 + \\frac{b}{a} = (b + \\frac{1}{2a})^2 - \\frac{1}{4a^2}$. This is minimized when $b = -\\frac{1}{2a}$. The problem becomes minimizing $a^2 + \\frac{1}{a^2} - \\frac{1}{4a^2} = a^2 + \\frac{3}{4a^2}$. By AM-GM, $a^2 + \\frac{3}{4a^2} \\geq 2\\sqrt{\\frac{3}{4}} = \\sqrt{3}$. Equality when $a = \\sqrt[4]{\\frac{3}{4}}$.

**Output:**
{
  "hint_1": "Group the $b$ terms: $b^2 + \\frac{b}{a}$. Complete the square in $b$.",
  "hint_2": "This gives $(b + \\frac{1}{2a})^2 - \\frac{1}{4a^2}$, minimized when $b = -\\frac{1}{2a}$.",
  "hint_3": "Substitute optimal $b$ to get $a^2 + \\frac{1}{a^2} - \\frac{1}{4a^2} = a^2 + \\frac{3}{4a^2}$.",
  "hint_4": "Apply AM-GM: $a^2 + \\frac{3}{4a^2} \\geq 2\\sqrt{a^2 \\cdot \\frac{3}{4a^2}}$.",
  "hint_5": "Evaluate $2\\sqrt{\\frac{3}{4}}$ to find the minimum value."
}

## Example 3 (Number Theory)

**Problem:** A very large number $x$ is equal to $2^2 3^3 4^4 5^5 6^6 7^7 8^8 9^9$. What is the smallest positive integer that, when multiplied with $x$, produces a perfect square?

**Output:**
{
  "hint_1": "For a product to be a perfect square, all prime exponents must be even.",
  "hint_2": "Rewrite each base as prime powers: $4^4 = 2^8$, $6^6 = 2^6 \\cdot 3^6$, $8^8 = 2^{24}$, $9^9 = 3^{18}$.",
  "hint_3": "Collect exponents by prime: $2^{2+8+6+24} = 2^{40}$, $3^{3+6+18} = 3^{27}$, $5^5$, $7^7$.",
  "hint_4": "Identify odd exponents: $3^{27}$, $5^5$, $7^7$ all have odd exponents.",
  "hint_5": "Multiply by $3 \\cdot 5 \\cdot 7$ to make all exponents even. Compute $3 \\cdot 5 \\cdot 7$."
}

## Output Format

Return ONLY a JSON object with this exact structure (no markdown code blocks):
{"hint_1": "...", "hint_2": "...", "hint_3": "...", "hint_4": "...", "hint_5": "..."}
"""

PREFIX_HINTS_USER_TEMPLATE = """\
Problem:
{problem}

Solution:
{solution}

Break this solution into 5 sequential steps (1-2 sentences each). Each hint should be INDEPENDENT (not repeating previous hints):"""


def detect_repetition(text: str, min_pattern_len: int = 2, min_repeats: int = 10) -> bool:
    """Detect if text contains excessive repetition (sign of model breakdown)."""
    if len(text) < min_pattern_len * min_repeats:
        return False
    # Check for repeated characters or short patterns at the end
    for pattern_len in range(min_pattern_len, 6):
        if len(text) < pattern_len * min_repeats:
            continue
        tail = text[-pattern_len * min_repeats:]
        pattern = tail[:pattern_len]
        if tail == pattern * min_repeats:
            return True
    return False


def validate_hints(
    hints: dict,
    max_tokens_per_hint: int = 200,
    max_total_tokens: int = 500,
) -> tuple[bool, str]:
    """Validate hints for quality issues. Returns (is_valid, error_message)."""
    total_tokens = 0

    for i in range(1, 7):
        key = f"hint_{i}"
        hint = hints[key]

        # Check for repetition patterns (model breakdown)
        if detect_repetition(hint):
            return False, f"{key} contains repetitive pattern (model breakdown)"

        tokens = count_tokens(hint)
        total_tokens += tokens

        # Check for excessive length per hint
        if tokens > max_tokens_per_hint:
            return False, f"{key} too long ({tokens} tokens > {max_tokens_per_hint})"

        # Check for too short (likely incomplete)
        if tokens < 3:
            return False, f"{key} too short ({tokens} tokens)"

    # Check total length
    if total_tokens > max_total_tokens:
        return False, f"total too long ({total_tokens} tokens > {max_total_tokens})"

    return True, ""


def parse_hints_response(content: str) -> dict | None:
    """Parse the JSON hints from GPT response."""
    if not content:
        return None

    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
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
        required_keys = {"hint_1", "hint_2", "hint_3", "hint_4", "hint_5"}
        if not required_keys.issubset(hints.keys()):
            print(f"  Warning: Missing keys in response. Got: {hints.keys()}")
            return None

        # Validate hint quality
        is_valid, error = validate_hints(hints)
        if not is_valid:
            print(f"  Warning: Invalid hints - {error}")
            return None

        return hints
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON: {e}")
        print(f"  Content was: {content[:200]}...")
        return None


def validate_hint_structure(hints: dict) -> dict:
    """Compute stats for incremental hints (in tokens)."""
    tokens = [count_tokens(hints[f"hint_{i}"]) for i in range(1, 7)]
    total = sum(tokens)
    return {
        "tokens": tokens,  # Token counts per hint
        "total_tokens": total,  # Total when concatenated
        "avg_tokens": total / 5,  # Average per hint
        "max_tokens": max(tokens),  # Longest hint
        "distribution": [round(t / total * 100, 1) if total > 0 else 0 for t in tokens],  # % contribution
    }


def parse_rate_limit_delay(error_message: str) -> float:
    """Extract retry delay from rate limit error message."""
    # Look for patterns like "Please try again in 145ms" or "in 2.5s"
    match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(error_message))
    if match:
        value, unit = float(match.group(1)), match.group(2)
        seconds = value / 1000 if unit == "ms" else value
        return seconds + 0.5  # Add buffer
    return 2.0  # Default fallback


async def generate_prefix_hints(
    client: AsyncOpenAI,
    problem: str,
    solution: str,
    model: str = "gpt-5-mini-2025-08-07",
    max_retries: int = 5,
    semaphore: asyncio.Semaphore | None = None,
    verbose: bool = False,
) -> dict | None:
    """Generate prefix hints for a single problem (async)."""
    user_content = PREFIX_HINTS_USER_TEMPLATE.format(problem=problem, solution=solution)

    async def _call():
        attempt = 0
        while attempt < max_retries:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PREFIX_HINTS_SYSTEM_PROMPT},
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
                    attempt += 1
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                    continue
                return hints
            except RateLimitError as e:
                # Rate limits don't count against max_retries
                delay = parse_rate_limit_delay(str(e))
                if verbose:
                    print(f"  Rate limited, sleeping {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                print(f"  API error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue
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
    """Process a single primitive: generate prefix hints."""
    solution = primitive.get("solution_original", primitive.get("solution", ""))

    hints = await generate_prefix_hints(
        client=client,
        problem=primitive["problem"],
        solution=solution,
        model=model,
        semaphore=semaphore,
        verbose=verbose,
    )

    new_p = primitive.copy()

    if hints is not None:
        new_p["prefix_hints"] = hints
        new_p["hint_stats"] = validate_hint_structure(hints)
    else:
        new_p["prefix_hints"] = None
        new_p["hint_stats"] = None

    return new_p


class IncrementalSaver:
    """Handles incremental saving of results with resume support."""

    def __init__(
        self,
        output_path: Path,
        primitives: list[dict],
        save_every: int = 30,
    ):
        self.output_path = output_path
        self.save_every = save_every
        self.unsaved_count = 0
        self.lock = asyncio.Lock()

        # Build index -> position mapping
        self.index_to_pos = {p["index"]: i for i, p in enumerate(primitives)}

        # Initialize or load results array
        if output_path.exists():
            with open(output_path, "r") as f:
                self.results = json.load(f)
            # Extend if input has more items
            if len(self.results) < len(primitives):
                self.results.extend([None] * (len(primitives) - len(self.results)))
        else:
            # Initialize with None placeholders
            self.results = [None] * len(primitives)
            self._save()

    def get_pending_indices(self) -> set[int]:
        """Return indices that are not yet completed (None or missing prefix_hints)."""
        pending = set()
        for idx, pos in self.index_to_pos.items():
            if pos >= len(self.results):
                pending.add(idx)
            elif self.results[pos] is None:
                pending.add(idx)
            elif self.results[pos].get("prefix_hints") is None:
                pending.add(idx)
        return pending

    async def save_result(self, result: dict) -> None:
        """Save a single result (thread-safe)."""
        async with self.lock:
            idx = result["index"]
            pos = self.index_to_pos[idx]
            self.results[pos] = result
            self.unsaved_count += 1

            if self.unsaved_count >= self.save_every:
                self._save()
                self.unsaved_count = 0

    def _save(self) -> None:
        """Write results to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=True)

    def finalize(self) -> list[dict]:
        """Final save and return completed results."""
        self._save()
        return [r for r in self.results if r is not None]


async def process_and_save(
    client: AsyncOpenAI,
    primitive: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    saver: IncrementalSaver,
    failed_queue: asyncio.Queue,
    verbose: bool = False,
) -> None:
    """Process a single primitive and save immediately. Requeue on failure."""
    result = await process_primitive(client, primitive, model, semaphore, verbose)
    if result.get("prefix_hints") is None:
        # Failed - add to retry queue (don't save yet)
        await failed_queue.put(primitive)
    else:
        await saver.save_result(result)


async def generate_all_prefix_hints(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5-mini-2025-08-07",
    max_samples: int | None = None,
    batch_size: int = 20,
    save_every: int = 30,
    max_retries: int = 3,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    """Generate prefix hints for all problems (async with incremental saving)."""
    # Load input
    with open(input_path, "r") as f:
        primitives = json.load(f)

    print(f"Loaded {len(primitives)} primitives from {input_path}")

    # Initialize saver with FULL primitives list (creates full JSON structure)
    if not resume and output_path.exists():
        output_path.unlink()  # Delete existing file if not resuming

    saver = IncrementalSaver(output_path, primitives, save_every=save_every)

    # Determine which primitives to process (may be limited by max_samples)
    primitives_to_process = primitives
    if max_samples is not None:
        primitives_to_process = primitives[:max_samples]
        print(f"Processing first {max_samples} samples (full JSON has {len(primitives)} slots)")

    # Find pending items within the range we're processing
    pending_indices = saver.get_pending_indices()
    pending = [p for p in primitives_to_process if p["index"] in pending_indices]

    total_in_range = len(primitives_to_process)
    completed_in_range = total_in_range - len(pending)
    print(f"Completed: {completed_in_range}/{total_in_range}")
    print(f"Pending: {len(pending)} to process (batch_size={batch_size}, save_every={save_every})")

    if not pending:
        print("Nothing to do!")
        return

    # Initialize async client
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(batch_size)

    # Process with retry loop
    current_batch = pending
    for retry_round in range(max_retries):
        if not current_batch:
            break

        round_label = "Generating" if retry_round == 0 else f"Retry {retry_round}"
        failed_queue: asyncio.Queue = asyncio.Queue()

        # Process all items in current batch
        tasks = [
            process_and_save(client, p, model, semaphore, saver, failed_queue, verbose)
            for p in current_batch
        ]

        await tqdm_asyncio.gather(*tasks, desc=f"{round_label} ({len(current_batch)} items)")

        # Collect failed items for next round
        failed_items = []
        while not failed_queue.empty():
            failed_items.append(await failed_queue.get())

        if failed_items:
            print(f"  {len(failed_items)} failed, will retry...")
            current_batch = failed_items
        else:
            current_batch = []

    # Save any remaining failures as null
    if current_batch:
        print(f"  {len(current_batch)} items failed after {max_retries} attempts")
        for p in current_batch:
            # Save with null prefix_hints
            failed_result = p.copy()
            failed_result["prefix_hints"] = None
            failed_result["hint_stats"] = None
            await saver.save_result(failed_result)

    # Final save and summary
    all_results = saver.finalize()
    success = sum(1 for r in all_results if r.get("prefix_hints") is not None)

    # Compute aggregate stats
    total_tokens = [
        r["hint_stats"]["total_tokens"]
        for r in all_results
        if r.get("hint_stats")
    ]
    avg_total = sum(total_tokens) / len(total_tokens) if total_tokens else 0
    max_total = max(total_tokens) if total_tokens else 0

    # Compute per-position averages
    position_avgs = []
    for i in range(5):
        pos_tokens = [
            r["hint_stats"]["tokens"][i]
            for r in all_results
            if r.get("hint_stats")
        ]
        position_avgs.append(sum(pos_tokens) / len(pos_tokens) if pos_tokens else 0)

    print(f"\nDone! {success}/{len(all_results)} problems processed successfully")
    print(f"Avg total tokens: {avg_total:.0f} ({avg_total/5:.0f} per hint)")
    print(f"Max total tokens: {max_total}")
    print(f"Per-position avg: {' | '.join(f'H{i+1}:{v:.0f}' for i, v in enumerate(position_avgs))}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate incremental solution hints (5 sequential steps) for competition math problems"
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
        default=ARTIFACTS_ROOT / "competition_math" / "primitives_prefix_hints.json",
        help="Output file with prefix hints",
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
        "--save-every", "-s",
        type=int,
        default=30,
        help="Save to disk every N completions (default: 30)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=3,
        help="Max retry rounds for failed items (default: 3)",
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

    asyncio.run(generate_all_prefix_hints(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        save_every=args.save_every,
        max_retries=args.max_retries,
        resume=not args.no_resume,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
