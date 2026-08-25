"""
LLM judge for profiling reasoning behavior.

Uses the OpenAI Batch API to analyze model generations:
- Abstention reason classification
- Verification and backtracking detection
- Solution attempt counting
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; the real import is lazy (see below)
    from openai import OpenAI
# The openai package is imported inside the functions that need it. At module
# level it sat on the critical path of every CLI invocation, including
# `--help`, and made the whole CLI unusable when it was absent.

from pipeline.core.io import load_json, save_json


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

JUDGE_ABSTAINED_PROMPT = (_PROMPTS_DIR / "judge_abstained.txt").read_text()
JUDGE_ANSWERED_PROMPT = (_PROMPTS_DIR / "judge_answered.txt").read_text()
JUDGE_CORRECTNESS_PROMPT = (_PROMPTS_DIR / "judge_correctness.txt").read_text()
JUDGE_VERIFY_PROMPT = (_PROMPTS_DIR / "judge_verify.txt").read_text()

USER_TEMPLATE = """\
## Problem

{problem}

## Ground truth

{ground_truth}

## Model's reasoning (enumerated lines)

{lines}

Respond in JSON."""


def _format_lines(lines: dict[str, str]) -> str:
    """Format enumerated lines dict into numbered text."""
    return "\n".join(f"{k}: {v}" for k, v in lines.items())


def _create_batch_requests(examples: list[dict], model: str) -> list[dict]:
    """Create JSONL batch request entries for all examples."""
    requests = []
    for ex in examples:
        is_abstained = ex.get("abstained", False)
        system_prompt = JUDGE_ABSTAINED_PROMPT if is_abstained else JUDGE_ANSWERED_PROMPT

        user_content = USER_TEMPLATE.format(
            problem=ex["problem"],
            ground_truth=json.dumps(ex["ground_truth"]),
            lines=_format_lines(ex["lines"]),
        )

        requests.append({
            "custom_id": str(ex["index"]),
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": system_prompt,
                "input": user_content,
                "max_output_tokens": 256,
                "reasoning": {"effort": "none"},
                "text": {"format": {"type": "json_object"}},
            },
        })
    return requests


def _submit_and_wait(
    client: OpenAI,
    requests: list[dict],
    poll_interval: int = 30,
    stall_timeout: int = 600,
) -> dict:
    """Upload batch JSONL, submit batch job, poll until done, return results by custom_id.

    Args:
        stall_timeout: Cancel batch if no progress for this many seconds (default: 10 min).
            Set to 0 to disable stall detection (wait indefinitely).
            Partial results are returned if available.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        jsonl_path = f.name

    try:
        print(f"Uploading {len(requests)} requests...")
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"  File ID: {file_obj.id}")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")

        last_completed = 0
        stall_start = None

        while batch.status not in ("completed", "failed", "cancelled", "expired"):
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch.id)
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else len(requests)
            failed = batch.request_counts.failed if batch.request_counts else 0
            print(f"  Status: {batch.status} ({completed}/{total} completed, {failed} failed)")

            # Stall detection (disabled when stall_timeout=0)
            if completed > last_completed:
                last_completed = completed
                stall_start = None
            elif stall_timeout > 0 and batch.status == "in_progress":
                if stall_start is None:
                    stall_start = time.time()
                elif time.time() - stall_start > stall_timeout:
                    print(f"  Stalled for >{stall_timeout}s. Cancelling batch to retrieve partial results...")
                    client.batches.cancel(batch.id)
                    # Wait indefinitely for cancellation to complete
                    while True:
                        time.sleep(30)
                        batch = client.batches.retrieve(batch.id)
                        print(f"  Cancelling... (status: {batch.status}, completed: {batch.request_counts.completed}/{batch.request_counts.total})")
                        if batch.status in ("cancelled", "completed", "failed"):
                            break
                    break

        print(f"Downloading results... (final status: {batch.status})")
        # Re-retrieve to ensure we have the latest output_file_id
        batch = client.batches.retrieve(batch.id)
        output_file_id = batch.output_file_id
        error_file_id = batch.error_file_id

        if error_file_id:
            error_content = client.files.content(error_file_id).text
            error_lines = error_content.strip().split("\n")[:3]
            print(f"  Batch errors ({len(error_content.strip().split(chr(10)))} total):")
            for line in error_lines:
                print(f"    {line[:300]}")

        if not output_file_id:
            if batch.status in ("cancelled", "cancelling"):
                print(f"  No results from cancelled batch (0 completed).")
                return {}
            raise RuntimeError(
                f"Batch {batch.id} has no output file. "
                f"All {batch.request_counts.failed} requests failed. See errors above."
            )

        content = client.files.content(output_file_id).text
        results = _parse_batch_results(content)
        total_requested = batch.request_counts.total if batch.request_counts else len(requests)
        if len(results) < total_requested:
            print(f"  Partial results: {len(results)}/{total_requested}")
        return results

    finally:
        os.unlink(jsonl_path)


def _parse_batch_results(content: str) -> dict:
    """Parse batch API response content into results by custom_id."""
    results_by_id = {}
    lines = content.strip().split("\n")
    errors = 0

    for line in lines:
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        response_body = entry.get("response", {}).get("body", {})

        # Extract text from Responses API output
        text_content = None
        output_items = response_body.get("output", [])
        for item in output_items:
            if item.get("type") == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "output_text":
                        text_content = content_part.get("text")
                        break

        if text_content:
            try:
                parsed = json.loads(text_content)
                results_by_id[custom_id] = parsed
                continue
            except json.JSONDecodeError:
                pass

        if errors < 5:
            status = response_body.get("status", "unknown")
            print(f"  Parse error for custom_id={custom_id}: status={status}, text={text_content!r}")
        errors += 1
        results_by_id[custom_id] = {"error": True}

    if errors:
        print(f"  Total parse errors: {errors}/{len(lines)}")

    return results_by_id


def judge_generations(
    preprocessed_path: Path,
    output_path: Path | None = None,
    model: str = "gpt-5.4-nano",
    poll_interval: int = 30,
    max_samples: int | None = None,
) -> Path:
    """
    Judge preprocessed generations using the OpenAI Batch API.

    Sends each example to an LLM judge with task-appropriate prompts
    (different for abstained vs non-abstained examples) and collects
    structured analysis of reasoning behavior.

    Args:
        preprocessed_path: Path to preprocessed JSON from preprocess_for_judge
        output_path: Where to save judge results
        model: OpenAI model to use as judge
        poll_interval: Seconds between batch status checks
        max_samples: Limit number of examples to judge (for testing)

    Returns:
        Path to judge results file
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    data = load_json(preprocessed_path)
    if not isinstance(data, dict) or "examples" not in data:
        raise ValueError(f"Expected preprocessed data with 'examples' key, got: {type(data)}")

    examples = data["examples"]
    if max_samples is not None:
        examples = examples[:max_samples]

    abstained = [e for e in examples if e.get("abstained", False)]
    answered = [e for e in examples if not e.get("abstained", False)]

    print(f"=== LLM Judge (Batch API) ===")
    print(f"Source: {preprocessed_path}")
    print(f"Task: {data.get('task', 'unknown')}")
    print(f"Judge model: {model}")
    print(f"Total examples: {len(examples)}")
    print(f"  Abstained: {len(abstained)}")
    print(f"  Answered: {len(answered)}")
    print(f"=============================")

    # Create and submit batch
    from openai import OpenAI

    client = OpenAI()
    requests = _create_batch_requests(examples, model)
    results_by_id = _submit_and_wait(client, requests, poll_interval)

    # Build index→token_counts lookup from preprocessed data
    token_counts_by_index = {
        ex["index"]: ex.get("line_token_counts", {})
        for ex in examples
    }

    # Merge judge results with example metadata and compute coverage
    judged = []
    parse_errors = 0
    for ex in examples:
        idx = str(ex["index"])
        judge_result = results_by_id.get(idx, {"error": True})

        if judge_result.get("error"):
            parse_errors += 1

        entry = {
            "index": ex["index"],
            "variant": ex["variant"],
            "correct": ex["correct"],
            "abstained": ex["abstained"],
            "token_count": ex["token_count"],
            "num_lines": ex["num_lines"],
            **judge_result,
        }

        # Compute token coverage percentages and raw counts
        if not judge_result.get("error"):
            ltc = token_counts_by_index.get(ex["index"], {})
            total_tokens = sum(ltc.values()) if ltc else 0
            if total_tokens > 0:
                verif_tokens = sum(ltc.get(str(i), 0) for i in judge_result.get("verification_indices", []))
                back_tokens = sum(ltc.get(str(i), 0) for i in judge_result.get("backtrack_indices", []))
                entry["verification_token_count"] = verif_tokens
                entry["verification_token_coverage"] = round(verif_tokens / total_tokens, 4)
                entry["backtrack_token_count"] = back_tokens
                entry["backtrack_token_coverage"] = round(back_tokens / total_tokens, 4)

        judged.append(entry)

    # Print summary
    abstained_judged = [j for j in judged if j["abstained"] and not j.get("error")]
    answered_judged = [j for j in judged if not j["abstained"] and not j.get("error")]

    if abstained_judged:
        from collections import Counter
        reasons = Counter(j.get("abstention_reason", "unknown") for j in abstained_judged)
        reached_correct = sum(1 for j in abstained_judged if j.get("reached_answer_correct") is True)
        reached_wrong = sum(1 for j in abstained_judged if j.get("reached_answer_correct") is False)
        reached_none = sum(1 for j in abstained_judged if j.get("reached_answer") is None)

        print(f"\n=== Abstained Examples ({len(abstained_judged)}) ===")
        print(f"Abstention reasons:")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count} ({count/len(abstained_judged):.1%})")
        print(f"Reached answer: {reached_correct} correct, {reached_wrong} wrong, {reached_none} none")

    all_judged = [j for j in judged if not j.get("error")]
    if all_judged:
        avg_verification = sum(len(j.get("verification_indices", [])) for j in all_judged) / len(all_judged)
        avg_backtrack = sum(len(j.get("backtrack_indices", [])) for j in all_judged) / len(all_judged)
        avg_attempts = sum(j.get("num_attempts", 0) for j in all_judged) / len(all_judged)
        with_coverage = [j for j in all_judged if "verification_token_coverage" in j]
        avg_verif_cov = sum(j["verification_token_coverage"] for j in with_coverage) / len(with_coverage) if with_coverage else 0
        avg_back_cov = sum(j["backtrack_token_coverage"] for j in with_coverage) / len(with_coverage) if with_coverage else 0

        print(f"\n=== All Examples ({len(all_judged)}) ===")
        print(f"Avg verification lines: {avg_verification:.1f} ({avg_verif_cov:.1%} token coverage)")
        print(f"Avg backtrack lines: {avg_backtrack:.1f} ({avg_back_cov:.1%} token coverage)")
        print(f"Avg attempts: {avg_attempts:.1f}")

    if parse_errors:
        print(f"\nParse errors: {parse_errors}/{len(examples)}")

    print(f"=============================")

    # Determine output path
    if output_path is None:
        stem = preprocessed_path.stem.replace("_preprocessed", "")
        output_path = preprocessed_path.parent / f"{stem}_judged.json"

    output = {
        "source": str(preprocessed_path),
        "task": data.get("task", "unknown"),
        "judge_model": model,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total": len(examples),
            "abstained": len(abstained),
            "answered": len(answered),
            "parse_errors": parse_errors,
        },
        "details": judged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)
    print(f"\nSaved to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Verify judge — confidence scoring & verification sentence identification
# ---------------------------------------------------------------------------

VERIFY_USER_TEMPLATE = """\
## Problem

{problem}

## Model's generation (indexed sentences)

{sentences}

Respond in JSON."""


def _format_problem_for_verify(ground_truth: dict, task_name: str) -> str:
    """Format problem context for verify judge — excludes ground truth answer."""
    if task_name == "countdown":
        target = ground_truth.get("target")
        numbers = ground_truth.get("numbers", [])
        return f"Task: Use the numbers {numbers} with basic arithmetic to reach {target}."
    elif task_name == "competition_math":
        problem = ground_truth.get("problem", "")
        if len(problem) > 300:
            problem = problem[:300] + "..."
        return f"Problem: {problem}"
    elif task_name == "code_output":
        return "Task: Predict the output of a program."
    else:
        return f"Task: {task_name}"


def _split_into_sentences(generation: str) -> list[str]:
    """Split the <think> section into sentences by periods and newlines.

    Only includes reasoning content (before </think>), excluding the
    <verify> section which is prompted verification. Periods between
    digits (e.g. 3.89) are preserved, not treated as delimiters.
    """
    from pipeline.core.utils import extract_think

    text = extract_think(generation)
    if text is None:
        # No </think> tag — take everything up to <answer> or end
        text = re.sub(r"^<think>\s*", "", generation, flags=re.IGNORECASE)
        text = re.sub(r"\s*<answer>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Split by newlines OR periods not followed by a digit (preserves decimals like 3.89)
    parts = re.split(r"\n|\.(?!\d)", text)
    # Filter empty, whitespace-only, and tag-only fragments
    tag_only = re.compile(r"^\s*</?[a-z_]+>\s*$", re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and not tag_only.match(p)]


def _format_indexed_sentences(sentences: list[str]) -> str:
    """Format sentences as an indexed list for the judge prompt."""
    return "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))


def preprocess_for_verify_judge(
    results_path: Path,
    task_name: str,
    output_path: Path | None = None,
) -> Path:
    """
    Preprocess eval results into indexed sentences for the verify judge.

    Splits each generation by periods and newlines, strips XML structural
    tags, and saves alongside metadata needed for judging.

    Args:
        results_path: Path to evaluation results JSON
        task_name: Task name (countdown, competition_math, code_output)
        output_path: Where to save preprocessed output

    Returns:
        Path to preprocessed JSON file
    """

    data = load_json(results_path)
    if not isinstance(data, dict) or "details" not in data:
        raise ValueError(f"Expected eval results with 'details' key, got: {type(data)}")

    details = data["details"]

    examples = []
    for d in details:
        generation = d.get("generation", "")
        sentences = _split_into_sentences(generation)
        ground_truth = d.get("ground_truth", {})

        examples.append({
            "index": d["index"],
            "variant": d.get("variant", "unknown"),
            "correct": d.get("correct", False),
            "committed": d.get("committed", d.get("metadata", {}).get("committed", False)),
            "abstained": d.get("abstained", d.get("metadata", {}).get("abstained", False)),
            "token_count": d.get("token_count", 0),
            "num_sentences": len(sentences),
            "sentences": {str(i): s for i, s in enumerate(sentences)},
            "problem": _format_problem_for_verify(ground_truth, task_name),
        })

    total = len(examples)
    committed = sum(1 for e in examples if e["committed"])
    abstained = sum(1 for e in examples if e["abstained"])
    avg_sentences = sum(e["num_sentences"] for e in examples) / total if total else 0

    print(f"=== Preprocessing for Verify Judge ===")
    print(f"Source: {results_path}")
    print(f"Task: {task_name}")
    print(f"Total examples: {total}")
    print(f"  Committed: {committed}")
    print(f"  Abstained: {abstained}")
    print(f"Avg sentences per generation: {avg_sentences:.1f}")
    print(f"======================================")

    if output_path is None:
        analysis_dir = results_path.parent.parent / "analysis"
        stem = results_path.stem
        output_path = analysis_dir / f"{stem}_verify_preprocessed.json"

    output = {
        "source": str(results_path),
        "task": task_name,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total": total,
            "committed": committed,
            "abstained": abstained,
            "avg_sentences": round(avg_sentences, 1),
        },
        "examples": examples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)
    print(f"Saved to {output_path}")

    return output_path


def _create_verify_batch_requests(examples: list[dict], model: str) -> list[dict]:
    """Create batch request entries for the verify judge."""
    requests = []
    for ex in examples:
        sentences = ex["sentences"]
        formatted = "\n".join(f"{k}: {v}" for k, v in sentences.items())
        user_content = VERIFY_USER_TEMPLATE.format(
            problem=ex["problem"],
            sentences=formatted,
        )
        requests.append({
            "custom_id": str(ex["index"]),
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": JUDGE_VERIFY_PROMPT,
                "input": user_content,
                "max_output_tokens": 512,
                "reasoning": {"effort": "none"},
                "text": {"format": {"type": "json_object"}},
            },
        })
    return requests


def _submit_sync(
    client: OpenAI,
    examples: list[dict],
    model: str,
    max_concurrent: int = 20,
) -> dict:
    """Call the Responses API concurrently. Returns results by index."""
    import concurrent.futures

    results_by_id = {}
    errors = 0
    completed = 0

    def _call(ex):
        sentences = ex["sentences"]
        formatted = "\n".join(f"{k}: {v}" for k, v in sentences.items())
        user_content = VERIFY_USER_TEMPLATE.format(
            problem=ex["problem"],
            sentences=formatted,
        )
        response = client.responses.create(
            model=model,
            instructions=JUDGE_VERIFY_PROMPT,
            input=user_content,
            max_output_tokens=512,
            reasoning={"effort": "none"},
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_call, ex): ex for ex in examples}
        for future in concurrent.futures.as_completed(futures):
            ex = futures[future]
            idx = str(ex["index"])
            completed += 1
            try:
                results_by_id[idx] = future.result()
            except Exception as e:
                if errors < 5:
                    print(f"  Error for index={idx}: {e}")
                errors += 1
                results_by_id[idx] = {"error": True}
            print(f"  {completed}/{len(examples)} completed", end="\r")

    print()
    if errors:
        print(f"  Total errors: {errors}/{len(examples)}")
    return results_by_id


def judge_verify(
    preprocessed_path: Path,
    output_path: Path | None = None,
    model: str = "gpt-5.4-nano",
    poll_interval: int = 30,
    max_samples: int | None = None,
    sync: bool = False,
    max_concurrent: int = 20,
) -> Path:
    """
    Judge preprocessed verify generations for confidence and verification coverage.

    Takes output from preprocess_for_verify_judge and sends to the LLM judge
    to evaluate:
      - generation_confidence (1-5): apparent model confidence
      - verification_sentences: sentence indices that are self-verification

    Args:
        preprocessed_path: Path to preprocessed JSON from preprocess_for_verify_judge
        output_path: Where to save judge results
        model: OpenAI model to use as judge
        poll_interval: Seconds between batch status checks
        max_samples: Limit number of examples to judge (for testing)
        sync: Use synchronous API calls instead of Batch API

    Returns:
        Path to judge results file
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    data = load_json(preprocessed_path)
    if not isinstance(data, dict) or "examples" not in data:
        raise ValueError(f"Expected preprocessed data with 'examples' key, got: {type(data)}")

    examples = data["examples"]
    if max_samples is not None:
        examples = examples[:max_samples]

    committed = sum(1 for e in examples if e.get("committed", False))
    abstained = sum(1 for e in examples if e.get("abstained", False))
    avg_sentences = sum(e["num_sentences"] for e in examples) / len(examples) if examples else 0

    mode = "Sync" if sync else "Batch API"
    print(f"=== Verify Judge ({mode}) ===")
    print(f"Source: {preprocessed_path}")
    print(f"Task: {data.get('task', 'unknown')}")
    print(f"Judge model: {model}")
    print(f"Total examples: {len(examples)}")
    print(f"  Committed: {committed}")
    print(f"  Abstained: {abstained}")
    print(f"Avg sentences per generation: {avg_sentences:.1f}")
    print(f"================================")

    from openai import OpenAI

    client = OpenAI()
    if sync:
        results_by_id = _submit_sync(client, examples, model, max_concurrent)
    else:
        requests = _create_verify_batch_requests(examples, model)
        results_by_id = _submit_and_wait(client, requests, poll_interval)

    # Merge judge results with example metadata
    judged = []
    parse_errors = 0
    for ex in examples:
        idx = str(ex["index"])
        judge_result = results_by_id.get(idx, {"error": True})

        if judge_result.get("error"):
            parse_errors += 1

        entry = {
            "index": ex["index"],
            "variant": ex["variant"],
            "correct": ex["correct"],
            "committed": ex["committed"],
            "abstained": ex["abstained"],
            "token_count": ex["token_count"],
            "num_sentences": ex["num_sentences"],
            **judge_result,
        }

        # Compute verification coverage ratio
        if not judge_result.get("error"):
            verif_indices = judge_result.get("verification_sentences", [])
            if not isinstance(verif_indices, list):
                verif_indices = [verif_indices]
            entry["verification_coverage"] = round(
                len(verif_indices) / ex["num_sentences"], 4
            ) if ex["num_sentences"] > 0 else 0

        judged.append(entry)

    # Compute summary stats
    valid = [j for j in judged if not j.get("error")]
    breakdown = {}

    if valid:
        correct_v = [j for j in valid if j["correct"]]
        incorrect_v = [j for j in valid if not j["correct"]]

        def _conf_breakdown(group):
            """Compute per-confidence-level stats for a group."""
            rows = {}
            for score in range(1, 6):
                g = [d for d in group if d.get("generation_confidence") == score]
                n = len(g)
                if n == 0:
                    rows[score] = {"n": 0, "commit_rate": 0, "abstain_rate": 0, "avg_verif_coverage": 0}
                    continue
                rows[score] = {
                    "n": n,
                    "commit_rate": round(sum(1 for d in g if d["committed"]) / n, 4),
                    "abstain_rate": round(sum(1 for d in g if d["abstained"]) / n, 4),
                    "avg_verif_coverage": round(sum(d.get("verification_coverage", 0) for d in g) / n, 4),
                }
            return rows

        breakdown = {
            "correct": _conf_breakdown(correct_v),
            "incorrect": _conf_breakdown(incorrect_v),
        }

        # Print summary
        print(f"\n=== Results ({len(valid)} examples, {parse_errors} errors) ===")
        for label, group in [("CORRECT", correct_v), ("INCORRECT", incorrect_v)]:
            avg_conf = sum(d.get("generation_confidence", 0) for d in group) / len(group) if group else 0
            print(f"\n{label} (n={len(group)}, avg confidence={avg_conf:.2f}):")
            print(f"  Conf     N   Commit%  Abstain%  VerifCov")
            print(f"  " + "-" * 46)
            rows = breakdown[label.lower()]
            for score in range(1, 6):
                r = rows[score]
                n = r["n"]
                if n == 0:
                    print(f"    {score}      0       -        -        -")
                else:
                    pct = n / len(group) * 100
                    print(f"    {score}   {n:>4d} ({pct:>5.1f}%)  {r['commit_rate']:>5.1%}    {r['abstain_rate']:>5.1%}    {r['avg_verif_coverage']:.3f}")

    if parse_errors:
        print(f"\nParse errors: {parse_errors}/{len(examples)}")

    print(f"\n================================")

    # Determine output path
    if output_path is None:
        stem = preprocessed_path.stem.replace("_verify_preprocessed", "")
        output_path = preprocessed_path.parent / f"{stem}_verify_judged.json"

    output = {
        "source": str(preprocessed_path),
        "task": data.get("task", "unknown"),
        "judge_model": model,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total": len(examples),
            "committed": committed,
            "abstained": abstained,
            "parse_errors": parse_errors,
            "confidence_breakdown": breakdown,
        },
        "details": judged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)
    print(f"\nSaved to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Correctness judge — LLM-based answer equivalence checking
# ---------------------------------------------------------------------------

CORRECTNESS_USER_TEMPLATE = """\
## Problem

{problem}

## Ground Truth Answer

{ground_truth_answer}

## Model's Generation

{generation}

Respond in JSON."""


def _create_correctness_batch_requests(examples: list[dict], model: str) -> list[dict]:
    """Create JSONL batch request entries for correctness judging."""
    requests = []
    for ex in examples:
        user_content = CORRECTNESS_USER_TEMPLATE.format(
            problem=ex["problem"],
            ground_truth_answer=ex["ground_truth_answer"],
            generation=ex["generation"],
        )
        requests.append({
            "custom_id": str(ex["index"]),
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": JUDGE_CORRECTNESS_PROMPT,
                "input": user_content,
                "max_output_tokens": 512,
                "reasoning": {"effort": "low"},
                "text": {"format": {"type": "json_object"}},
            },
        })
    return requests


def _submit_correctness_sync(
    client: OpenAI,
    examples: list[dict],
    model: str,
    max_concurrent: int = 20,
) -> dict:
    """Call the Responses API concurrently for correctness judging."""
    import concurrent.futures

    results_by_id = {}
    errors = 0
    completed = 0

    def _call(ex):
        user_content = CORRECTNESS_USER_TEMPLATE.format(
            problem=ex["problem"],
            ground_truth_answer=ex["ground_truth_answer"],
            generation=ex["generation"],
        )
        response = client.responses.create(
            model=model,
            instructions=JUDGE_CORRECTNESS_PROMPT,
            input=user_content,
            max_output_tokens=512,
            reasoning={"effort": "low"},
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_call, ex): ex for ex in examples}
        for future in concurrent.futures.as_completed(futures):
            ex = futures[future]
            idx = str(ex["index"])
            completed += 1
            try:
                results_by_id[idx] = future.result()
            except Exception as e:
                if errors < 5:
                    print(f"  Error for index={idx}: {e}")
                errors += 1
                results_by_id[idx] = {"error": True}
            print(f"  {completed}/{len(examples)} completed", end="\r")

    print()
    if errors:
        print(f"  Total errors: {errors}/{len(examples)}")
    return results_by_id


def judge_correctness(
    results_path: Path,
    task_name: str,
    output_path: Path | None = None,
    model: str = "gpt-5.4-mini",
    poll_interval: int = 30,
    stall_timeout: int = 600,
    sync: bool = False,
    max_concurrent: int = 20,
) -> Path:
    """
    Judge correctness of model answers using LLM.

    Takes eval results and asks an LLM judge whether each model answer
    matches the ground truth. Useful for math where symbolic equivalence
    checking may miss valid answers.

    Args:
        results_path: Path to eval results JSON (from evaluate command)
        task_name: Task name (for extracting problem text)
        output_path: Where to save judge results
        model: OpenAI model to use as judge
        poll_interval: Seconds between batch status checks
        stall_timeout: Cancel batch if no progress for this many seconds
        sync: Use synchronous concurrent API calls instead of Batch API
        max_concurrent: Max concurrent requests in sync mode

    Returns:
        Path to judge results file
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    data = load_json(results_path)
    if isinstance(data, dict) and "details" in data:
        details = data["details"]
    else:
        details = data

    # Build examples for judging
    examples = []
    for d in details:
        gt = d.get("ground_truth", {})

        # Extract problem text based on task
        if task_name == "competition_math":
            problem = gt.get("problem", "")
            gt_answer = gt.get("answer", "")
        elif task_name == "countdown":
            problem = f"Use the numbers {gt.get('numbers', [])} with basic arithmetic to reach {gt.get('target', '')}."
            gt_answer = gt.get("answer", "")
        elif task_name == "code_output":
            problem = f"Predict the output of a program.\nExpected output: {gt.get('expected_stdout', '')}"
            gt_answer = gt.get("expected_stdout", "")
        else:
            problem = str(gt)
            gt_answer = gt.get("answer", "")

        generation = d.get("generation", "")
        if not generation:
            continue
        # Truncate after </answer> — judge only needs the reasoning + answer, not verify/commit
        answer_end = generation.find("</answer>")
        if answer_end != -1:
            generation = generation[:answer_end + len("</answer>")]

        examples.append({
            "index": d["index"],
            "problem": problem,
            "ground_truth_answer": str(gt_answer),
            "generation": generation,
            "original_correct": d.get("correct", False),
            "variant": d.get("variant", "unknown"),
        })

    mode = "Sync" if sync else "Batch API"
    print(f"=== Correctness Judge ({mode}) ===")
    print(f"Source: {results_path}")
    print(f"Task: {task_name}")
    print(f"Judge model: {model}")
    print(f"Total examples: {len(examples)}")
    print(f"=====================================")

    from openai import OpenAI

    client = OpenAI()
    if sync:
        results_by_id = _submit_correctness_sync(client, examples, model, max_concurrent)
    else:
        requests = _create_correctness_batch_requests(examples, model)
        results_by_id = _submit_and_wait(client, requests, poll_interval, stall_timeout)

    # Merge results
    judged = []
    agree = 0
    disagree = 0
    parse_errors = 0
    for ex in examples:
        idx = str(ex["index"])
        judge_result = results_by_id.get(idx, {"error": True})

        if judge_result.get("error"):
            parse_errors += 1
            judge_correct = None
        else:
            judge_correct = judge_result.get("correct", None)

        original = ex["original_correct"]
        if judge_correct is not None:
            if judge_correct == original:
                agree += 1
            else:
                disagree += 1

        judged.append({
            "index": ex["index"],
            "variant": ex["variant"],
            "original_correct": original,
            "judge_correct": judge_correct,
            "explanation": judge_result.get("explanation", ""),
            "agreement": judge_correct == original if judge_correct is not None else None,
        })

    total_judged = agree + disagree
    print(f"\n=== Results ===")
    if total_judged:
        print(f"Agreement: {agree}/{total_judged} ({100*agree/total_judged:.1f}%)")
    print(f"Disagreements: {disagree}")
    if disagree > 0:
        flipped_correct = sum(1 for j in judged if j["judge_correct"] is True and not j["original_correct"])
        flipped_incorrect = sum(1 for j in judged if j["judge_correct"] is False and j["original_correct"])
        print(f"  Original wrong, judge says correct: {flipped_correct}")
        print(f"  Original correct, judge says wrong: {flipped_incorrect}")
    if parse_errors:
        print(f"Parse errors: {parse_errors}")

    # Determine output path
    if output_path is None:
        stem = results_path.stem
        output_path = results_path.parent / f"{stem}_correctness_judged.json"

    output = {
        "source": str(results_path),
        "task": task_name,
        "judge_model": model,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total": len(examples),
            "agree": agree,
            "disagree": disagree,
            "parse_errors": parse_errors,
        },
        "details": judged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)
    print(f"\nSaved to {output_path}")

    return output_path
