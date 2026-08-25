"""
Analysis commands - LLM judge for abstention analysis.

Uses the OpenAI Batch API for cost-efficient, rate-limit-free judging.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; the real import is lazy (see below)
    from openai import OpenAI
# The openai package is imported inside the functions that need it. At module
# level it sat on the critical path of every CLI invocation, including
# `--help`, and made the whole CLI unusable when it was absent.

from pipeline.core.io import load_json, save_json
from pipeline.core.utils import extract_think


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

ABSTENTION_JUDGE_SYSTEM_PROMPT = (_PROMPTS_DIR / "abstention_judge.txt").read_text()

ABSTENTION_JUDGE_USER_TEMPLATE = """\
## Model's chain-of-thought (before it abstained)

{generation}

## Correct answer

{ground_truth}

Did the model arrive at the correct answer at any point in its reasoning? Respond in JSON."""


def _format_ground_truth_for_judge(ground_truth: dict, task_name: str) -> str:
    """Format ground truth into a human-readable string for the LLM judge."""
    if task_name == "countdown":
        target = ground_truth.get("target")
        answer = ground_truth.get("answer")
        numbers = ground_truth.get("numbers", [])
        return (
            f"Task: Use the numbers {numbers} with basic arithmetic to reach {target}.\n"
            f"Correct expression: {answer}"
        )
    elif task_name == "competition_math":
        problem = ground_truth.get("problem", "")
        answer = ground_truth.get("answer", "")
        if len(problem) > 300:
            problem = problem[:300] + "..."
        return f"Problem: {problem}\nCorrect answer: {answer}"
    elif task_name == "code_output":
        expected = ground_truth.get("expected_stdout", "")
        return f"Expected program output: {expected}"
    else:
        return f"Ground truth: {json.dumps(ground_truth)}"


def _create_batch_requests(details: list[dict], task_name: str, model: str) -> list[dict]:
    """Create JSONL batch request entries for each abstained example."""
    requests = []
    for detail in details:
        generation = detail["generation"]
        ground_truth_str = _format_ground_truth_for_judge(detail["ground_truth"], task_name)
        user_content = ABSTENTION_JUDGE_USER_TEMPLATE.format(
            generation=generation,
            ground_truth=ground_truth_str,
        )
        requests.append({
            "custom_id": str(detail["index"]),
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": ABSTENTION_JUDGE_SYSTEM_PROMPT,
                "input": user_content,
                "max_output_tokens": 128,
                "reasoning": {"effort": "none"},
                "text": {"format": {"type": "json_object"}},
            },
        })
    return requests


def _submit_and_wait(client: OpenAI, requests: list[dict], poll_interval: int = 30) -> dict:
    """Upload batch JSONL, submit batch job, poll until done, return results by custom_id."""
    # Write requests to temp JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        jsonl_path = f.name

    try:
        # Upload
        print(f"Uploading {len(requests)} requests...")
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"  File ID: {file_obj.id}")

        # Create batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")

        # Poll
        while batch.status not in ("completed", "failed", "cancelled", "expired"):
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch.id)
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else len(requests)
            failed = batch.request_counts.failed if batch.request_counts else 0
            print(f"  Status: {batch.status} ({completed}/{total} completed, {failed} failed)")

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

        # Download results (check error file if no output file)
        print(f"Downloading results...")
        output_file_id = batch.output_file_id
        error_file_id = batch.error_file_id

        if error_file_id:
            error_content = client.files.content(error_file_id).text
            error_lines = error_content.strip().split("\n")[:3]
            print(f"  Batch errors ({len(error_content.strip().split(chr(10)))} total):")
            for line in error_lines:
                print(f"    {line[:300]}")

        if not output_file_id:
            raise RuntimeError(
                f"Batch {batch.id} has no output file. "
                f"All {batch.request_counts.failed} requests failed. See errors above."
            )

        content = client.files.content(output_file_id).text

        # Parse results
        results_by_id = {}
        lines = content.strip().split("\n")

        # Log first entry for debugging
        if lines:
            first = json.loads(lines[0])
            print(f"  Sample response keys: {list(first.keys())}")
            print(f"  Sample response: {json.dumps(first)[:500]}")

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
                    judgment = parsed.get("judgment", "").lower().strip()
                    if judgment in ("yes", "no"):
                        results_by_id[custom_id] = judgment
                        continue
                except json.JSONDecodeError:
                    pass

            # Log first few errors
            if errors < 5:
                status = response_body.get("status", "unknown")
                print(f"  Parse error for custom_id={custom_id}: status={status}, text={text_content!r}")
            errors += 1
            results_by_id[custom_id] = "error"

        if errors:
            print(f"  Total parse errors: {errors}/{len(lines)}")

        return results_by_id

    finally:
        os.unlink(jsonl_path)


def _compute_quality_counts(details: list[dict], judgment_by_index: dict) -> dict:
    """
    Compute TP/FP/TN/FN/precision/recall/F1 for a set of examples.

    Positive class = model answers (does not abstain):
      - TP: didn't abstain + correct   (correctly answered)
      - FP: didn't abstain + wrong     (answered but shouldn't have)
      - TN: abstained + answer_absent  (correctly abstained)
      - FN: abstained + answer_present (had the answer but abstained)
    """
    tp = fp = tn = fn = 0
    errors = 0
    for d in details:
        if d.get("abstained", False):
            judgment = judgment_by_index.get(d["index"])
            if judgment == "no":  # answer_absent → correctly abstained
                tn += 1
            elif judgment == "yes":  # answer_present → should have answered
                fn += 1
            else:
                errors += 1
        else:
            if d.get("correct", False):
                tp += 1  # correctly answered
            else:
                fp += 1  # answered but wrong

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "errors": errors,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _compute_abstention_quality(all_details: list[dict], judged: list[dict], group_by: str = "variant") -> dict:
    """Compute abstention quality metrics overall and per group."""
    judgment_by_index = {j["index"]: j["judgment"] for j in judged}

    # Overall
    overall = _compute_quality_counts(all_details, judgment_by_index)

    # Per group
    by_group = defaultdict(list)
    for d in all_details:
        g = d.get(group_by, d.get("ground_truth", {}).get(group_by, "unknown"))
        by_group[g].append(d)

    per_group = {}
    for group, details in sorted(by_group.items()):
        per_group[group] = _compute_quality_counts(details, judgment_by_index)

    return {**overall, "by_group": per_group}


def analyze_abstention(
    results_path: Path,
    task_name: str,
    output_path: Path | None = None,
    method_name: str | None = None,
    model: str = "gpt-5.4-nano",
    poll_interval: int = 30,
    max_samples: int | None = None,
    group_by: str = "variant",
) -> Path:
    """
    Analyze abstained examples using an LLM judge via the OpenAI Batch API.

    For each abstained example, asks GPT whether the model actually arrived at
    the correct answer in its chain-of-thought before choosing to abstain.
    Then computes precision/recall/F1 for the abstention decision across all examples.

    Args:
        results_path: Path to evaluation results JSON
        task_name: Task name (countdown, competition_math, code_output)
        output_path: Where to save analysis results
        method_name: Method name (for auto-derived output path)
        model: OpenAI model to use as judge
        poll_interval: Seconds between batch status checks

    Returns:
        Path to analysis results file
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Load eval results
    data = load_json(results_path)
    if not isinstance(data, dict) or "details" not in data:
        raise ValueError(f"Expected eval results with 'details' key, got: {type(data)}")

    all_details = data["details"]
    abstained = [d for d in all_details if d.get("abstained", False)]
    non_abstained = [d for d in all_details if not d.get("abstained", False)]

    if not abstained:
        print("No abstained examples found in results.")
        return results_path

    if max_samples is not None:
        abstained = abstained[:max_samples]
        print(f"Limiting to first {max_samples} abstained examples")

    print(f"=== Abstention Analysis (Batch API) ===")
    print(f"Results: {results_path}")
    print(f"Task: {task_name}")
    print(f"Judge model: {model}")
    print(f"Total examples: {len(all_details)}")
    print(f"Abstained: {len(abstained)}")
    print(f"Non-abstained: {len(non_abstained)} ({sum(1 for d in non_abstained if d.get('correct'))} correct, {sum(1 for d in non_abstained if not d.get('correct'))} wrong)")
    print(f"=======================================")

    # Create batch requests and submit
    from openai import OpenAI

    client = OpenAI()
    requests = _create_batch_requests(abstained, task_name, model)
    results_by_id = _submit_and_wait(client, requests, poll_interval)

    # Build judged list
    judged = []
    for detail in abstained:
        index = detail["index"]
        judgment = results_by_id.get(str(index), "error")
        judged.append({
            "index": index,
            "variant": detail.get("variant", "unknown"),
            "judgment": judgment,
        })

    # Abstention judge summary
    present = sum(1 for j in judged if j["judgment"] == "yes")
    absent = sum(1 for j in judged if j["judgment"] == "no")
    error_count = sum(1 for j in judged if j["judgment"] == "error")

    by_group = defaultdict(lambda: {"answer_present": 0, "answer_absent": 0, "error": 0, "total": 0})
    for detail, j in zip(abstained, judged):
        g = detail.get(group_by, detail.get("ground_truth", {}).get(group_by, "unknown"))
        key = {"yes": "answer_present", "no": "answer_absent"}.get(j["judgment"], "error")
        by_group[g][key] += 1
        by_group[g]["total"] += 1

    # Compute full precision/recall/F1
    quality = _compute_abstention_quality(all_details, judged, group_by)

    metrics = {
        "total": len(all_details),
        "total_abstained": len(abstained),
        "answer_present": present,
        "answer_absent": absent,
        "errors": error_count,
        "answer_present_rate": present / len(abstained) if abstained else 0,
        "by_group": dict(by_group),
        "abstention_quality": quality,
    }

    # Print summary
    print(f"\n=== Abstention Judge Results ===")
    print(f"Abstained examples: {len(abstained)}")
    print(f"  Answer present (model had correct answer): {present} ({metrics['answer_present_rate']:.1%})")
    print(f"  Answer absent  (model did not have answer): {absent}")
    if error_count:
        print(f"  Errors: {error_count}")
    print()
    print(f"By {group_by}:")
    for group, counts in sorted(by_group.items()):
        rate = counts["answer_present"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {group}: {counts['answer_present']}/{counts['total']} answer present ({rate:.0%})")

    print(f"\n=== Abstention Quality (full dataset) ===")
    print(f"Confusion matrix (positive = model answers):")
    print(f"                        Predicted")
    print(f"                   Answer    Abstain")
    print(f"  Should answer     {quality['tp']:5d}     {quality['fn']:5d}")
    print(f"  Should abstain    {quality['fp']:5d}     {quality['tn']:5d}")
    print()
    print(f"  Precision: {quality['precision']:.4f} (when it answers, how often is it correct)")
    print(f"  Recall:    {quality['recall']:.4f} (of all it should answer, how many did it answer)")
    print(f"  F1:        {quality['f1']:.4f}")
    print()
    print(f"By {group_by}:")
    print(f"  {group_by:<25s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
    for group, q in sorted(quality["by_group"].items()):
        print(f"  {str(group):<25s}  {q['tp']:4d}  {q['fp']:4d}  {q['tn']:4d}  {q['fn']:4d}  {q['precision']:6.3f}  {q['recall']:6.3f}  {q['f1']:6.3f}")
    print(f"============================================")

    # Determine output path
    if output_path is None:
        stem = results_path.stem
        output_path = results_path.parent / f"{stem}_abstention_analysis.json"

    results = {
        "source": str(results_path),
        "task": task_name,
        "judge_model": model,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "details": judged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved analysis to {output_path}")

    return output_path


def _enumerate_lines(generation: str) -> list[str]:
    """Split a generation's thinking into non-empty lines for judge input.

    Strips structural tags (<think>, </think>, <answer>, <abstain>) and
    returns only reasoning content lines.
    """
    # Extract thinking content (before </think> or <abstain>)
    think_content = extract_think(generation)
    if think_content is None:
        # No </think> tag — take everything (may have hit token limit)
        # Strip leading <think> if present
        text = re.sub(r"^<think>\s*", "", generation, flags=re.IGNORECASE)
        # Strip trailing <abstain> if present
        text = re.sub(r"\s*<abstain>\s*$", "", text)
    else:
        text = think_content
    # Filter out empty lines and lines that are just XML tags
    tag_only = re.compile(r"^\s*</?(?:think|answer|abstain|response|request)[^>]*>\s*$", re.IGNORECASE)
    raw_lines = [line.strip() for line in text.split("\n") if line.strip() and not tag_only.match(line)]
    # Merge $$ ... $$ blocks into single lines
    merged = []
    inside_latex = False
    for line in raw_lines:
        if not inside_latex and line.strip() == "$$":
            inside_latex = True
            merged.append("$$")
        elif inside_latex and line.strip() == "$$":
            merged[-1] += " $$"
            inside_latex = False
        elif inside_latex:
            merged[-1] += " " + line
        else:
            merged.append(line)
    return merged


def preprocess_for_judge(
    results_path: Path,
    task_name: str,
    output_path: Path | None = None,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
) -> Path:
    """
    Preprocess eval results for LLM judge analysis.

    Splits each generation into enumerated lines and saves alongside
    metadata needed for judging (ground truth, correctness, abstention status).
    Also computes per-line token counts for coverage measurement.

    Args:
        results_path: Path to evaluation results JSON
        task_name: Task name (countdown, competition_math, code_output)
        output_path: Where to save preprocessed output
        tokenizer_name: HuggingFace tokenizer to use for per-line token counts

    Returns:
        Path to preprocessed JSON file
    """
    from transformers import AutoTokenizer

    data = load_json(results_path)
    if not isinstance(data, dict) or "details" not in data:
        raise ValueError(f"Expected eval results with 'details' key, got: {type(data)}")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    details = data["details"]

    # Filter out token-limit exhaustion (no useful signal for judge). The limit
    # must come from the run that produced this file -- it was hardcoded to 2048
    # while every recorded evaluation used --max-new-tokens 4096, which silently
    # discarded ~18% of records as "truncated" when they had finished normally.
    token_limit = data.get("config", {}).get("max_new_tokens")
    if token_limit is None:
        token_limit = 2048
        print(
            "Warning: results file has no config.max_new_tokens; "
            f"falling back to {token_limit} for the truncation filter."
        )
    filtered = [d for d in details if d.get("token_count", 0) < token_limit]
    skipped = len(details) - len(filtered)
    if skipped:
        print(f"Filtered out {skipped} examples that hit token limit ({token_limit})")

    examples = []
    for d in filtered:
        generation = d.get("generation", "")
        raw_lines = _enumerate_lines(generation)
        lines = {str(i): line for i, line in enumerate(raw_lines, 1)}
        line_token_counts = {
            k: len(tokenizer.encode(v, add_special_tokens=False))
            for k, v in lines.items()
        }
        ground_truth = d.get("ground_truth", {})

        examples.append({
            "index": d["index"],
            "variant": d.get("variant", "unknown"),
            "correct": d.get("correct", False),
            "abstained": d.get("abstained", False),
            "token_count": d.get("token_count", 0),
            "ground_truth": ground_truth,
            "problem": _format_ground_truth_for_judge(ground_truth, task_name),
            "num_lines": len(lines),
            "lines": lines,
            "line_token_counts": line_token_counts,
        })

    # Stats
    total = len(examples)
    abstained = sum(1 for e in examples if e["abstained"])
    avg_lines = sum(e["num_lines"] for e in examples) / total if total else 0

    print(f"=== Preprocessing for Judge ===")
    print(f"Source: {results_path}")
    print(f"Task: {task_name}")
    print(f"Total examples: {total}")
    print(f"Abstained: {abstained} ({abstained/total:.1%})" if total else "")
    print(f"Avg lines per generation: {avg_lines:.1f}")
    print(f"===============================")

    # Determine output path
    if output_path is None:
        analysis_dir = results_path.parent.parent / "analysis"
        stem = results_path.stem
        output_path = analysis_dir / f"{stem}_preprocessed.json"

    output = {
        "source": str(results_path),
        "task": task_name,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total": total,
            "abstained": abstained,
            "answered": total - abstained,
            "avg_lines": round(avg_lines, 1),
        },
        "examples": examples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)
    print(f"Saved to {output_path}")

    return output_path
