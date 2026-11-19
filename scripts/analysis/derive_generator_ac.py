"""Derive the "Generator for Methods A and C" eval from Method C's chain traces.

Methods A and C share one solver (`artifacts/{task}/models/method_ac_models/
{model}_v3`). Method C records every solver attempt it made in `chain_trace`, and
entry 0 is always the zero-hint attempt, so the generator's standalone accuracy is
already sitting in those files: it never needed an eval run of its own.

Method A cannot supply the same numbers -- its `verifier_trace` holds no solver
generations, and its solver is not even run at zero hints when the verifier decides
to hint first.

`chain_trace` stores the generation but no correctness label, so every level-0
attempt is re-graded here with the task's own `check_correctness`. Rows that never
escalated (`num_hints == 0`) already carry the level-0 verdict in `correct`; those
are re-graded too and the results compared, which checks this grader agrees with the
pipeline that wrote the file.

Writes one eval-shaped JSON per run to `generator_ac/`.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.tasks import get_task  # noqa: E402

METHOD_C_DIR = REPO / "method_c"
OUTPUT_DIR = REPO / "generator_ac"

# (task, run filename) -- shared by method_a/, method_c/ and the workbook tabs
RUNS = [
    ("countdown", "countdown_qwen2.5-1.5b_v3"),
    ("countdown", "countdown_qwen2.5-3b_v3"),
    ("countdown", "countdown_qwen3-4b-base_v3"),
    ("competition_math", "competition_math_qwen2.5-1.5b_v3"),
    ("competition_math", "competition_math_qwen2.5-3b_v3"),
    ("competition_math", "competition_math_qwen3-4b-base_v3"),
]

_TASK = None


def _init(task_name: str) -> None:
    global _TASK
    _TASK = get_task(task_name)


def _grade(job: tuple) -> tuple[bool, str | None]:
    """Grade one zero-hint solver generation. Returns (correct, error)."""
    ground_truth, generation = job
    if not isinstance(generation, str):
        return False, "missing generation"
    try:
        correct, _ = _TASK.check_correctness(ground_truth, generation)
    except Exception as exc:  # surfaced in the summary rather than silently dropped
        return False, f"{type(exc).__name__}: {exc}"
    return bool(correct), None


def level_zero_jobs(rows: list[dict]) -> list[tuple]:
    """Pull the zero-hint attempt out of every row, asserting entry 0 really is it."""
    jobs = []
    for row in rows:
        trace = row.get("chain_trace") or []
        if not trace:
            raise ValueError(f"row {row['index']} has an empty chain_trace")
        head = trace[0]
        if head.get("hint_level") != 0:
            raise ValueError(
                f"row {row['index']} starts at hint_level {head.get('hint_level')}, not 0"
            )
        if len(trace) != row["num_hints"] + 1:
            raise ValueError(
                f"row {row['index']}: len(chain_trace)={len(trace)} "
                f"but num_hints={row['num_hints']}"
            )
        jobs.append((row["ground_truth"], head.get("solver_generation")))
    return jobs


def summarise(details: list[dict]) -> dict:
    """Overall and per-difficulty counts, mirroring the eval JSONs' metrics block."""
    by_variant: dict = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    by_level: dict = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    for row in details:
        for field, table in (("variant", by_variant), ("level", by_level)):
            key = row.get(field)
            if key is None:
                continue
            table[key]["total"] += 1
            table[key]["correct"] += int(row["correct"])
    total = len(details)
    correct = sum(int(row["correct"]) for row in details)
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "counts_by_variant": dict(by_variant),
        "counts_by_level": dict(by_level),
    }


def derive(task_name: str, run_name: str, workers: int) -> dict:
    source = METHOD_C_DIR / run_name
    payload = json.loads(source.read_text())
    rows = payload["details"]
    jobs = level_zero_jobs(rows)

    start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(task_name,)) as pool:
        graded = pool.map(_grade, jobs, chunksize=8)
    elapsed = time.time() - start

    details = []
    agree = disagree = 0
    disagreements = []
    errors = collections.Counter()
    for row, (correct, error) in zip(rows, graded):
        if error:
            errors[error] += 1
        head = row["chain_trace"][0]
        details.append(
            {
                "index": row["index"],
                "source_index": row.get("source_index"),
                "variant": row.get("variant"),
                "level": row.get("level"),
                "num_hints": 0,
                "correct": correct,
                "finish_reason": head.get("solver_finish_reason"),
                "token_count": head.get("solver_token_count"),
            }
        )
        if row["num_hints"] == 0:
            # The pipeline's own verdict on this very generation.
            if correct == bool(row["correct"]):
                agree += 1
            else:
                disagree += 1
                if len(disagreements) < 5:
                    disagreements.append(row["index"])

    result = {
        "solver_model": payload["solver_model"],
        "eval_dataset": payload["eval_dataset"],
        "derived_from": str(source.relative_to(REPO)),
        "derivation": (
            "chain_trace[0] (the zero-hint solver attempt) of every row, re-graded "
            f"with {task_name}.check_correctness"
        ),
        "source_timestamp": payload["timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": payload["config"]["num_samples"],
            "max_hints": 0,
            "solver": payload["config"]["solver"],
        },
        "validation": {
            "note": (
                "Rows that never escalated carry the pipeline's own label for this "
                "same generation; this grader is checked against it."
            ),
            "rows_with_recorded_label": agree + disagree,
            "agreements": agree,
            "disagreements": disagree,
            "disagreement_indices": disagreements,
            "grading_errors": dict(errors),
        },
        "metrics": summarise(details),
        "details": details,
    }

    acc = result["metrics"]["accuracy"]
    flag = "OK " if disagree == 0 else "MISMATCH "
    print(
        f"{flag}{run_name}: acc={acc:.4f} ({result['metrics']['correct']}/"
        f"{result['metrics']['total']})  validation {agree}/{agree + disagree}"
        f"  {elapsed:.1f}s",
        flush=True,
    )
    if errors:
        print(f"    grading errors: {dict(errors)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run", help="derive a single run by filename")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    runs = [r for r in RUNS if args.run is None or r[1] == args.run]
    if not runs:
        raise SystemExit(f"no run matching {args.run!r}")

    for task_name, run_name in runs:
        result = derive(task_name, run_name, args.workers)
        (OUTPUT_DIR / run_name).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
