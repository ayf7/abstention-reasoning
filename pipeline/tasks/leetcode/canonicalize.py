"""Rewrite a generated dataset into the canonical answer format and re-score it.

    python -m pipeline.tasks.leetcode.canonicalize IN.json OUT.json

Each record's generation goes through `canonicalize`; rewritten records are
re-verified and their `correct` / `metadata` replaced. Records that cannot be
rewritten (thinking never closed, no code after it) are kept unchanged.
"""

import argparse
from collections import Counter
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.tasks.leetcode.task import LeetCodeTask, canonicalize


def canonicalize_dataset(records: list[dict]) -> tuple[list[dict], Counter]:
    rewritten = {i: g for i, r in enumerate(records) if (g := canonicalize(r["generation"])) is not None}
    idx = list(rewritten)
    scored = LeetCodeTask().check_correctness_batch([records[i]["ground_truth"] for i in idx],
                                                    [rewritten[i] for i in idx])
    out = [dict(r) for r in records]
    for i, (ok, meta) in zip(idx, scored):
        out[i].update(generation=rewritten[i], correct=ok, metadata=meta, canonicalized=True)
    stats = Counter(r["metadata"]["verdict"] for r in out)
    stats["not_rewritable"] = len(records) - len(idx)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("output must differ from input")
    records, stats = canonicalize_dataset(load_json(args.input))
    save_json(args.output, records)
    print(f"{args.input.name}: {dict(stats)} -> {args.output}")


if __name__ == "__main__":
    main()
