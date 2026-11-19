"""Difficulty x hint-count tables for the hint arms.

Four tables per arm, all over the same cells: how many problems at each MATH
level ended with each number of hints, and how many of those were solved.
Correctness here is *assisted* -- the hint is in context when the answer is
written -- so this is the companion to the precision-accuracy plane, which
scores a hinted problem as an abstention instead.

Hint counts run past 5 with a long tail (one arm reaches 28), so the last column
is a "5+" bucket rather than a column per value; --max-hints moves the cut.

    R=artifacts/competition_math/hint_ablations/results
    python scripts/analysis/hint_tables.py --out hint_tables.md \
        --results $R/eval_augmented_rl_p0.025_last.json \
                  $R/eval_augmented_rl_p0.05_last.json

Results also live one per run directory as models/<run>/evals/eval.json, where
the arm is the directory rather than the filename. Globs work, and --format tsv
prints tab-separated tables to stdout for pasting into a sheet:

    M=/share/goyal/ayf7/artifacts/competition_math/models
    python scripts/analysis/hint_tables.py --format tsv \
        --tables attempted correct --results "$M/method_b_models/*/evals/eval.json"
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

# Correct, attempted, the two side by side, and the rate. Every table reads the
# same (attempted, correct) pair, so they can only disagree in formatting.
TABLES = {
    "correct": ("Number of Correct", lambda a, c: str(c)),
    "attempted": ("Number of Attempted", lambda a, c: str(a)),
    "fraction": ("Fractional Correct", lambda a, c: f"{c}/{a}"),
    "accuracy": ("Accuracy %", lambda a, c: f"{100 * c / a:.2f}" if a else "--"),
}


def arm_label(path: Path) -> str:
    """Human label for one results file.

    Results land in two shapes: hint_ablations/results/eval_..._p0.05_last.json,
    where the arm is in the filename, and models/<run>/evals/eval.json, where
    every file shares one name and the arm is the run directory.
    """
    m = re.search(r"_p([0-9.]+)_", path.name)
    if m:
        return f"hint_penalty = {m.group(1)}"
    if path.parent.name in ("evals", "results"):
        return path.parent.parent.name
    return path.stem


def sort_key(path: Path) -> tuple:
    """Natural order, so 1.5b sorts before 3b and a0.3 before a0.4.

    Chunks alternate text, number, text, ...; each is tagged so a text chunk is
    never compared against a numeric one when labels have different shapes.
    """
    parts = re.split(r"([0-9]+\.?[0-9]*)", arm_label(path))
    return tuple((0, float(s), "") if i % 2 else (1, 0.0, s)
                 for i, s in enumerate(parts))


def tally(details: list, cap: int) -> tuple:
    """(attempted, correct) per (level, hint bucket), plus unlabelled rows."""
    att, cor = defaultdict(int), defaultdict(int)
    unlabelled = 0
    for r in details:
        # Difficulty lives in `level` for competition_math ("Level 4") and in
        # `variant` for countdown ("4_operands"). The math eval set carries one
        # "Level ?" problem, which is why its totals come to 1763 and not 1764.
        m = (re.search(r"\d+", str(r.get("level") or ""))
             or re.search(r"\d+", str(r.get("variant") or "")))
        if not m:
            unlabelled += 1
            continue
        key = (int(m.group()), min(r.get("num_hints") or 0, cap))
        att[key] += 1
        cor[key] += bool(r["correct"])
    return att, cor, unlabelled


def grid(att: dict, cor: dict, levels: list, cap: int) -> list:
    """Rows of (label, [(attempted, correct) per bucket], (attempted, correct))."""
    rows = []
    for lv in levels + [None]:
        src = levels if lv is None else [lv]
        cells = [(sum(att[(l, h)] for l in src), sum(cor[(l, h)] for l in src))
                 for h in range(cap + 1)]
        rows.append((str(lv) if lv is not None else "Total", cells,
                     (sum(a for a, _ in cells), sum(c for _, c in cells))))
    return rows


def render_md(title: str, fmt, rows: list, cap: int) -> str:
    head = ["Difficulty", "Count"] + [f"{h} Hints" if h < cap else f"{cap}+ Hints"
                                      for h in range(cap + 1)]
    body = "\n".join(
        "| " + " | ".join(
            [f"**{label}**" if label == "Total" else label, str(total[0])]
            + [fmt(a, c) for a, c in cells]
        ) + " |"
        for label, cells, total in rows
    )
    sep = ["---"] + ["---:"] * (len(head) - 1)
    return (f"### {title}\n\n| " + " | ".join(head) + " |\n| "
            + " | ".join(sep) + " |\n" + body + "\n")


def render_tsv(title: str, fmt, rows: list, cap: int) -> str:
    head = ["Difficulty"] + [f"h={h}" if h < cap else f"h={cap}+"
                             for h in range(cap + 1)] + ["Total"]
    body = "\n".join(
        "\t".join([label] + [fmt(a, c) for a, c in cells] + [fmt(*total)])
        for label, cells, total in rows
    )
    return f"{title}\n" + "\t".join(head) + "\n" + body + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, help="Results JSON paths or globs")
    ap.add_argument("--out", help="Write here instead of stdout")
    ap.add_argument("--format", choices=["md", "tsv"], default="md")
    ap.add_argument("--tables", nargs="+", choices=list(TABLES), default=list(TABLES),
                    help="Which tables to emit, in order")
    ap.add_argument("--max-hints", type=int, default=5,
                    help="Last column buckets this many hints and above")
    args = ap.parse_args()

    paths = sorted({Path(p) for pattern in args.results for p in glob.glob(pattern)},
                   key=sort_key)
    if not paths:
        raise SystemExit(f"no results matched: {args.results}")

    render = render_md if args.format == "md" else render_tsv
    out = ["# Hint usage by difficulty",
           "",
           "Assisted correctness: a problem counts as solved if the final answer is "
           "right, whatever it cost in hints.",
           ""] if args.format == "md" else []

    for path in paths:
        data = json.loads(path.read_text())
        att, cor, unlabelled = tally(data["details"], args.max_hints)
        levels = sorted({lv for lv, _ in att})
        rows = grid(att, cor, levels, args.max_hints)
        note = f" ({unlabelled} problem without a difficulty label omitted)" if unlabelled else ""
        if args.format == "md":
            out += [f"## {arm_label(path)}", "", f"`{path}`{note}", ""]
        else:
            out += [f"# {arm_label(path)}{note}", ""]
        out += [render(*TABLES[t], rows, args.max_hints) for t in args.tables]

    text = "\n".join(out).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(paths)} arms)")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
