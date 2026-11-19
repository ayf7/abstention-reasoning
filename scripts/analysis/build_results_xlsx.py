"""Build a one-sheet results workbook per model size from the raw eval files.

The layout is not authored here -- it is inherited from the `cd-1.5b` tab of
`results_v2.xlsx`, which was exported from the Google Sheet and carries the
formulas, the fills, and the white->green colour scale. This script copies that
file, strips every tab but the template, and writes the numbers in.

Copying the file (rather than `openpyxl`'s `copy_worksheet`) is deliberate:
`copy_worksheet` silently drops conditional formatting, so a sheet built that
way loses the colour scale on import.

Usage:
    python scripts/analysis/build_results_xlsx.py                 # all sizes
    python scripts/analysis/build_results_xlsx.py --size 3b
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter as CL

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "results_v2.xlsx"
TEMPLATE_SHEET = "cd-1.5b"
MODELS = Path("/share/goyal/ayf7/artifacts/countdown/models")

# Model directory per size label.
SIZES = {"1.5b": "qwen2.5-1.5b", "3b": "qwen2.5-3b", "4b": "qwen3-4b-base"}

ARMS = ("0.3", "0.4", "0.5", "0.6")
COL0 = {"0.3": 1, "0.4": 10, "0.5": 19, "0.6": 28}  # A, J, S, AB
DIFFICULTIES = (4, 5, 6)
MAX_HINTS = 5  # the h=5 column is really h>=5; longer runs fold into it

# Row anchors inside one Method B arm block.
R_ATT, R_COR, R_PCT, R_PATH = 24, 31, 38, 20
# Row anchors for the two baseline sections: (title row, first data row).
BASELINES = (("sft", 1, 3), ("rl", 8, 12))


def eval_path(kind: str, size: str, arm: str | None = None) -> Path:
    """Absolute path to one eval file. `kind` is 'sft', 'rl' or 'method_b'.

    The RL stage writes into `{method}_models` and the SFT stage into
    `{method}_sft`, so the baseline's finished model is `baseline_models`.
    """
    model = SIZES[size]
    if kind == "sft":
        return MODELS / "baseline_sft" / model / "evals" / "eval.json"
    if kind == "rl":
        return MODELS / "baseline_models" / model / "evals" / "eval.json"
    return MODELS / "method_b_models" / f"{model}__quad-a{arm}" / "evals" / "eval.json"


def difficulty(row: dict) -> int:
    """Countdown encodes difficulty as `variant` ('4_operands'), not `level`."""
    return int(re.search(r"\d+", str(row["variant"])).group())


def tally_hints(details: list[dict]) -> tuple[dict, dict]:
    """(attempted, correct) keyed by (difficulty, hint bucket)."""
    att, cor = defaultdict(int), defaultdict(int)
    for row in details:
        key = (difficulty(row), min(row.get("num_hints") or 0, MAX_HINTS))
        att[key] += 1
        cor[key] += bool(row["correct"])
    return att, cor


def tally_flat(details: list[dict]) -> tuple[dict, dict]:
    """(attempted, correct) keyed by difficulty alone, for the baselines."""
    att, cor = defaultdict(int), defaultdict(int)
    for row in details:
        att[difficulty(row)] += 1
        cor[difficulty(row)] += bool(row["correct"])
    return att, cor


def rel(path: Path) -> str:
    """Path relative to artifacts/, which is how the sheet records sources."""
    return str(path.relative_to(MODELS.parents[1]))


def write_method_b(ws, size: str) -> list[str]:
    """Fill the four arm blocks. Returns the cells given a literal 0."""
    zeroed = []
    for arm in ARMS:
        path = eval_path("method_b", size, arm)
        att, cor = tally_hints(json.loads(path.read_text())["details"])
        c0 = COL0[arm]
        ws.cell(R_PATH, c0 + 1).value = "File path: " + rel(path)
        for i, lv in enumerate(DIFFICULTIES):
            for h in range(MAX_HINTS + 1):
                col, n = c0 + 1 + h, att[(lv, h)]
                ws.cell(R_ATT + i, col).value = n
                ws.cell(R_COR + i, col).value = cor[(lv, h)]
                pct = ws.cell(R_PCT + i, col)
                if n == 0:
                    # The template's own convention for an empty cell: a literal
                    # 0 rather than a #DIV/0! (see F38/G38 as exported).
                    pct.value = 0
                    zeroed.append(f"{CL(col)}{R_PCT + i}")
                else:
                    pct.value = f"={CL(col)}{R_COR + i}/{CL(col)}{R_ATT + i}*100"
    return zeroed


def write_baseline(ws, size: str) -> None:
    """Fill the SFT and RL baseline sections.

    Both now run the Method B protocol -- same 1500-problem eval split, same
    sampling -- so the three sections of the sheet are read against each other
    directly. The note records the truncation rate because it is what the SFT
    numbers are mostly measuring: a rollout that hits the token cap never emits
    an <answer> tag and scores 0, so a low SFT figure is a termination failure
    rather than an arithmetic one.
    """
    for kind, title_row, r0 in BASELINES:
        path = eval_path(kind, size)
        data = json.loads(path.read_text())
        details = data["details"]
        cfg = data["config"]
        att, cor = tally_flat(details)
        n = sum(att.values())
        trunc = sum(1 for r in details if r.get("finish_reason") == "length")

        ws.cell(title_row, 2).value = (
            f"File path: {rel(path)}  |  {n} problems, T={cfg['temperature']}, "
            f"top_p={cfg['top_p']}, {cfg['max_new_tokens']} tok, single turn "
            f"(same protocol as Method B below)  |  {trunc}/{n} "
            f"({trunc / n:.0%}) hit the token cap; a truncated rollout has no "
            f"<answer> tag and scores 0"
        )
        for i, lv in enumerate(DIFFICULTIES):
            r = r0 + i
            ws.cell(r, 2).value = att[lv]
            ws.cell(r, 3).value = cor[lv]
            ws.cell(r, 4).value = f"=C{r}/B{r}*100"
        r = r0 + len(DIFFICULTIES)
        ws.cell(r, 2).value = f"=SUM(B{r0}:B{r - 1})"
        ws.cell(r, 3).value = f"=SUM(C{r0}:C{r - 1})"
        ws.cell(r, 4).value = f"=C{r}/B{r}*100"


def build(size: str, out: Path) -> None:
    shutil.copy(TEMPLATE, out)
    wb = openpyxl.load_workbook(out)
    for name in [n for n in wb.sheetnames if n != TEMPLATE_SHEET]:
        del wb[name]
    ws = wb[TEMPLATE_SHEET]
    ws.title = f"cd-{size}"

    zeroed = write_method_b(ws, size)
    write_baseline(ws, size)

    # openpyxl writes formulas without cached values; this makes the reader
    # compute them on open.
    wb.calculation.fullCalcOnLoad = True
    wb.save(out)

    rules = sum(len(rng.rules) for rng in ws.conditional_formatting)
    print(f"{out.name}: sheet 'cd-{size}', {rules} conditional-format rule(s), "
          f"{len(zeroed)} literal-0 cell(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=sorted(SIZES), help="default: all sizes")
    ap.add_argument("--out-dir", type=Path, default=REPO)
    args = ap.parse_args()

    for size in ([args.size] if args.size else SIZES):
        build(size, args.out_dir / f"results_cd-{size}.xlsx")


if __name__ == "__main__":
    main()
