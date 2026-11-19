#!/usr/bin/env python
"""Build results_v2_final.xlsx from results_v2_new.xlsx.

`cd-1.5b` in the source workbook is the formatting template: SFT / RL baseline
tables, a four-arm Hint-Reasoning (Method B) section, a single-arm Methods A&C
section, and -- below it, in plain font -- the raw pasted source the A&C section
was built from (recognisable because its `total` column sits on the LEFT).

This script:

  * fills the SFT and RL baseline tables on every tab from the eval JSONs;
  * rebuilds `cd-3b`, `cd-4b`, `math-1.5b`, `math-3b`, `math-4b` with the same
    layout and cell styles as `cd-1.5b`;
  * fills Method B on those five tabs from method_b_models/*/evals/eval.json;
  * fills Methods A&C on those five tabs from the raw blocks already pasted
    into the source workbook (no eval JSON exists for those runs).

`cd-1.5b` itself is left alone apart from its two baseline tables, which is the
only thing it was missing.

The math tabs carry five difficulty rows (MATH levels 1-5) against countdown's
three (4/5/6 operands), so the layout is generated from a row-anchor function
rather than copied cell-for-cell.

Idempotent: re-run it once the pending competition_math baseline evals land.
"""

import json
import re
import shutil
from collections import defaultdict
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import column_index_from_string, get_column_letter as col_letter

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results_v2_new.xlsx"
OUTPUT = REPO / "results_v2_final.xlsx"
ARTIFACTS = Path("/share/goyal/ayf7/artifacts")

TEMPLATE_TAB = "cd-1.5b"
ARMS = ("0.3", "0.4", "0.5", "0.6")
ARM_COL = {"0.3": 1, "0.4": 10, "0.5": 19, "0.6": 28}  # A, J, S, AB
AC_ARM = "0.5"  # Methods A&C only ever ran at alpha = 0.5
BUCKETS = 6  # h=0 .. h=5; the h=5 column aggregates 5 *or more* hints
BANNER_COLS = 36  # the cream section banners span A:AJ
TEMPLATE_A_WIDTH = 6.4  # cd-1.5b's difficulty-column width

# tab -> (task, model directory, difficulty rows, rebuild?)
TABS = [
    ("cd-1.5b",   "countdown",        "qwen2.5-1.5b",  (4, 5, 6),       False),
    ("cd-3b",     "countdown",        "qwen2.5-3b",    (4, 5, 6),       True),
    ("cd-4b",     "countdown",        "qwen3-4b-base", (4, 5, 6),       True),
    ("math-1.5b", "competition_math", "qwen2.5-1.5b",  (1, 2, 3, 4, 5), True),
    ("math-3b",   "competition_math", "qwen2.5-3b",    (1, 2, 3, 4, 5), True),
    ("math-4b",   "competition_math", "qwen3-4b-base", (1, 2, 3, 4, 5), True),
]

BLOCK_TITLES = (
    "Num attempted",
    "Number correct",
    "Percentage Correct, Per (Difficulty, Hint) Pair",
    "Hint Distribution, Per Difficulty",
)
MB_TITLE = "Hint-Reasoning Model (Method B)"
MB_SUBTITLE = ("Quadratic Penalties: a*(-0.1, -0.2, -0.3, -0.4); "
               "a = multiplier factor (higher = harsher penalty)")
AC_TITLE = "Question and Hint Generation Method (Methods A&C)"
AC_SUBTITLE = "Reward shaping: quadratic with alpha = 0.5"


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
class Layout:
    """Row anchors for a tab with `n` difficulty rows.

    Calibrated so that n=3 reproduces cd-1.5b's rows exactly.
    """

    def __init__(self, n):
        self.n = n
        self.sft_title, self.sft_path, self.sft_hdr = 1, 3, 5
        self.sft_data, self.sft_total = 6, 6 + n
        self.rl_title, self.rl_path, self.rl_hdr = 8 + n, 10 + n, 12 + n
        self.rl_data, self.rl_total = 13 + n, 13 + 2 * n
        self.mb_title, self.mb_subtitle, self.mb_arm = 15 + 2 * n, 16 + 2 * n, 18 + 2 * n
        self.mb_blocks = [20 + 2 * n + k * (n + 4) for k in range(4)]
        self.ac_title = self.block_total(self.mb_blocks[3]) + 2
        self.ac_subtitle, self.ac_arm = self.ac_title + 1, self.ac_title + 3
        self.ac_blocks = [self.ac_title + 5 + k * (n + 4) for k in range(4)]
        self.raw = [self.block_total(self.ac_blocks[3]) + 5 + j * (n + 2) for j in range(2)]
        self.notes = self.raw[-1] + n + 3

    def block_data(self, title_row):
        return title_row + 2

    def block_total(self, title_row):
        return title_row + 2 + self.n


# --------------------------------------------------------------------------
# styles: copied out of the template tab, keyed by role
# --------------------------------------------------------------------------
class Styles:
    """Style rows lifted from cd-1.5b, addressed by role.

    Rows 43/50 (not 42/49) are used for percentage data because the template's
    first percentage row carries hand-edited literals and a stray border.
    """

    ROLE_ROWS = {
        "banner": 1, "subtitle": 22, "path": 3, "arm": 24,
        "base_hdr": 5, "base_data": 6, "base_last": 8, "base_total": 9,
        "sub_title": 26, "hdr": 27, "count_data": 28, "count_total": 31,
        "pct_first": 42, "pct_data": 43, "pct_last": 44, "pct_total": 45,
        "dist_first": 49, "dist_data": 50, "dist_last": 51, "dist_total": 52,
        "raw": 90,
    }

    def __init__(self, ws):
        self.s = {role: [copy(ws.cell(row, c)._style) for c in range(1, BANNER_COLS + 1)]
                  for role, row in self.ROLE_ROWS.items()}

    def apply(self, cell, role, offset):
        """Style `cell` as the template's `role` row at relative column `offset`."""
        cell._style = copy(self.s[role][offset])


# --------------------------------------------------------------------------
# data sources
# --------------------------------------------------------------------------
def eval_path(task, kind, model, arm=None):
    base = ARTIFACTS / task / "models"
    if kind == "sft":
        return base / "baseline_sft" / model / "evals" / "eval.json"
    if kind == "rl":
        return base / "baseline_models" / model / "evals" / "eval.json"
    return base / "method_b_models" / f"{model}__quad-a{arm}" / "evals" / "eval.json"


def difficulty_of(task, row):
    """Countdown encodes difficulty in `variant` ('4_operands'); MATH in `level`."""
    field = row["variant"] if task == "countdown" else row.get("level")
    match = re.search(r"\d+", str(field))
    return int(match.group()) if match else None


def tally(path, task, difficulties):
    """-> (attempted grid, correct grid, n_rows, n_dropped).

    Grids are indexed [difficulty][hint bucket]; hint counts of 5 or more all
    land in the last bucket. Rows whose difficulty cannot be parsed (MATH has a
    single 'Level ?' problem) are dropped, which is also what the Methods A&C
    source did -- keeping them would make the two sections disagree on N.
    """
    details = json.load(open(path))["details"]
    attempted, correct = defaultdict(int), defaultdict(int)
    dropped = 0
    for row in details:
        d = difficulty_of(task, row)
        if d not in difficulties:
            dropped += 1
            continue
        h = min(int(row.get("num_hints") or 0), BUCKETS - 1)
        attempted[(d, h)] += 1
        if row.get("correct"):
            correct[(d, h)] += 1
    grid = lambda t: [[t[(d, h)] for h in range(BUCKETS)] for d in difficulties]
    return grid(attempted), grid(correct), len(details), dropped


def read_raw_blocks(ws, n):
    """Pull the two pasted A&C grids (attempted, correct) off an unformatted tab.

    They are found by their 'variant' header; the `total` column sits in B and
    the hint buckets run from C rightwards, which is the giveaway that a block
    is raw source rather than a formatted table.
    """
    headers = [r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "variant"]
    if len(headers) < 2:
        raise ValueError(f"{ws.title}: expected >=2 raw blocks, found {len(headers)}")
    blocks = []
    for header in headers[:2]:
        grid = []
        for r in range(header + 1, header + 1 + n):
            row = [int(ws.cell(r, 3 + b).value or 0) for b in range(BUCKETS)]
            total = ws.cell(r, 2).value
            if total is not None and int(total) != sum(row):
                raise ValueError(f"{ws.title} r{r}: total {int(total)} != sum {sum(row)}")
            grid.append(row)
        blocks.append(grid)
    return blocks


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_banner(ws, st, row, text):
    for c in range(1, BANNER_COLS + 1):
        st.apply(ws.cell(row, c), "banner", c - 1)
    ws.cell(row, 1).value = text


def write_label_row(ws, st, row, role, offset, labels):
    for i in range(8):
        st.apply(ws.cell(row, offset + i), role, i)
    for i, text in labels.items():
        ws.cell(row, offset + i).value = text


def write_baseline(ws, st, lay, kind, task, model, difficulties):
    """SFT / RL table: Difficulty | Count | Correct | Acc%."""
    title_row, path_row, hdr = (
        (lay.sft_title, lay.sft_path, lay.sft_hdr) if kind == "sft"
        else (lay.rl_title, lay.rl_path, lay.rl_hdr))
    data_row = hdr + 1
    total_row = hdr + 1 + len(difficulties)

    write_banner(ws, st, title_row, f"{kind.upper()} Baseline Model")
    for c in range(1, 9):
        st.apply(ws.cell(path_row, c), "path", c - 1)
    ws.cell(path_row, 1).value = "File path:"

    for i, text in enumerate(("Difficulty", "Count", "Correct", "Acc%")):
        st.apply(ws.cell(hdr, 1 + i), "base_hdr", i)
        ws.cell(hdr, 1 + i).value = text

    path = eval_path(task, kind, model)
    have = path.exists()
    ws.cell(path_row, 2).value = str(path) if have else f"PENDING - not yet evaluated: {path}"
    if have:
        attempted, correct, _, _ = tally(path, task, difficulties)

    for i, d in enumerate(difficulties):
        r = data_row + i
        role = "base_last" if i == len(difficulties) - 1 else "base_data"
        for c in range(4):
            st.apply(ws.cell(r, 1 + c), role, c)
        ws.cell(r, 1).value = d
        ws.cell(r, 4).number_format = "0.00"
        if have:
            ws.cell(r, 2).value = sum(attempted[i])
            ws.cell(r, 3).value = sum(correct[i])
            ws.cell(r, 4).value = f"=C{r}/B{r}*100" if sum(attempted[i]) else 0

    for c in range(4):
        st.apply(ws.cell(total_row, 1 + c), "base_total", c)
    ws.cell(total_row, 1).value = "Total"
    ws.cell(total_row, 4).number_format = "0.00"
    if have:
        ws.cell(total_row, 2).value = f"=SUM(B{data_row}:B{total_row - 1})"
        ws.cell(total_row, 3).value = f"=SUM(C{data_row}:C{total_row - 1})"
        ws.cell(total_row, 4).value = f"=C{total_row}/B{total_row}*100"


def write_grid(ws, st, lay, block_rows, offset, difficulties, attempted, correct):
    """The four stacked tables: attempted / correct / % correct / hint mix.

    `attempted` and `correct` are [difficulty][hint bucket] count grids. Only the
    counts are written as literals; everything below them is a formula, so the
    sheet still recomputes if a count is edited.
    """
    n = lay.n
    L = lambda i: col_letter(offset + i)
    att, cor = lay.block_data(block_rows[0]), lay.block_data(block_rows[1])
    att_tot, cor_tot = lay.block_total(block_rows[0]), lay.block_total(block_rows[1])

    header = {0: "Difficulty", 7: "Total"}
    header.update({1 + b: f"h={b}" for b in range(BUCKETS)})
    for k, title_row in enumerate(block_rows):
        write_label_row(ws, st, title_row, "sub_title", offset, {0: BLOCK_TITLES[k]})
        write_label_row(ws, st, title_row + 1, "hdr", offset, header)

    # --- the two count tables ------------------------------------------------
    for grid, data_row in ((attempted, att), (correct, cor)):
        total_row = data_row + n
        for i in range(n):
            r = data_row + i
            for c in range(8):
                st.apply(ws.cell(r, offset + c), "count_data", c)
            ws.cell(r, offset).value = difficulties[i]
            for b in range(BUCKETS):
                ws.cell(r, offset + 1 + b).value = grid[i][b]
            ws.cell(r, offset + 7).value = f"=SUM({L(1)}{r}:{L(6)}{r})"
        for c in range(8):
            st.apply(ws.cell(total_row, offset + c), "count_total", c)
        ws.cell(total_row, offset).value = "Total"
        for c in range(1, 8):
            ws.cell(total_row, offset + c).value = (
                f"=SUM({L(c)}{data_row}:{L(c)}{total_row - 1})")

    def column_totals(grid):
        """Per-bucket totals plus the grand total: the table's own Total row."""
        return [sum(row[b] for row in grid) for b in range(BUCKETS)] + [
            sum(sum(row) for row in grid)]

    def styled_row(row, family, i, is_total):
        """Border variants: the Total column is boxed across the data rows."""
        for c in range(8):
            if is_total:
                role = f"{family}_total"
            elif c < 7 or 0 < i < n - 1:
                role = f"{family}_data"
            else:
                role = f"{family}_first" if i == 0 else f"{family}_last"
            st.apply(ws.cell(row, offset + c), role, c)

    # --- percentage correct: correct / attempted, guarded at zero denominators
    pct = lay.block_data(block_rows[2])
    for i in range(n + 1):
        r, is_total = pct + i, i == n
        styled_row(r, "pct", i, is_total)
        denoms = (column_totals(attempted) if is_total
                  else attempted[i] + [sum(attempted[i])])
        num_row, den_row = (cor_tot, att_tot) if is_total else (cor + i, att + i)
        ws.cell(r, offset).value = "Total" if is_total else difficulties[i]
        for c in range(1, 8):
            ws.cell(r, offset + c).value = (
                f"={L(c)}{num_row}/{L(c)}{den_row}*100" if denoms[c - 1] else 0)

    # --- hint distribution: share of that difficulty's CORRECT answers --------
    dist = lay.block_data(block_rows[3])
    for i in range(n + 1):
        r, is_total = dist + i, i == n
        styled_row(r, "dist", i, is_total)
        src_row = cor_tot if is_total else cor + i
        denom = sum(sum(row) for row in correct) if is_total else sum(correct[i])
        ws.cell(r, offset).value = "Total" if is_total else difficulties[i]
        for c in range(1, 7):
            ws.cell(r, offset + c).value = (
                f"={L(c)}{src_row}/{L(7)}{src_row}*100" if denom else 0)
        ws.cell(r, offset + 7).value = (
            f"=SUM({L(1)}{r}:{L(6)}{r})" if is_total else "-")


def write_raw(ws, st, lay, difficulties, labels, blocks):
    """Re-emit the pasted A&C source in plain font, `total` on the left."""
    for j, grid in enumerate(blocks):
        header = lay.raw[j]
        for c in range(1, 9):
            st.apply(ws.cell(header, c), "raw", c - 1)
        ws.cell(header, 1).value = "variant"
        ws.cell(header, 2).value = "total"
        for b in range(BUCKETS):
            ws.cell(header, 3 + b).value = b
        for i in range(len(difficulties)):
            r = header + 1 + i
            for c in range(1, 9):
                st.apply(ws.cell(r, c), "raw", c - 1)
            ws.cell(r, 1).value = labels[i]
            ws.cell(r, 2).value = sum(grid[i])
            for b in range(BUCKETS):
                ws.cell(r, 3 + b).value = grid[i][b]


def add_color_scale(ws, lay):
    """White-to-green scale over the two percentage tables of every section."""
    rule = ColorScaleRule(start_type="formula", start_value="0", start_color="FFFFFFFF",
                          end_type="formula", end_value="100", end_color="FF57BB8A")
    ranges = []
    for blocks, offsets in ((lay.mb_blocks, list(ARM_COL.values())),
                            (lay.ac_blocks, [1])):
        for offset in offsets:
            for k in (2, 3):
                ranges.append(f"{col_letter(offset + 1)}{lay.block_data(blocks[k])}:"
                              f"{col_letter(offset + 6)}{lay.block_total(blocks[k])}")
    ws.conditional_formatting.add(" ".join(ranges), rule)


PERCENT_FORMULA = re.compile(r"^=([A-Z]+\d+)/([A-Z]+\d+)\*100$")
SUM_FORMULA = re.compile(r"^=SUM\(([A-Z]+)(\d+):([A-Z]+)(\d+)\)$")


def resolve(ws, addr):
    """Recursively evaluate a cell. Handles the literals, SUM ranges and
    `a/b*100` ratios that are the only formulas these sheets contain."""
    value = ws[addr].value
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value
    match = SUM_FORMULA.match(str(value))
    if match:
        c1, r1 = column_index_from_string(match.group(1)), int(match.group(2))
        c2, r2 = column_index_from_string(match.group(3)), int(match.group(4))
        return sum(resolve(ws, f"{col_letter(c)}{r}")
                   for r in range(r1, r2 + 1) for c in range(c1, c2 + 1))
    match = PERCENT_FORMULA.match(str(value))
    if match:
        denominator = resolve(ws, match.group(2))
        if denominator == 0:
            return 0
        return resolve(ws, match.group(1)) / denominator * 100
    return 0


def repair_zero_denominators(ws):
    """Replace `=a/b*100` with a literal 0 wherever b is zero.

    cd-1.5b was filled in by hand and left eight such cells behind; they render
    as #DIV/0! on import. The sheet already uses a literal 0 for this case
    elsewhere (F42, G42, F75), so this just applies that convention uniformly.
    """
    repaired = []
    for row in ws.iter_rows():
        for cell in row:
            match = PERCENT_FORMULA.match(str(cell.value or ""))
            if match and resolve(ws, match.group(2)) == 0:
                cell.value = 0
                repaired.append(cell.coordinate)
    return repaired


def clear(ws):
    """Blank a tab completely -- values and styles -- before rebuilding it."""
    for row in ws.iter_rows():
        for cell in row:
            cell.value = None
            cell.style = "Normal"
    ws.conditional_formatting._cf_rules.clear()


def build():
    shutil.copy(SOURCE, OUTPUT)
    src = load_workbook(SOURCE)
    wb = load_workbook(OUTPUT)
    st = Styles(wb[TEMPLATE_TAB])
    report = []

    for tab, task, model, difficulties, rebuild in TABS:
        ws = wb[tab]
        n = len(difficulties)
        lay = Layout(n)
        labels = ([f"{d}_operands" for d in difficulties] if task == "countdown"
                  else list(difficulties))

        raw_blocks = read_raw_blocks(src[tab], n) if rebuild else None
        if rebuild:
            clear(ws)

        write_baseline(ws, st, lay, "sft", task, model, difficulties)
        write_baseline(ws, st, lay, "rl", task, model, difficulties)
        for kind in ("sft", "rl"):
            if not eval_path(task, kind, model).exists():
                report.append(f"{tab}: {kind.upper()} baseline PENDING")

        if not rebuild:
            fixed = repair_zero_denominators(ws)
            if fixed:
                report.append(f"{tab}: zeroed {len(fixed)} #DIV/0! cells "
                              f"({', '.join(fixed)})")
            continue

        write_banner(ws, st, lay.mb_title, MB_TITLE)
        ws.cell(lay.mb_subtitle, 1).value = MB_SUBTITLE
        st.apply(ws.cell(lay.mb_subtitle, 1), "subtitle", 0)
        for arm in ARMS:
            offset = ARM_COL[arm]
            write_label_row(ws, st, lay.mb_arm, "arm", offset,
                            {0: f"a={arm}", 1: "File path: "})
            path = eval_path(task, "method_b", model, arm)
            ws.cell(lay.mb_arm, offset + 2).value = str(path)
            attempted, correct, n_rows, dropped = tally(path, task, difficulties)
            if dropped:
                report.append(f"{tab} a={arm}: dropped {dropped}/{n_rows} unparseable difficulty")
            write_grid(ws, st, lay, lay.mb_blocks, offset, difficulties,
                       attempted, correct)

        write_banner(ws, st, lay.ac_title, AC_TITLE)
        ws.cell(lay.ac_subtitle, 1).value = AC_SUBTITLE
        st.apply(ws.cell(lay.ac_subtitle, 1), "subtitle", 0)
        write_label_row(ws, st, lay.ac_arm, "arm", 1,
                        {0: f"a={AC_ARM}", 1: "File path: "})
        ws.cell(lay.ac_arm, 3).value = (
            f"pasted source, results_v2_new.xlsx!{tab} "
            f"(reproduced below at rows {lay.raw[0]}-{lay.raw[-1] + n})")
        write_grid(ws, st, lay, lay.ac_blocks, 1, difficulties,
                   raw_blocks[0], raw_blocks[1])
        write_raw(ws, st, lay, difficulties, labels, raw_blocks)

        note = ("Notes: the h=5 column aggregates 5 or more hints. "
                "Hint Distribution is the share of each difficulty's CORRECT answers. "
                "Methods A&C has no eval JSON; its numbers are the pasted source shown above.")
        if task == "competition_math":
            note += " Problems with an unparseable MATH level are dropped, matching the A&C source."
        ws.cell(lay.notes, 1).value = note
        st.apply(ws.cell(lay.notes, 1), "raw", 0)

        add_color_scale(ws, lay)
        ws.column_dimensions["A"].width = TEMPLATE_A_WIDTH

    wb.calculation.fullCalcOnLoad = True
    wb.save(OUTPUT)
    print(f"wrote {OUTPUT}")
    for line in report:
        print("  !", line)


if __name__ == "__main__":
    build()
