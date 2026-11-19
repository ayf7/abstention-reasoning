"""Build results_v3.xlsx from results_v2.xlsx.

Section order on every tab:

    SFT Baseline -> RL Baseline -> Generator for Methods A and C (empty)
    -> A vs B vs C side by side -> Method B alpha ablation

Method A and Method C counts come from the v3 eval JSONs in method_a/ and
method_c/. The side-by-side's Method B column links to the alpha = 0.5 arm of
the ablation below it, so the ablation stays the single source for those counts.

"Generator for Methods A and C" is derived rather than run: Methods A and C share
one solver, and Method C records its zero-hint attempt for every problem, so
scripts/analysis/derive_generator_ac.py pools and re-grades those into
generator_ac/.
"""

from __future__ import annotations

import collections
import json
import re
from copy import copy
from pathlib import Path

import openpyxl
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.formatting.formatting import ConditionalFormattingList
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results_v2.xlsx"
OUTPUT = REPO / "results_v3.xlsx"
METHOD_A_DIR = REPO / "method_a"
METHOD_C_DIR = REPO / "method_c"
GENERATOR_DIR = REPO / "generator_ac"

TEMPLATE_TAB = "cd-1.5b"
BUCKETS = 6  # h=0 .. h=5; the h=5 column aggregates 5 *or more* hints
BANNER_COLS = 36
ARM_OFFSETS = (0, 9, 18, 27)  # the four Method B arms: one 8-wide table + a spacer
SBS_OFFSETS = (0, 9, 18)  # side by side: Method A | Method B (a=0.5) | Method C
ALPHA_ARM = 18  # column offset of the a = 0.5 arm inside the ablation

# tab -> (task, v3 eval filename, shared by method_a/ and method_c/, difficulty rows)
TABS = [
    ("cd-1.5b", "countdown", "countdown_qwen2.5-1.5b_v3", (4, 5, 6)),
    ("cd-3b", "countdown", "countdown_qwen2.5-3b_v3", (4, 5, 6)),
    ("cd-4b", "countdown", "countdown_qwen3-4b-base_v3", (4, 5, 6)),
    ("math-1.5b", "competition_math", "competition_math_qwen2.5-1.5b_v3", (1, 2, 3, 4, 5)),
    ("math-3b", "competition_math", "competition_math_qwen2.5-3b_v3", (1, 2, 3, 4, 5)),
    ("math-4b", "competition_math", "competition_math_qwen3-4b-base_v3", (1, 2, 3, 4, 5)),
]

HEADER = ["Difficulty", "h=0", "h=1", "h=2", "h=3", "h=4", "h=5", "Total"]
BLOCK_TITLES = [
    "Num attempted",
    "Number correct",
    "Percentage Correct, Per (Difficulty, Hint) Pair",
    "Hint Distribution, Per Difficulty",
]

SBS_TITLE = "Methods A / B / C, Side by Side"
SBS_SUBTITLE = (
    "All three at alpha = 0.5. The Method B column links to the alpha ablation below, "
    "which stays the source for its counts."
)
ABLATION_TITLE = "Hint-Reasoning Model (Method B) - Alpha Ablation"


class Layout:
    """Row anchors for one tab, parameterised by the number of difficulty rows.

    SFT and RL keep the rows they have in results_v2; everything below them is
    torn down and rebuilt in the new order.
    """

    def __init__(self, n: int):
        self.n = n
        self.sft_title, self.sft_path, self.sft_hdr = 1, 3, 5
        self.sft_data, self.sft_total = 6, 6 + n
        self.rl_title, self.rl_path, self.rl_hdr = 8 + n, 10 + n, 12 + n
        self.rl_data, self.rl_total = 13 + n, 13 + 2 * n

        self.gen_title = self.rl_total + 2
        self.gen_subtitle = self.gen_title + 1
        self.gen_hdr = self.gen_title + 3
        self.gen_data = self.gen_title + 4
        self.gen_total = self.gen_data + n

        self.sbs_title = self.gen_total + 3
        self.sbs_subtitle = self.sbs_title + 1
        self.sbs_arm = self.sbs_title + 3
        self.sbs_blocks = [self.sbs_title + 5 + k * (n + 4) for k in range(4)]

        self.ab_title = self.block_total(self.sbs_blocks[3]) + 3
        self.ab_subtitle = self.ab_title + 1
        self.ab_arm = self.ab_title + 3
        self.ab_blocks = [self.ab_title + 5 + k * (n + 4) for k in range(4)]
        self.end = self.block_total(self.ab_blocks[3])

    def block_data(self, title_row: int) -> int:
        return title_row + 2

    def block_total(self, title_row: int) -> int:
        return title_row + 2 + self.n


class TemplateRows:
    """Style source rows in the results_v2 cd-1.5b tab (n == 3).

    Only styling is reused. The template's own section order is the pre-v3 one,
    so its Method B banner -- the one region already styled across all four
    column groups -- is the source for both multi-group sections.
    """

    gen_title, gen_subtitle, gen_hdr, gen_data, gen_total = 54, 55, 57, 58, 61
    group_title, group_subtitle, group_arm = 21, 22, 24
    group_blocks = (26, 33, 40, 47)
    blank = 63
    span = 80  # rows snapshotted before the template tab is itself rewritten


def data_style_rows(src_first: int, src_mid: int, src_last: int, n: int) -> list[int]:
    """Style rows for n data rows: first, middle repeated, last."""
    if n == 1:
        return [src_first]
    return [src_first] + [src_mid] * (n - 2) + [src_last]


def difficulty_of(task: str, row: dict) -> int | None:
    """Countdown encodes difficulty in `variant` ('4_operands'); MATH in `level`."""
    field = row["variant"] if task == "countdown" else row.get("level")
    match = re.search(r"\d+", str(field))
    return int(match.group()) if match else None


def tally(path: Path, task: str, difficulties: tuple[int, ...]):
    """Fold an eval JSON into (difficulty, hint-bucket) attempted/correct grids."""
    payload = json.loads(path.read_text())
    attempted: dict = collections.defaultdict(int)
    correct: dict = collections.defaultdict(int)
    dropped = 0
    for row in payload["details"]:
        level = difficulty_of(task, row)
        if level not in difficulties:
            dropped += 1
            continue
        bucket = min(row["num_hints"], BUCKETS - 1)
        attempted[(level, bucket)] += 1
        if row["correct"]:
            correct[(level, bucket)] += 1
    return attempted, correct, len(payload["details"]), dropped


def tally_generator(path: Path, task: str, difficulties: tuple[int, ...]):
    """Per-difficulty counts for the zero-hint generator, which has no hint axis."""
    attempted, correct, rows, dropped = tally(path, task, difficulties)
    fold = lambda grid: {  # noqa: E731
        level: sum(grid[(level, b)] for b in range(BUCKETS)) for level in difficulties
    }
    return fold(attempted), fold(correct), rows, dropped


def read_ablation(ws, n: int, difficulties: tuple[int, ...]) -> dict:
    """Lift the four Method B arms out of the pre-v3 layout, before it is cleared.

    Only the counts, labels and paths are kept: the derived cells are rewritten
    as equations at the new location, which also retires the hand-patched zeros
    that stood in for divisions by an empty hint bucket.
    """
    arm_row = 18 + 2 * n
    blocks = [20 + 2 * n + k * (n + 4) for k in range(4)]
    arms = []
    for off in ARM_OFFSETS:
        grids = []
        for title_row in blocks[:2]:
            first = title_row + 2
            grids.append(
                {
                    (level, bucket): ws.cell(first + i, 2 + off + bucket).value or 0
                    for i, level in enumerate(difficulties)
                    for bucket in range(BUCKETS)
                }
            )
        arms.append(
            {
                "label": ws.cell(arm_row, 1 + off).value,
                "path": ws.cell(arm_row, 3 + off).value,
                "attempted": grids[0],
                "correct": grids[1],
            }
        )
    return {"subtitle": ws.cell(15 + 2 * n + 1, 1).value, "arms": arms}


def link_grid(lay: Layout, title_row: int, difficulties: tuple[int, ...]) -> dict:
    """Cell references into the ablation's alpha = 0.5 arm, so it stays the one source."""
    first = lay.block_data(title_row)
    return {
        (level, bucket): f"={get_column_letter(2 + ALPHA_ARM + bucket)}{first + i}"
        for i, level in enumerate(difficulties)
        for bucket in range(BUCKETS)
    }


def blank_or_sum(first_col: str, last_col: str, row_a: int, row_b: int) -> str:
    """SUM that stays visually blank until someone enters the counts it adds up."""
    rng = f"{first_col}{row_a}:{last_col}{row_b}"
    return f'=IF(COUNT({rng})=0,"",SUM({rng}))'


def ratio(numerator: str, denominator: str, fallback: str) -> str:
    return f"=IFERROR({numerator}/{denominator}*100,{fallback})"


def write_grid(
    ws, lay: Layout, blocks, difficulties, attempted=None, correct=None, off: int = 0
) -> None:
    """Write the four stacked tables (attempted, correct, accuracy, distribution).

    `off` shifts the whole 8-column table right, which is how several methods sit
    side by side on one set of rows.

    With `attempted`/`correct` given, the count cells are filled and the derived
    cells fall back to 0 -- a zero denominator there is a real observation (nobody
    took that many hints). Without them the block is an empty but live template:
    the same equations, falling back to blank, because nothing has been measured.
    Count values may be plain numbers or formulas referring to another section.
    """
    att_t, cor_t, pct_t, dist_t = blocks
    n = lay.n
    filled = attempted is not None
    fallback = "0" if filled else '""'

    def col(index: int) -> str:
        return get_column_letter(index + off)

    for title_row, title in zip(blocks, BLOCK_TITLES):
        ws.cell(title_row, 1 + off).value = title
        for index, head in enumerate(HEADER, start=1):
            ws.cell(title_row + 1, index + off).value = head

    for title_row, grid in ((att_t, attempted), (cor_t, correct)):
        first = lay.block_data(title_row)
        for i, level in enumerate(difficulties):
            row = first + i
            ws.cell(row, 1 + off).value = level
            if filled:
                for bucket in range(BUCKETS):
                    ws.cell(row, 2 + off + bucket).value = grid[(level, bucket)]
                ws.cell(row, 8 + off).value = f"=SUM({col(2)}{row}:{col(7)}{row})"
            else:
                ws.cell(row, 8 + off).value = blank_or_sum(col(2), col(7), row, row)
        total = lay.block_total(title_row)
        ws.cell(total, 1 + off).value = "Total"
        for index in range(2, 9):
            letter = col(index)
            if filled:
                ws.cell(total, index + off).value = (
                    f"=SUM({letter}{first}:{letter}{first + n - 1})"
                )
            else:
                ws.cell(total, index + off).value = blank_or_sum(
                    letter, letter, first, first + n - 1
                )

    # Percentage correct = correct / attempted, cell for cell.
    for i in range(n + 1):
        row = lay.block_data(pct_t) + i
        cor_row = lay.block_data(cor_t) + i
        att_row = lay.block_data(att_t) + i
        ws.cell(row, 1 + off).value = "Total" if i == n else difficulties[i]
        for index in range(2, 9):
            letter = col(index)
            ws.cell(row, index + off).value = ratio(
                f"{letter}{cor_row}", f"{letter}{att_row}", fallback
            )

    # Hint distribution = share of this difficulty's CORRECT answers, per the template.
    for i in range(n + 1):
        row = lay.block_data(dist_t) + i
        cor_row = lay.block_data(cor_t) + i
        ws.cell(row, 1 + off).value = "Total" if i == n else difficulties[i]
        for index in range(2, 8):
            letter = col(index)
            ws.cell(row, index + off).value = ratio(
                f"{letter}{cor_row}", f"{col(8)}{cor_row}", fallback
            )
        if i < n:
            ws.cell(row, 8 + off).value = "-"
        elif filled:
            ws.cell(row, 8 + off).value = f"=SUM({col(2)}{row}:{col(7)}{row})"
        else:
            ws.cell(row, 8 + off).value = blank_or_sum(col(2), col(7), row, row)


def write_generator(ws, lay: Layout, difficulties, counts, correct, source: str) -> None:
    """Generator for Methods A and C: counts are data, Acc% stays an equation."""
    ws.cell(lay.gen_title, 1).value = "Generator for Methods A and C"
    ws.cell(lay.gen_subtitle, 1).value = (
        "Zero-hint solver attempt on every problem, pooled from Method C's chain "
        f"traces and re-graded: {source}"
    )
    for col, head in enumerate(["Difficulty", "Count", "Correct", "Acc%"], start=1):
        ws.cell(lay.gen_hdr, col).value = head
    for i, level in enumerate(difficulties):
        row = lay.gen_data + i
        ws.cell(row, 1).value = level
        ws.cell(row, 2).value = counts[level]
        ws.cell(row, 3).value = correct[level]
        ws.cell(row, 4).value = ratio(f"C{row}", f"B{row}", '""')
    total = lay.gen_total
    ws.cell(total, 1).value = "Total"
    last = lay.gen_data + lay.n - 1
    ws.cell(total, 2).value = blank_or_sum("B", "B", lay.gen_data, last)
    ws.cell(total, 3).value = blank_or_sum("C", "C", lay.gen_data, last)
    ws.cell(total, 4).value = ratio(f"C{total}", f"B{total}", '""')


def section_style_plan(lay: Layout, title: int, arm: int, blocks) -> list[tuple[int, int]]:
    """(destination row, template row) for one multi-column-group section."""
    tpl = TemplateRows
    plan = [
        (title, tpl.group_title),
        (title + 1, tpl.group_subtitle),
        (title + 2, tpl.group_subtitle + 1),
        (arm, tpl.group_arm),
        (arm + 1, tpl.group_arm + 1),
    ]
    for dst, src in zip(blocks, tpl.group_blocks):
        plan.append((dst, src))
        plan.append((dst + 1, src + 1))
        for i, src_row in enumerate(data_style_rows(src + 2, src + 3, src + 4, lay.n)):
            plan.append((dst + 2 + i, src_row))
        plan.append((lay.block_total(dst), src + 5))
        plan.append((lay.block_total(dst) + 1, src + 6))  # blank separator
    return plan


def style_plan(lay: Layout) -> list[tuple[int, int]]:
    """(destination row, cd-1.5b source row) for every row this script rewrites."""
    tpl = TemplateRows
    plan: list[tuple[int, int]] = [
        (lay.gen_title, tpl.gen_title),
        (lay.gen_subtitle, tpl.gen_subtitle),
        (lay.gen_subtitle + 1, tpl.gen_subtitle + 1),
        (lay.gen_hdr, tpl.gen_hdr),
    ]
    for i, src in enumerate(
        data_style_rows(tpl.gen_data, tpl.gen_data + 1, tpl.gen_data + 2, lay.n)
    ):
        plan.append((lay.gen_data + i, src))
    plan.append((lay.gen_total, tpl.gen_total))
    plan += section_style_plan(lay, lay.sbs_title, lay.sbs_arm, lay.sbs_blocks)
    plan += section_style_plan(lay, lay.ab_title, lay.ab_arm, lay.ab_blocks)
    return plan


def snapshot_styles(ws) -> dict[int, list]:
    """Copy the template tab's styling before that same tab gets rewritten."""
    return {
        row: [copy(ws.cell(row, col)._style) for col in range(1, BANNER_COLS + 1)]
        for row in range(1, TemplateRows.span + 1)
    }


def apply_styles(ws, styles: dict[int, list], lay: Layout) -> None:
    for dst, src in style_plan(lay):
        for col in range(1, BANNER_COLS + 1):
            ws.cell(dst, col)._style = copy(styles[src][col - 1])


def clear(ws, first_row: int, last_row: int, blank_style) -> None:
    for row in range(first_row, last_row + 1):
        for col in range(1, BANNER_COLS + 1):
            cell = ws.cell(row, col)
            cell.value = None
            cell._style = copy(blank_style)


def apply_conditional_formatting(ws, lay: Layout) -> None:
    """A white-to-green scale on every percentage / distribution block, per cd-1.5b."""
    ranges = []
    for blocks, offsets in ((lay.sbs_blocks, SBS_OFFSETS), (lay.ab_blocks, ARM_OFFSETS)):
        for title_row in (blocks[2], blocks[3]):
            for offset in offsets:
                start = get_column_letter(2 + offset)
                end = get_column_letter(7 + offset)
                ranges.append(
                    f"{start}{lay.block_data(title_row)}:{end}{lay.block_total(title_row)}"
                )

    ws.conditional_formatting = ConditionalFormattingList()
    ws.conditional_formatting.add(
        " ".join(ranges),
        ColorScaleRule(
            start_type="formula",
            start_value="0",
            start_color="FFFFFFFF",
            end_type="formula",
            end_value="100",
            end_color="FF57BB8A",
        ),
    )


def main() -> None:
    workbook = openpyxl.load_workbook(SOURCE)
    styles = snapshot_styles(workbook[TEMPLATE_TAB])
    blank_style = styles[TemplateRows.blank][0]

    for tab, task, eval_name, difficulties in TABS:
        ws = workbook[tab]
        lay = Layout(len(difficulties))
        a_path = METHOD_A_DIR / eval_name
        c_path = METHOD_C_DIR / eval_name
        g_path = GENERATOR_DIR / eval_name

        a_att, a_cor, a_rows, a_drop = tally(a_path, task, difficulties)
        c_att, c_cor, c_rows, c_drop = tally(c_path, task, difficulties)
        g_cnt, g_cor, g_rows, g_drop = tally_generator(g_path, task, difficulties)
        ablation = read_ablation(ws, lay.n, difficulties)

        clear(ws, lay.gen_title, max(ws.max_row, lay.end) + 40, blank_style)
        apply_styles(ws, styles, lay)
        write_generator(
            ws, lay, difficulties, g_cnt, g_cor, str(g_path.relative_to(REPO))
        )

        ws.cell(lay.sbs_title, 1).value = SBS_TITLE
        ws.cell(lay.sbs_subtitle, 1).value = SBS_SUBTITLE
        alpha_arm = ablation["arms"][ARM_OFFSETS.index(ALPHA_ARM)]
        groups = [
            ("Method A", str(a_path.relative_to(REPO)), a_att, a_cor),
            (
                f"Method B ({alpha_arm['label']})",
                alpha_arm["path"],
                link_grid(lay, lay.ab_blocks[0], difficulties),
                link_grid(lay, lay.ab_blocks[1], difficulties),
            ),
            ("Method C", str(c_path.relative_to(REPO)), c_att, c_cor),
        ]
        for off, (label, path, attempted, correct) in zip(SBS_OFFSETS, groups):
            ws.cell(lay.sbs_arm, 1 + off).value = label
            ws.cell(lay.sbs_arm, 2 + off).value = "File path:"
            ws.cell(lay.sbs_arm, 3 + off).value = path
            write_grid(ws, lay, lay.sbs_blocks, difficulties, attempted, correct, off)

        ws.cell(lay.ab_title, 1).value = ABLATION_TITLE
        ws.cell(lay.ab_subtitle, 1).value = ablation["subtitle"]
        for off, arm in zip(ARM_OFFSETS, ablation["arms"]):
            ws.cell(lay.ab_arm, 1 + off).value = arm["label"]
            ws.cell(lay.ab_arm, 2 + off).value = "File path:"
            ws.cell(lay.ab_arm, 3 + off).value = arm["path"]
            write_grid(
                ws, lay, lay.ab_blocks, difficulties, arm["attempted"], arm["correct"], off
            )

        apply_conditional_formatting(ws, lay)

        # Clearing materialises the emptied cells, which would leave the sheet
        # claiming a used range hundreds of rows past the last section.
        if ws.max_row > lay.end:
            ws.delete_rows(lay.end + 1, ws.max_row - lay.end)

        for name, attempted, correct, rows, dropped in (
            ("gen", g_cnt, g_cor, g_rows, g_drop),
            ("A", a_att, a_cor, a_rows, a_drop),
            ("C", c_att, c_cor, c_rows, c_drop),
        ):
            total = sum(attempted.values())
            n_correct = sum(correct.values())
            print(
                f"{tab:10s} method {name}: n={total} correct={n_correct} "
                f"acc={100 * n_correct / total:.2f}%  (rows={rows}, dropped={dropped})"
            )
        for arm in ablation["arms"]:
            total = sum(arm["attempted"].values())
            n_correct = sum(arm["correct"].values())
            print(
                f"{tab:10s} method B {arm['label']}: n={total} correct={n_correct} "
                f"acc={100 * n_correct / total:.2f}%"
            )

    workbook.calculation.fullCalcOnLoad = True
    workbook.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
