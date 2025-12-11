"""
Build a group-level JSONL dataset from the raw CSV export.

Each output record is a single group in the format:
{
    "index": <int>,
    "description": <group name>,
    "words": [w1, w2, w3, w4],
    "contest": "NYT Connections <game_id> - <date>"
}
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "connections_data.csv"
DEFAULT_OUTPUT = BASE_DIR / "groups.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert connections_data.csv into a puzzle-level JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _parse_date_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _normalize_text(text: str) -> str:
    """
    Normalize curly quotes/dashes and strip extraneous whitespace.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.strip()


def _load_rows(path: Path) -> Dict[int, List[dict]]:
    puzzles: Dict[int, List[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = int(row["Game ID"])
            puzzles[game_id].append(row)
    return puzzles


def main() -> None:
    args = parse_args()
    puzzles = _load_rows(args.input)

    output_records: List[dict] = []
    idx = 0

    sorted_puzzles: List[Tuple[int, int, List[dict]]] = []
    for game_id, rows in puzzles.items():
        if not rows:
            continue
        puzzle_date = rows[0]["Puzzle Date"]
        date_ms = _parse_date_ms(puzzle_date)
        sorted_puzzles.append((date_ms, game_id, rows))

    # Sort chronologically, then by game_id for determinism.
    sorted_puzzles.sort(key=lambda t: (t[0], t[1]))

    for _, game_id, rows in sorted_puzzles:
        puzzle_date = rows[0]["Puzzle Date"]
        contest = f"NYT Connections {game_id} - {puzzle_date}"

        groups: Dict[str, List[str]] = defaultdict(list)

        for r in rows:
            name = _normalize_text(r["Group Name"])
            word = _normalize_text(r["Word"])
            groups[name].append(
                (
                    int(r["Starting Row"]),
                    int(r["Starting Column"]),
                    word,
                )
            )

        # Sort groups alphabetically for deterministic order; words inside a group
        # follow board order (row, column).
        for name in sorted(groups):
            word_entries = sorted(groups[name], key=lambda t: (t[0], t[1]))
            words = [w for _, _, w in word_entries]
            record = {
                "index": idx,
                "description": name,
                "words": words,
                "contest": contest,
            }
            output_records.append(record)
            idx += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(output_records)} group records to {args.output} "
        f"(from {len(puzzles)} puzzles)"
    )


if __name__ == "__main__":
    main()
