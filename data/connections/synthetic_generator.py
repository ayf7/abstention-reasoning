"""
Generate synthetic Connections-style groups using the OpenAI Chat API.

Each generated record has the shape:
{
    "index": <int>,
    "description": <str>,
    "words": [w1, w2, w3, w4],
    "contest": <str>,
}

Notes:
- Requires an `OPENAI_API_KEY` environment variable.
- Install the client library if needed: `pip install openai`.
"""

from __future__ import annotations

import argparse
import json
import os
import ast
from pathlib import Path
from typing import Iterable, List, Any, Optional, Set

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency: install with `pip install openai`"
    ) from exc


DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "synthetic_groups.jsonl"
DEFAULT_GROUPS = BASE_DIR / "groups.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Connections-style groups via OpenAI."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of groups to generate (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "Starting index to assign to generated groups. If omitted, "
            "auto-detects from synthetic_groups.jsonl (or groups.jsonl)."
        ),
    )
    parser.add_argument(
        "--contest",
        type=str,
        default="Synthetic",
        help='Contest field value for generated rows (default: "Synthetic")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _build_prompt(count: int) -> List[dict]:
    user_prompt = f"""
Generate {count} distinct New York Times Connections-style groups.

Rules:
- Each group must have a short description (e.g., "PALINDROMES", "NBA TEAMS"). Do not make the group too basic; should be somewhat specific.
- Each group must have exactly 4 related words.
- Each word should be UPPERCASE ASCII, single tokens (no phrases), no duplicates within a group.
- No copyrighted or offensive content.
- Do not use previous NYT connections puzzle themes.
"""
    return [
        {
            "role": "system",
            "content": (
                "You are a data generator for NYT Connections-style puzzles. "
                "Return only strict JSON, no commentary."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _parse_response(data: Any) -> List[dict]:
    """
    Accept either:
    - a dict with a top-level 'groups' list
    - a list of group dicts
    """
    # If the model returned a JSON string, decode it
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(data)
            except Exception as exc:  # pragma: no cover
                raise ValueError(f"Could not parse model response: {exc}") from exc

    # Allow dict with 'groups'
    if isinstance(data, dict):
        if "groups" not in data:
            raise ValueError("Model response missing 'groups' field.")
        groups = data.get("groups") or []
    elif isinstance(data, list):
        groups = data
    else:
        raise ValueError("Model response must be dict or list.")

    if not isinstance(groups, list):
        raise ValueError("'groups' must be a list.")

    cleaned = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        words = item.get("words") or []
        if not desc or not isinstance(words, list):
            continue
        words_clean = [str(w).strip().upper() for w in words if str(w).strip()]
        if len(words_clean) != 4:
            continue
        cleaned.append({"description": desc, "words": words_clean})
    if not cleaned:
        raise ValueError("Model response contained no usable groups.")
    return cleaned


def _max_index_from_file(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    max_idx: Optional[int] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("index")
            if isinstance(idx, int):
                max_idx = idx if max_idx is None else max(max_idx, idx)
    return max_idx


def _descriptions_from_file(path: Path) -> Set[str]:
    """
    Collect normalized (uppercased) descriptions from a JSONL file.
    """
    descs: Set[str] = set()
    if not path.is_file():
        return descs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            desc = rec.get("description")
            if isinstance(desc, str):
                descs.add(desc.strip().upper())
    return descs


def _write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.is_file() and path.stat().st_size > 0:
        try:
            with path.open("rb") as f:
                f.seek(-1, 2)
                last_char = f.read(1)
                needs_newline = last_char != b"\n"
        except OSError:
            needs_newline = False

    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)

    messages = _build_prompt(args.count)

    print(messages)

    schema = {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "words": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["description", "words"],
                    "additionalProperties": False,
                },
                "minItems": args.count,
                "maxItems": args.count,
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=args.model,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "connections_groups",
                "schema": schema,
                "strict": True,
            }
        },
    )

    # Try parsed output first; fall back to raw text if needed.
    raw_data = getattr(response, "output_parsed", None)
    if raw_data is None:
        raw_data = getattr(response, "output_text", None)
    if raw_data is None:
        raise ValueError("No response content returned by model.")
    groups = _parse_response(raw_data)

    # Determine starting index (auto-detect when not provided).
    start_index: int
    if args.start_index is not None:
        start_index = args.start_index
    else:
        candidates = [_max_index_from_file(args.output), _max_index_from_file(DEFAULT_GROUPS)]
        max_found = max((c for c in candidates if c is not None), default=-1)
        start_index = max_found + 1

    # Drop groups whose description already exists (case-insensitive) in
    # the output file or the canonical groups file.
    existing_descs = _descriptions_from_file(args.output) | _descriptions_from_file(DEFAULT_GROUPS)

    output_records = []
    skipped = 0
    for g in groups:
        desc_norm = g["description"].strip().upper()
        if desc_norm in existing_descs:
            skipped += 1
            continue
        existing_descs.add(desc_norm)
        output_records.append(
            {
                "index": start_index + len(output_records),
                "description": g["description"],
                "words": g["words"],
                "contest": args.contest,
            }
        )

    _write_jsonl(output_records, args.output)
    print(
        f"Wrote {len(output_records)} groups to {args.output} "
        f"(skipped {skipped} duplicate descriptions; "
        f"model={args.model}, temp={args.temperature})"
    )


if __name__ == "__main__":
    main()
