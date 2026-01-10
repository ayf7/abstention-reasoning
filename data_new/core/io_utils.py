"""
I/O utilities for JSONL and Parquet files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_parquet(
    path: Path,
    records: List[Dict[str, Any]],
    json_serialize_keys: List[str] | None = None,
) -> None:
    """
    Write records to a Parquet file using HuggingFace datasets.

    This preserves nested structures (lists of dicts) which is required
    for verl compatibility.

    Args:
        path: Output path
        records: List of record dicts
        json_serialize_keys: Keys whose values should be JSON-serialized
                           (for nested dicts that don't fit the schema)
                           Note: Lists of dicts (like prompt messages) are
                           preserved natively by HuggingFace datasets.
    """
    from datasets import Dataset

    if not records:
        raise ValueError("Cannot write empty records to parquet")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare records - JSON serialize specified keys only
    if json_serialize_keys:
        prepared = []
        for rec in records:
            new_rec = rec.copy()
            for key in json_serialize_keys:
                if key in new_rec and not isinstance(new_rec[key], str):
                    new_rec[key] = json.dumps(new_rec[key])
            prepared.append(new_rec)
        records = prepared

    # Use HuggingFace datasets to write parquet - preserves nested structures
    dataset = Dataset.from_list(records)
    dataset.to_parquet(str(path))


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file as DataFrame."""
    return pd.read_parquet(path)


def build_index(records: List[Dict[str, Any]], key: str = "index") -> Dict[Any, Dict]:
    """Build an index dict for O(1) lookups by key."""
    return {rec[key]: rec for rec in records}
