"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Save JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_parquet(path: Path | str, records: list[dict]) -> None:
    """Save list of dicts as parquet file."""
    from datasets import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use HuggingFace datasets to preserve nested structures
    ds = Dataset.from_list(records)
    ds.to_parquet(str(path))
