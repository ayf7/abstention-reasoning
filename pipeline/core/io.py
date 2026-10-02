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


def save_parquet(path: Path | str, records: list[dict], max_shard_bytes: int = 45 * 1024 * 1024) -> list[Path]:
    """Save list of dicts as parquet file(s), sharded to stay under GitHub's
    100MB hard limit (and 50MB "soft warning" threshold) on file size.

    When the full dataset would serialize to more than `max_shard_bytes`,
    writes multiple numbered shards instead of one big file: `<stem>.shard000
    <suffix>`, `<stem>.shard001<suffix>`, etc, next to the originally
    requested `path`, and removes `path` itself (plus any stale shards from a
    previous run with a different shard count) so there's no ambiguity about
    which file(s) are current. verl's RLDataset natively accepts a list of
    parquet files, so training code just needs to glob for these shards
    instead of assuming a single path.

    Returns the list of file(s) actually written (length 1 if unsharded).
    """
    from datasets import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use HuggingFace datasets to preserve nested structures
    ds = Dataset.from_list(records)

    # Clear out any stale output from a previous run before (re)writing,
    # since the new run may produce a different number of shards (or none).
    path.unlink(missing_ok=True)
    for stale in path.parent.glob(f"{path.stem}.shard*{path.suffix}"):
        stale.unlink()

    # Estimate on-disk parquet size via a single-shot write, then split only
    # if that exceeds the threshold -- avoids guessing row count up front.
    ds.to_parquet(str(path))
    if path.stat().st_size <= max_shard_bytes or len(records) <= 1:
        return [path]

    total_bytes = path.stat().st_size
    num_shards = -(-total_bytes // max_shard_bytes)  # ceil division
    rows_per_shard = -(-len(records) // num_shards)
    path.unlink()

    written = []
    for shard_idx, start in enumerate(range(0, len(records), rows_per_shard)):
        shard_records = records[start:start + rows_per_shard]
        shard_path = path.parent / f"{path.stem}.shard{shard_idx:03d}{path.suffix}"
        Dataset.from_list(shard_records).to_parquet(str(shard_path))
        written.append(shard_path)
    return written


def load_parquet_shards(path: Path | str) -> list[Path]:
    """Resolve a parquet path written by `save_parquet` back to its file(s).

    If `path` exists as a single file, returns `[path]` unchanged. Otherwise
    looks for `<stem>.shard*<suffix>` next to it (see `save_parquet`) and
    returns them in order. Raises FileNotFoundError if neither is found.
    """
    path = Path(path)
    if path.exists():
        return [path]
    shards = sorted(path.parent.glob(f"{path.stem}.shard*{path.suffix}"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet file or shards found for {path} "
            f"(looked for {path.stem}.shard*{path.suffix} in {path.parent})"
        )
    return shards
