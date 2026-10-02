"""Combine per-source hint files (dev + train, single style) into one filtered,
globally-reindexed primitives file.

Unlike combine_hints.py (which writes separate dev/train output files), this
script merges ALL dev+train sources into a SINGLE output file with globally
unique `index` values, then applies the filters historically used to build
`data/sql_conceptual/problems/primitives.json` / `data/sql_partial_sql/problems/primitives.json`:

1. is_complex / multi_step: already enforced upstream by `generate_hints.py`
   (non-matching records are `None` placeholders in the `_hints_{style}.json`
   files and are dropped here when loading). Kept as an explicit assertion so
   this script fails loudly if that invariant ever changes.
2. Largest-9-BIRD-databases exclusion: records whose source is "bird" and
   whose db_id is one of the 9 largest BIRD sqlite databases (by file size,
   used as a proxy for database/schema size to control prompt length) are
   dropped.

Usage:
    python -m pipeline.tasks.sql.combine_and_filter_hints --style conceptual --output data/sql_conceptual/problems/primitives.json
    python -m pipeline.tasks.sql.combine_and_filter_hints --style partial_sql --output data/sql_partial_sql/problems/primitives.json
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/home/tanyagoyal/abstention-reasoning/data/sql")

# Dev is combined first, then train, so the final file preserves a dev-then-train
# ordering (matching the historical primitives.json layout).
DEV_PREFIXES = [
    "primitives_spider_dev",
    "primitives_bird_dev",
    "primitives_sparc_dev",
    "primitives_cosql_dev",
]

TRAIN_PREFIXES = [
    "primitives_spider_train",
    "primitives_spider_train_others",
    "primitives_bird_train",
    "primitives_sparc_train",
    "primitives_cosql_train",
]

# The 9 largest BIRD databases by raw sqlite file size (used as a proxy for
# schema/data size to control prompt length). Computed via:
#   find data/sql/raw/bird/databases -iname "*.sqlite" -exec du -b {} \; | sort -rn | head -9
LARGEST_BIRD_DBS = {
    "bike_share_1",
    "donor",
    "codebase_comments",
    "movie_platform",
    "world_development_indicators",
    "language_corpus",
    "talkingdata",
    "music_platform_2",
    "coinmarketcap",
}


def load_source_files(file_prefixes: list[str], style: str) -> list[dict]:
    """Load and concatenate valid (non-None) records from the given per-source files."""
    combined = []
    for prefix in file_prefixes:
        file_path = DATA_DIR / f"{prefix}_hints_{style}.json"
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_items = [x for x in data if x is not None]
        logger.info(f"Loaded {prefix}: {len(valid_items)} valid records")
        combined.extend(valid_items)
    return combined


def apply_filters(records: list[dict]) -> list[dict]:
    filtered = []
    dropped_largest_bird = 0
    for item in records:
        md = item["metadata"]
        # Sanity check: generate_hints.py's FILTER="complex" should have already
        # restricted records to is_complex AND multi_step. Fail loudly if not.
        assert md.get("is_complex") and md.get("multi_step"), (
            f"Unexpected non-complex/non-multi_step record reached combine_and_filter_hints "
            f"(index={item.get('index')}, source={md.get('source')}); "
            f"upstream generate_hints filter invariant may have changed."
        )
        if md.get("source") == "bird" and md.get("db_id") in LARGEST_BIRD_DBS:
            dropped_largest_bird += 1
            continue
        filtered.append(item)

    logger.info(f"Dropped {dropped_largest_bird} records from the 9 largest BIRD databases")
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-source hint files (dev+train) into one filtered, globally-reindexed primitives file."
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        choices=["conceptual", "partial_sql"],
        help="Hint style to combine",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the combined, filtered primitives file.",
    )
    args = parser.parse_args()

    dev_records = load_source_files(DEV_PREFIXES, args.style)
    train_records = load_source_files(TRAIN_PREFIXES, args.style)
    combined = dev_records + train_records
    logger.info(f"Combined dev+train total: {len(combined)}")

    filtered = apply_filters(combined)
    logger.info(f"After filtering: {len(filtered)}")

    # Globally reindex, preserving the original per-source index in metadata.
    for new_idx, item in enumerate(filtered):
        item["metadata"]["original_split_index"] = item["index"]
        item["index"] = new_idx

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(filtered)} records to {args.output}")


if __name__ == "__main__":
    main()
