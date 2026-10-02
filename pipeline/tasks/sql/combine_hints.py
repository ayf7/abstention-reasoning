"""Combine individual generated hint files into unified train and dev files.

Usage:
    python -m pipeline.tasks.sql.combine_hints --style conceptual
    python -m pipeline.tasks.sql.combine_hints --style partial_sql
    python -m pipeline.tasks.sql.combine_hints --style all
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/home/tanyagoyal/abstention-reasoning/data/sql")

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


def combine_files(file_prefixes: list[str], style: str, output_path: Path):
    combined = []
    total_loaded = 0
    total_with_hints = 0

    for prefix in file_prefixes:
        file_path = DATA_DIR / f"{prefix}_hints_{style}.json"
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}, skipping.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_items = [x for x in data if x is not None]
        with_hints = [x for x in valid_items if x.get("prefix_hints")]
        logger.info(f"Loaded {prefix}: {len(valid_items)} total, {len(with_hints)} with hints")
        combined.extend(valid_items)
        total_loaded += len(valid_items)
        total_with_hints += len(with_hints)

    # Re-index combined items cleanly while keeping original indices in metadata
    for new_idx, item in enumerate(combined):
        item["metadata"]["original_split_index"] = item["index"]
        item["index"] = new_idx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(combined)} records ({total_with_hints} with hints) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine generated hint files into unified train/dev files.")
    parser.add_argument(
        "--style",
        type=str,
        default="all",
        choices=["conceptual", "partial_sql", "all"],
        help="Hint style to combine",
    )
    args = parser.parse_args()

    styles = ["conceptual", "partial_sql"] if args.style == "all" else [args.style]

    for style in styles:
        logger.info(f"=== Combining style: {style} ===")
        dev_out = DATA_DIR / f"primitives_combined_dev_hints_{style}.json"
        train_out = DATA_DIR / f"primitives_combined_train_hints_{style}.json"

        combine_files(DEV_PREFIXES, style, dev_out)
        combine_files(TRAIN_PREFIXES, style, train_out)


if __name__ == "__main__":
    main()
