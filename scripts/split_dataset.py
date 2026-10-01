#!/usr/bin/env python3
"""Interactive, distribution-preserving dataset splitter.

Given an input JSON file containing a list of problem dicts (e.g.
data/math_o1/problems/primitives_all.json), this script:

  1. Asks the user which fields to stratify on (maintain distribution for),
     e.g. "variant", "level".
  2. Asks the user for the desired sizes of the `eval` and `sft` splits
     (must sum to <= total number of datapoints); the remaining datapoints
     are assigned to `rl`.
  3. Produces `eval`, `sft`, and `rl` splits whose joint distribution over
     the chosen fields roughly matches the full dataset's distribution.
  4. Further splits `sft` and `rl` into 90:10 train/val subsplits
     (again preserving the distribution over the chosen fields).
  5. Optionally (if the user opts in) further splits `rl_train` into
     `rl_gen_train` / `rl_ver_train`, and `rl_val` into
     `rl_gen_val` / `rl_ver_val` -- `rl_train` and `rl_val` are still
     written out in addition to the gen/ver subsplits.

Output files are written as `<split_name>.json` directly alongside the input
file (default: the same directory as the input file), matching the layout
`pipeline`'s `create_prompts`/`create_partitions` expect (partition files
live next to `primitives.json`, not in a nested subdirectory).

Usage:
    python scripts/split_dataset.py path/to/primitives.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_data(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return data


def prompt_fields(sample_record: dict) -> list:
    print("\nAvailable fields in each record:")
    top_level_fields = sorted(sample_record.keys())
    for i, field in enumerate(top_level_fields):
        print(f"  [{i}] {field}")

    print(
        "\nEnter the fields you want to maintain the distribution for "
        "(comma-separated names or indices)."
    )
    raw = input("Fields: ").strip()
    if not raw:
        raise ValueError("You must specify at least one field.")

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    chosen = []
    for tok in tokens:
        if tok.isdigit() and int(tok) < len(top_level_fields):
            chosen.append(top_level_fields[int(tok)])
        elif tok in sample_record:
            chosen.append(tok)
        else:
            raise ValueError(f"Field '{tok}' not found in record.")
    return chosen


def prompt_split_sizes(total: int) -> dict:
    print(f"\nTotal number of datapoints: {total}")
    sizes = {}
    for split in ["eval", "sft"]:
        while True:
            raw = input(f"  Size of '{split}' split: ").strip()
            try:
                n = int(raw)
                if n < 0:
                    raise ValueError
                sizes[split] = n
                break
            except ValueError:
                print("    Please enter a non-negative integer.")

    requested_total = sum(sizes.values())
    if requested_total > total:
        raise ValueError(
            f"Requested split sizes sum to {requested_total}, "
            f"which exceeds the total of {total} datapoints."
        )
    sizes["rl"] = total - requested_total
    print(f"  Size of 'rl' split (remaining datapoints): {sizes['rl']}")
    return sizes


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def record_key(record: dict, fields: list) -> tuple:
    return tuple(record.get(f) for f in fields)


def stratified_group(data: list, fields: list) -> dict:
    groups = defaultdict(list)
    for record in data:
        groups[record_key(record, fields)].append(record)
    return groups


def allocate_counts(group_sizes: dict, target_total: int) -> dict:
    """Proportionally allocate `target_total` items across groups,
    preserving each group's share of the overall distribution, using
    largest-remainder rounding so the counts sum exactly to target_total
    (or less, if a group runs out of items — callers must ensure enough
    items are available overall)."""
    grand_total = sum(group_sizes.values())
    if grand_total == 0:
        return {}

    raw_alloc = {
        key: (size / grand_total) * target_total
        for key, size in group_sizes.items()
    }
    floor_alloc = {key: int(v) for key, v in raw_alloc.items()}
    allocated = sum(floor_alloc.values())
    remainder = target_total - allocated

    # Distribute leftover units to groups with the largest fractional
    # remainder first.
    remainders = sorted(
        raw_alloc.keys(), key=lambda k: raw_alloc[k] - floor_alloc[k], reverse=True
    )
    for key in remainders[:remainder]:
        floor_alloc[key] += 1

    return floor_alloc


def split_by_distribution(groups: dict, split_sizes: dict, rng: random.Random):
    """Given grouped records (key -> list of records) and target split
    sizes (split_name -> count), returns split_name -> list of records,
    where each split's distribution across group keys approximates the
    overall distribution, drawing without replacement across splits."""

    # Shuffle each group's records so sampling is random.
    group_pools = {key: list(records) for key, records in groups.items()}
    for pool in group_pools.values():
        rng.shuffle(pool)

    group_sizes = {key: len(records) for key, records in groups.items()}

    result = {split: [] for split in split_sizes}
    remaining_sizes = dict(group_sizes)

    for split, target in split_sizes.items():
        if target == 0:
            continue
        alloc = allocate_counts(remaining_sizes, target)
        # Fix shortfall: if a group doesn't have enough remaining items
        # (can happen due to rounding against already-shrinking pools),
        # redistribute the shortfall to the groups with spare capacity.
        shortfall = 0
        for key, n in list(alloc.items()):
            available = len(group_pools[key])
            if n > available:
                shortfall += n - available
                alloc[key] = available

        if shortfall > 0:
            # Give leftover slots to groups that still have spare items,
            # largest pools first.
            candidates = sorted(
                group_pools.keys(),
                key=lambda k: len(group_pools[k]) - alloc.get(k, 0),
                reverse=True,
            )
            for key in candidates:
                if shortfall <= 0:
                    break
                spare = len(group_pools[key]) - alloc.get(key, 0)
                if spare <= 0:
                    continue
                take = min(spare, shortfall)
                alloc[key] = alloc.get(key, 0) + take
                shortfall -= take

        for key, n in alloc.items():
            if n <= 0:
                continue
            chosen = group_pools[key][:n]
            group_pools[key] = group_pools[key][n:]
            result[split].extend(chosen)
            remaining_sizes[key] = len(group_pools[key])

    return result


def split_two_way(records: list, fields: list, frac_a: float, rng: random.Random):
    """Split `records` into two lists (a, b), with `frac_a` fraction of
    each stratification group going to `a`, preserving distribution."""
    groups = stratified_group(records, fields)
    target_a = {
        key: round(len(recs) * frac_a) for key, recs in groups.items()
    }
    a, b = [], []
    for key, recs in groups.items():
        pool = list(recs)
        rng.shuffle(pool)
        n_a = min(target_a[key], len(pool))
        a.extend(pool[:n_a])
        b.extend(pool[n_a:])
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def print_distribution(name: str, records: list, fields: list):
    groups = stratified_group(records, fields)
    print(f"\n[{name}] n={len(records)}")
    for key in sorted(groups.keys(), key=lambda k: str(k)):
        count = len(groups[key])
        pct = 100 * count / len(records) if records else 0
        key_str = ", ".join(f"{f}={v}" for f, v in zip(fields, key))
        print(f"    {key_str}: {count} ({pct:.1f}%)")


def write_split(records: list, out_dir: Path, name: str):
    rng = random.Random()
    rng.shuffle(records)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Wrote {len(records)} records -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Path to input JSON file (list of records).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write split files to (default: same directory as input file)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if not args.input_file.exists():
        print(f"Error: input file {args.input_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    data = load_data(args.input_file)
    total = len(data)
    if total == 0:
        print("Error: input file contains no records.", file=sys.stderr)
        sys.exit(1)

    fields = prompt_fields(data[0])
    print(f"\nMaintaining distribution over: {fields}")
    print_distribution("full dataset", data, fields)

    split_sizes = prompt_split_sizes(total)

    groups = stratified_group(data, fields)
    top_splits = split_by_distribution(groups, split_sizes, rng)

    eval_split = top_splits.get("eval", [])
    sft_split = top_splits.get("sft", [])
    rl_split = top_splits.get("rl", [])

    for name, recs in [("eval", eval_split), ("sft", sft_split), ("rl", rl_split)]:
        print_distribution(name, recs, fields)

    # 90:10 train/val split for sft and rl.
    sft_train, sft_val = split_two_way(sft_split, fields, 0.9, rng)
    rl_train, rl_val = split_two_way(rl_split, fields, 0.9, rng)

    print_distribution("sft_train", sft_train, fields)
    print_distribution("sft_val", sft_val, fields)
    print_distribution("rl_train", rl_train, fields)
    print_distribution("rl_val", rl_val, fields)

    outputs = {
        "eval": eval_split,
        "sft_train": sft_train,
        "sft_val": sft_val,
    }

    split_rl_for_verification = False
    if rl_split:
        split_rl_for_verification = prompt_yes_no(
            "\nDo you want to split 'rl' for verification "
            "(rl_train -> rl_gen_train/rl_ver_train, rl_val -> rl_gen_val/rl_ver_val)?",
            default=False,
        )

    if split_rl_for_verification:
        while True:
            raw = input(
                "  Fraction (0-1) of rl_train/rl_val to allocate to 'gen' "
                "(remainder goes to 'ver') [default 0.5]: "
            ).strip()
            if not raw:
                gen_frac = 0.5
                break
            try:
                gen_frac = float(raw)
                if 0 <= gen_frac <= 1:
                    break
            except ValueError:
                pass
            print("    Please enter a number between 0 and 1.")

        rl_gen_train, rl_ver_train = split_two_way(rl_train, fields, gen_frac, rng)
        rl_gen_val, rl_ver_val = split_two_way(rl_val, fields, gen_frac, rng)

        print_distribution("rl_gen_train", rl_gen_train, fields)
        print_distribution("rl_ver_train", rl_ver_train, fields)
        print_distribution("rl_gen_val", rl_gen_val, fields)
        print_distribution("rl_ver_val", rl_ver_val, fields)

        outputs.update(
            {
                "rl_train": rl_train,
                "rl_val": rl_val,
                "rl_gen_train": rl_gen_train,
                "rl_ver_train": rl_ver_train,
                "rl_gen_val": rl_gen_val,
                "rl_ver_val": rl_ver_val,
            }
        )
    else:
        outputs.update({"rl_train": rl_train, "rl_val": rl_val})

    out_dir = args.output_dir or args.input_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting splits to {out_dir} ...")
    for name, recs in outputs.items():
        write_split(recs, out_dir, name)

    print("\nDone.")


if __name__ == "__main__":
    main()
