# Text-to-SQL data pipeline

This document describes the end-to-end process used to build the SQL task's
training data (`data/sql_conceptual/` and `data/sql_partial_sql/`) from the raw
Spider/BIRD/SParC/CoSQL datasets. Follow this if you ever need to regenerate
`primitives.json` or the `sft_*`/`rl_*` splits from scratch.

## Pipeline overview

```
raw datasets                 create_primitives.py
(data/sql/raw/*)      ─────────────────────────────▶  primitives_{source}_{split}.json
                                                        (9 files, index per-file, 0-based)

                              run_generate_hints.sh
                              (-> generate_hints.py, per file/style, TRAPI)
primitives_{source}_{split}.json ───────────────────▶  primitives_{source}_{split}_hints_{style}.json
                                                        (18 files: 9 sources x 2 styles
                                                         {conceptual, partial_sql}; records that
                                                         fail the FILTER become `null` placeholders)

                              combine_and_filter_hints.py
primitives_*_hints_{style}.json ────────────────────▶  data/sql_{style variant}/problems/primitives.json
                                                        (single file, dev+train merged,
                                                         globally unique `index`, largest-9-BIRD-db
                                                         databases excluded)

                              (symlink databases/ -- see below)

                              scripts/split_dataset.py
data/sql_{variant}/problems/primitives.json ────────▶  eval.json, sft_train.json, sft_val.json,
                                                        rl_train.json, rl_val.json, ...

                              pipeline generate (generate_sft_data.sh etc.)
sft_train.json / sft_val.json ──────────────────────▶  data/sql_{variant}/sft_datasets/*.json
```

## Step 1: `create_primitives.py` -- raw datasets -> per-source primitive files

```
python -m pipeline.tasks.sql.create_primitives --raw-dir data/sql/raw --datasets spider bird sparc cosql
```

Converts each raw dataset into the common primitive schema (`index`, `variant`
(=db_id), `level`, `problem`, `answer`, `prefix_hints={}`, `metadata`).
**Does NOT combine anything across sources.** It writes one output file per
source-and-split, each independently 0-indexed via `enumerate(data)`:

- `primitives_spider_train.json`, `primitives_spider_train_others.json`, `primitives_spider_dev.json`
- `primitives_bird_train.json`, `primitives_bird_dev.json`
- `primitives_sparc_train.json`, `primitives_sparc_dev.json`
- `primitives_cosql_train.json`, `primitives_cosql_dev.json`

SParC and CoSQL are multi-turn datasets; `process_sparc`/`process_cosql`
flatten each interaction to its final turn and preserve the earlier turns as
`metadata.intermediate_turns`.

**Important**: indices here are *not* globally unique (each file restarts at
0). Do not use these files directly as input to `pipeline generate` or
`split_dataset.py` -- always go through the combine step below first.

## Step 2: `run_generate_hints.sh` -- generate progressive hints via TRAPI

```
bash pipeline/tasks/sql/run_generate_hints.sh --style both --split all --filter complex
```

For every `primitives_{source}_{split}.json` file, calls `generate_hints.py`
to generate 5 independent progressive hints (`hint_1..hint_5`) per record, in
two styles:
- `conceptual`: natural-language hints only
- `partial_sql`: hints may include partial SQL syntax fragments

`--filter complex` (the default) restricts hint generation to records where
`metadata.is_complex` and `metadata.multi_step` are both true (see
`is_multi_step_query`/`is_complex_query` in `create_primitives.py`). Records
that don't pass the filter are written out as `null` placeholders at their
original index position (NOT omitted), so the output file has the same length
as the input file. Output: `primitives_{source}_{split}_hints_{style}.json`
(18 files total: 9 source/split files x 2 styles).

## Step 3: `combine_and_filter_hints.py` -- merge + filter into one primitives.json

```
python -m pipeline.tasks.sql.combine_and_filter_hints --style conceptual   --output data/sql_conceptual/problems/primitives.json
python -m pipeline.tasks.sql.combine_and_filter_hints --style partial_sql --output data/sql_partial_sql/problems/primitives.json
```

For a given style, loads all 9 `_hints_{style}.json` files (dev sources
first, then train sources -- see `DEV_PREFIXES`/`TRAIN_PREFIXES`), drops the
`null` placeholders left over from Step 2's filter, concatenates everything,
then applies one further filter:

- **Largest-9-BIRD-databases exclusion**: records whose `metadata.source ==
  "bird"` and `metadata.db_id` is one of the 9 largest BIRD sqlite databases
  (by raw file size, used as a proxy for schema/data size to control prompt
  length) are dropped. The 9 excluded db_ids are hardcoded in
  `LARGEST_BIRD_DBS` in the script: `bike_share_1`, `donor`,
  `codebase_comments`, `movie_platform`, `world_development_indicators`,
  `language_corpus`, `talkingdata`, `music_platform_2`, `coinmarketcap`.
  Recomputed via:
  ```
  find data/sql/raw/bird/databases -iname "*.sqlite" -exec du -b {} \; | sort -rn | head -9
  ```

Finally, reassigns a fresh **globally unique** `index` (0..N-1) over the
combined, filtered list, preserving the old per-source index in
`metadata.original_split_index`.

Because Step 2 generates both styles from the same input file in the same
record order, `conceptual` and `primitives.json` end up with the exact same
`index` <-> `problem`/`answer` mapping as `partial_sql`'s `primitives.json` --
i.e. index `k` refers to the same underlying problem in both variants.

### Why this script exists (instead of the older `combine_hints.py`)

`combine_hints.py` (still present, kept for the dev/train-separated output it
produces under `data/sql/primitives_combined_{dev,train}_hints_{style}.json`)
has two properties that `combine_and_filter_hints.py` fixes for the purposes
of building the final `primitives.json`:

1. It writes dev and train out as **two separate files**, each independently
   reindexed from 0 -- so a dev record and a train record can share the same
   `index`, which is exactly the bug that silently dropped ~300 records from
   `pipeline generate` output (`records_by_index` is keyed only by `index` and
   silently overwrites on collision). `combine_and_filter_hints.py` merges
   dev+train into one list before reindexing, guaranteeing uniqueness across
   the whole file.
2. It does not apply the largest-9-BIRD-db filter.

If you ever need separate dev/train hint files for some other purpose,
`combine_hints.py` is still the right tool -- just don't use its output
directly as `primitives.json` without re-merging and filtering.

**Historical note**: the `primitives.json` files in place before 2026-10-02
were built by some ad-hoc/one-off process that concatenated all 18
`_hints_{style}.json` files directly, without any reindexing step at all
(confirmed by the complete absence of `metadata.original_split_index` and a
record ordering that didn't match `combine_hints.py`'s prefix lists). That
process is not in the repo and was never identified. `combine_and_filter_hints.py`
was written to replace it with something reproducible; it was validated by
reproducing the exact historical record count (16248) from the is_complex/
multi_step + largest-9-BIRD-db filters before being adopted.

## Step 4: `databases/` symlinks

`verl/recipe/sql/reward_function.py` and `pipeline/core/method.py` resolve
SQL execution against `$DATABASE_PATH/<db_id>/<db_id>.sqlite`. Locally,
`DATABASE_PATH` defaults to `data/<data_name>/databases` (see
`pipeline/core/method.py`), so each dataset variant needs a `databases/`
folder containing one entry per `db_id` referenced in its `primitives.json`.

These are git-ignored (see `.gitignore`) and must be (re)built locally as
symlinks into the real sqlite files under `data/sql/raw/`:

- BIRD db_ids -> `data/sql/raw/bird/databases/<db_id>`
- Spider/SParC/CoSQL db_ids -> `data/sql/raw/spider/spider_data/database/<db_id>`
  (SParC and CoSQL reuse Spider's database files; all of their db_ids are a
  subset of Spider's)

Rebuild with (example; adapt per dataset variant):

```python
import json
from pathlib import Path

REPO = Path(".").resolve()
SEARCH_DIRS = [
    REPO / "data/sql/raw/bird/databases",
    REPO / "data/sql/raw/spider/spider_data/database",
]

def find_db_dir(db_id):
    for base in SEARCH_DIRS:
        cand = base / db_id
        if (cand / f"{db_id}.sqlite").exists():
            return cand
    return None

for variant_dir in ["sql_conceptual", "sql_partial_sql"]:
    primitives = json.load(open(REPO / f"data/{variant_dir}/problems/primitives.json"))
    out_dir = REPO / f"data/{variant_dir}/databases"
    out_dir.mkdir(parents=True, exist_ok=True)
    for db_id in sorted(set(x["variant"] for x in primitives)):
        target = find_db_dir(db_id)
        link = out_dir / db_id
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
```

**On remote/cluster pods**: these symlinks are baked with absolute paths from
the dev machine and won't resolve on a different host. Remote launch scripts
(`launch_scripts/initialization_script.sh`) instead export
`DATABASE_PATH=/data/tanyagoyal/sql_databases`, a flat PVC-mounted directory
with the same `<db_id>/<db_id>.sqlite` layout, bypassing the symlinks
entirely. If you add new db_ids locally, make sure the same databases are
also present under that PVC path (e.g. via the `data-mover` pod) before
training/generating remotely.

## Step 5: `scripts/split_dataset.py` -- stratified train/val/eval/rl splits

```
python scripts/split_dataset.py data/sql_conceptual/problems/primitives.json
python scripts/split_dataset.py data/sql_partial_sql/problems/primitives.json
```

Reads a single primitives file and writes `eval.json`, `sft_train.json`,
`sft_val.json`, `rl_train.json`, `rl_val.json` (plus `rl_gen_*`/`rl_ver_*`
variants) stratified by source/difficulty into the same directory (or
`--output-dir` if given), using a 90:10 train/val split within the sft and rl
groups. This step is purely a deterministic redistribution of whatever
records (and indices) are already in the input file -- it does not introduce
or fix index collisions itself, so **always run it against a
`combine_and_filter_hints.py` output**, never against a raw/ad-hoc combined
file.

## Step 6: `pipeline generate` -- produce final SFT datasets

```
bash launch_scripts/generate_sft_data.sh ...
```

Runs model generation + correctness verification over `sft_train.json`/
`sft_val.json`, producing `data/sql_{variant}/sft_datasets/sft_{train,val}__baseline.json`.
This is the expensive (LLM + SQL execution) step and should only be re-run
after confirming Steps 1-5 produced a clean, collision-free `primitives.json`.
