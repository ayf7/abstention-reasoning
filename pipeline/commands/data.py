"""
Data commands - primitives and prompts creation.
"""

import inspect
from pathlib import Path

from pipeline.core.io import load_json, save_json, save_parquet
from pipeline.core.method import TASKS_ROOT, Method, get_primitives_path
from pipeline.tasks import get_task


def create_primitives(
    task_name: str,
    output_path: Path | None = None,
    num_puzzles: int | None = None,
    seed: int = 42,
    **kwargs,
) -> Path:
    """
    Generate raw puzzle data.

    Args:
        task_name: Name of task (e.g., "countdown")
        output_path: Where to save primitives.json (default: artifacts/{task}/primitives.json)
        num_puzzles: Number of puzzles to generate (None = all available)
        seed: Random seed
        **kwargs: Additional task-specific options (e.g., tracer="uniform" for code_output)

    Returns:
        Path to created primitives.json
    """
    task = get_task(task_name)

    # Validate task-specific options against the task's actual signature. Tasks
    # differ in what they accept (only code_output takes `tracer`), so an option
    # the task can't use must fail loudly rather than raise a bare TypeError or
    # be silently dropped.
    if kwargs:
        sig = inspect.signature(task.create_primitives)
        takes_var_kw = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if not takes_var_kw:
            unsupported = [key for key in kwargs if key not in sig.parameters]
            if unsupported:
                raise ValueError(
                    f"Task '{task_name}' does not support "
                    f"{', '.join(repr(k) for k in sorted(unsupported))}. "
                    f"That option applies to other tasks only."
                )

    # Default output path
    if output_path is None:
        output_path = get_primitives_path(task_name)

    count_str = str(num_puzzles) if num_puzzles is not None else "all"
    print(f"Generating {count_str} primitives for task '{task_name}'...")

    primitives = task.create_primitives(num_puzzles, seed, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, primitives)
    print(f"Saved {len(primitives)} primitives to {output_path}")

    return output_path


def create_prompts(
    task_name: str,
    method_name: str | None = None,
    primitives_path: Path | None = None,
    output_dir: Path | None = None,
    split_name: str = "all",
    seed: int = 42,
    include_assistant_prefix: bool = True,
    num_hints: int | None = None,
    force_json: bool = False,
) -> Path | dict[str, Path]:
    """
    Create prompts from primitives for a given split (or all splits).

    The task's get_split_indices() method determines which primitives
    belong to each split.

    Args:
        task_name: Name of task
        method_name: Method name for auto-derived paths and template selection
        primitives_path: Path to primitives.json (default: artifacts/{task}/primitives.json)
        output_dir: Directory to save prompts (default: artifacts/{task}/{method}/prompts/)
        split_name: Name of split (sft, rl_train, rl_val, eval, eval_augmented, or 'all')
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening
        num_hints: Number of hints to extract from prefix_hints (0-6). If None, no hint injection.

    Returns:
        Path to created prompts file, or dict of paths if split="all"
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default primitives path
    if primitives_path is None:
        primitives_path = get_primitives_path(task_name)

    # Default output directory
    if output_dir is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_dir = method.prompts_dir(task_name)

    # Get template variant from method or task default
    template_variant = None
    if method is not None:
        template_variant = method.template_variant
    else:
        template_variant = getattr(task, "default_template_variant", None)

    # Get assistant_prefix from method (if provided) or task default
    assistant_prefix = None
    if method is not None and method.assistant_prefix is not None:
        assistant_prefix = method.assistant_prefix
    else:
        assistant_prefix = getattr(task, "assistant_prefix", None)

    # Handle "all" splits
    if split_name == "all":
        output_dir.mkdir(parents=True, exist_ok=True)
        # eval_augmented is the split nearly every recorded evaluation reads
        # (prompts/eval_augmented.json). Leaving it out of "all" meant the
        # documented setup path silently produced none of it.
        splits = ["sft", "rl_train", "rl_val", "eval", "eval_augmented"]
        results = {}
        for split in splits:
            # Determine output format based on split
            ext = ".json" if force_json else (".parquet" if split.startswith("rl") else ".json")
            output_path = output_dir / f"{split}{ext}"
            results[split] = _create_prompts_single(
                task=task,
                task_name=task_name,
                primitives_path=primitives_path,
                output_path=output_path,
                split_name=split,
                template_variant=template_variant,
                seed=seed,
                include_assistant_prefix=include_assistant_prefix,
                assistant_prefix=assistant_prefix,
                method=method,
                num_hints=num_hints,
            )
        return results

    # Single split
    ext = ".json" if force_json else (".parquet" if split_name.startswith("rl") else ".json")
    output_path = output_dir / f"{split_name}{ext}"
    return _create_prompts_single(
        task=task,
        task_name=task_name,
        primitives_path=primitives_path,
        output_path=output_path,
        split_name=split_name,
        template_variant=template_variant,
        seed=seed,
        include_assistant_prefix=include_assistant_prefix,
        assistant_prefix=assistant_prefix,
        method=method,
        num_hints=num_hints,
    )


def _create_prompts_single(
    task,
    task_name: str,
    primitives_path: Path,
    output_path: Path,
    split_name: str,
    template_variant: str | None,
    seed: int,
    include_assistant_prefix: bool,
    assistant_prefix: str | None = None,
    method: "Method | None" = None,
    num_hints: int | None = None,
) -> Path:
    """Create prompts for a single split."""
    # Override task's assistant_prefix if method provided one
    if assistant_prefix is not None:
        task.assistant_prefix = assistant_prefix

    # Load primitives
    primitives = load_json(primitives_path)

    # Get indices for this split from task (pass primitives for stratified sampling)
    split_indices = task.get_split_indices(len(primitives), split_name, seed, primitives)
    index_set = set(split_indices)

    # Filter primitives
    primitives = [p for p in primitives if p["index"] in index_set]

    # Determine format from output path
    fmt = "parquet" if str(output_path).endswith(".parquet") else "json"

    # For RL splits (parquet), store primitives only - template applied at runtime
    # For other splits (json), apply template now
    if fmt == "parquet":
        print(f"Creating {split_name} data for {len(primitives)} primitives (template applied at runtime)...")

        # Determine interaction name for multi-turn methods
        # Maps method name to interaction class name (e.g., "hint" -> "countdown_hint")
        interaction_name = None
        if method is not None and method.multi_turn:
            # Use method's interaction_name if specified, else default to {task}_{method}
            interaction_name = method.interaction_name or f"{task_name}_{method.name}"
            print(f"  Multi-turn enabled: interaction_name={interaction_name}")

        records = []
        for primitive in primitives:
            # Enrich primitive with derived fields if task supports it
            # (verl's runtime template does simple substitution, so we pre-compute fields)
            if hasattr(task, 'enrich_primitive_for_rl'):
                enriched_primitive = task.enrich_primitive_for_rl(primitive)
            else:
                enriched_primitive = primitive

            ground_truth = task.get_ground_truth(primitive)

            # Build extra_info with interaction_kwargs for SGLang multi-turn
            extra_info = {
                "index": primitive["index"],
            }
            if interaction_name is not None:
                extra_info["interaction_kwargs"] = {
                    "name": interaction_name,
                    "ground_truth": ground_truth,
                }

            record = {
                "index": primitive["index"],
                "primitive": enriched_primitive,  # Store enriched primitive for runtime template
                "ground_truth": ground_truth,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
                "data_source": task_name,
                "assistant_prefix": assistant_prefix,  # For verl runtime consistency
                "extra_info": extra_info,  # For SGLang interaction system
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
            }
            records.append(record)

        # Print reminder for verl config
        if assistant_prefix:
            print(f"  Note: Set verl config data.runtime_assistant_prefix=\"{assistant_prefix}\"")
    else:
        # Resolve template path (map split names to template file names)
        template_split = split_name
        if split_name == "eval_augmented":
            template_split = "eval"
        elif split_name in ("rl_train", "rl_val"):
            template_split = "rl"
        if template_variant:
            template_path = TASKS_ROOT / task_name / "templates" / template_variant / f"{template_split}.txt"
        else:
            # Legacy fallback (no variant subdirectory)
            template_path = TASKS_ROOT / task_name / "templates" / f"{template_split}.txt"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}. "
                f"Check that --method is correct."
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        print(f"Creating {split_name} prompts for {len(primitives)} primitives (template: {template_path})...")
        records = []
        for primitive in primitives:
            # Inject hints if --num-hints is specified
            # Supports both hint_exprs (list, countdown) and prefix_hints (dict, competition_math)
            if num_hints is not None:
                hints_list = primitive.get("hint_exprs", [])
                if not hints_list:
                    prefix_hints = primitive.get("prefix_hints", {})
                    for i in range(1, 7):
                        key = f"hint_{i}"
                        if key in prefix_hints:
                            hints_list.append(prefix_hints[key])
                primitive = {**primitive, "hints": hints_list[:num_hints]}

            prompt = task.format_prompt(primitive, template, include_assistant_prefix)
            ground_truth = task.get_ground_truth(primitive)
            record = {
                "index": primitive["index"],
                "prompt": prompt,
                "ground_truth": ground_truth,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
            }
            records.append(record)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        save_parquet(output_path, records)
    else:
        save_json(output_path, records)

    print(f"Saved {len(records)} records to {output_path}")
    return output_path


def create_verify_prompts(
    task_name: str,
    source_dataset_path: Path,
    method_name: str = "verify",
    primitives_path: Path | None = None,
    output_dir: Path | None = None,
    split_name: str = "all",
    seed: int = 42,
    include_assistant_prefix: bool = True,
) -> Path | dict[str, Path]:
    """
    Create verification prompts from a source model's generations.

    Takes an existing dataset (e.g., from a simple model's generation) and creates
    prompts that ask a model to verify whether each generation is correct.

    Args:
        task_name: Name of task (e.g., "countdown")
        source_dataset_path: Path to source model's dataset JSON (must have index, generation, correct fields)
        method_name: Method name for path derivation and template selection (default: "verify")
        primitives_path: Path to primitives.json (default: artifacts/{task}/primitives.json)
        output_dir: Output directory (default: artifacts/{task}/{method}/prompts/)
        split_name: Which split to create ("sft", "rl_train", "rl_val", "eval", "eval_augmented", or "all")
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening

    Returns:
        Path to created prompts file, or dict of paths if split="all"
    """
    task = get_task(task_name)
    method = Method.load(method_name, task_name)

    # Default paths
    if primitives_path is None:
        primitives_path = get_primitives_path(task_name)
    if output_dir is None:
        output_dir = method.prompts_dir(task_name)

    # Get assistant prefix from method or task
    assistant_prefix = method.assistant_prefix or getattr(task, "assistant_prefix", None)
    if assistant_prefix is not None:
        task.assistant_prefix = assistant_prefix

    # Load source dataset and primitives
    source_records = load_json(source_dataset_path)
    # Handle evaluate results format ({"details": [...]}) vs flat dataset format
    if isinstance(source_records, dict) and "details" in source_records:
        source_records = source_records["details"]
    primitives = load_json(primitives_path)

    # Filter out incomplete generations (truncated by token limit)
    total = len(source_records)
    source_records = [r for r in source_records if r.get("finish_reason") != "length"]
    if len(source_records) < total:
        print(f"Filtered out {total - len(source_records)} truncated generations (finish_reason=length)")

    # Build lookups
    primitives_by_index = {p["index"]: p for p in primitives}
    source_by_index = {r["index"]: r for r in source_records}
    source_indices = set(source_by_index.keys())

    print(f"Loaded {len(source_records)} source records from {source_dataset_path}")
    print(f"Loaded {len(primitives)} primitives from {primitives_path}")

    # Handle "all" splits
    if split_name == "all":
        output_dir.mkdir(parents=True, exist_ok=True)
        # eval_augmented is the split nearly every recorded evaluation reads
        # (prompts/eval_augmented.json). Leaving it out of "all" meant the
        # documented setup path silently produced none of it.
        splits = ["sft", "rl_train", "rl_val", "eval", "eval_augmented"]
        results = {}
        for split in splits:
            ext = ".parquet" if split.startswith("rl") else ".json"
            output_path = output_dir / f"{split}{ext}"
            results[split] = _create_verify_prompts_single(
                task=task,
                task_name=task_name,
                primitives_by_index=primitives_by_index,
                source_by_index=source_by_index,
                source_indices=source_indices,
                num_primitives=len(primitives),
                output_path=output_path,
                split_name=split,
                method=method,
                seed=seed,
                include_assistant_prefix=include_assistant_prefix,
                assistant_prefix=assistant_prefix,
            )
        return results

    # Single split
    # If output_dir looks like a file path (has .json/.parquet extension), use it directly
    if output_dir.suffix in (".json", ".parquet"):
        output_path = output_dir
    else:
        ext = ".parquet" if split_name.startswith("rl") else ".json"
        output_path = output_dir / f"{split_name}{ext}"
    return _create_verify_prompts_single(
        task=task,
        task_name=task_name,
        primitives_by_index=primitives_by_index,
        source_by_index=source_by_index,
        source_indices=source_indices,
        num_primitives=len(primitives),
        output_path=output_path,
        split_name=split_name,
        method=method,
        seed=seed,
        include_assistant_prefix=include_assistant_prefix,
        assistant_prefix=assistant_prefix,
    )


def _create_verify_prompts_single(
    task,
    task_name: str,
    primitives_by_index: dict,
    source_by_index: dict,
    source_indices: set,
    num_primitives: int,
    output_path: Path,
    split_name: str,
    method: "Method",
    seed: int,
    include_assistant_prefix: bool,
    assistant_prefix: str | None = None,
) -> Path:
    """Create verification prompts for a single split."""
    # Get split indices, filtered to those present in source dataset
    split_indices = task.get_split_indices(num_primitives, split_name, seed)
    valid_indices = [i for i in split_indices if i in source_indices]

    if not valid_indices:
        print(f"  Warning: No source records found for split '{split_name}', skipping")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = "parquet" if str(output_path).endswith(".parquet") else "json"
        if fmt == "parquet":
            save_parquet(output_path, [])
        else:
            save_json(output_path, [])
        return output_path

    # Load template
    template_variant = method.template_variant
    template_split = split_name
    if split_name == "eval_augmented":
        template_split = "eval"
    elif split_name in ("rl_train", "rl_val"):
        template_split = "rl"
    template_path = TASKS_ROOT / task_name / "templates" / template_variant / f"{template_split}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    fmt = "parquet" if str(output_path).endswith(".parquet") else "json"

    print(f"Creating {split_name} verify prompts for {len(valid_indices)} items (template: {template_path})...")

    records = []
    for idx in valid_indices:
        primitive = primitives_by_index[idx]
        source = source_by_index[idx]
        generation_text = source["generation"]
        generation_correct = source["correct"]

        # Build ground_truth: for JSON include full task ground truth, for parquet keep minimal
        task_ground_truth = task.get_ground_truth(primitive)
        ground_truth_full = {"generation_correct": generation_correct}
        if isinstance(task_ground_truth, dict):
            ground_truth_full.update(task_ground_truth)
        # Minimal ground_truth for parquet (avoids varying-length arrays breaking verl batching)
        ground_truth_minimal = {"generation_correct": generation_correct}

        if fmt == "parquet":
            # RL: store enriched primitive with generation field for runtime template substitution
            enriched = dict(primitive)
            enriched["generation"] = generation_text
            enriched["generation_correct"] = generation_correct
            if hasattr(task, 'enrich_primitive_for_rl'):
                enriched = task.enrich_primitive_for_rl(enriched)

            # For RL reward_model, only include what compute_score_verify needs
            # (full ground_truth has varying-length arrays that break verl's batching)
            reward_ground_truth = {"generation_correct": generation_correct}

            record = {
                "index": idx,
                "primitive": enriched,
                "ground_truth": ground_truth_minimal,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
                "data_source": task_name,
                "assistant_prefix": assistant_prefix,
                "extra_info": {"index": idx},
                "reward_model": {
                    "style": "rule",
                    "ground_truth": reward_ground_truth,
                },
            }
        else:
            # JSON: substitute {generation} first, then let format_prompt handle task fields
            template_with_gen = template.replace("{generation}", generation_text)
            # Also add generation_correct to primitive so check_correctness can use it
            enriched_primitive = {**primitive, "generation_correct": generation_correct}
            prompt = task.format_prompt(enriched_primitive, template_with_gen, include_assistant_prefix)

            record = {
                "index": idx,
                "prompt": prompt,
                "ground_truth": ground_truth_full,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
            }

        records.append(record)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        save_parquet(output_path, records)
        if assistant_prefix:
            print(f"  Note: Set verl config data.runtime_assistant_prefix=\"{assistant_prefix}\"")
    else:
        save_json(output_path, records)

    n_correct = sum(1 for i in valid_indices if source_by_index[i]["correct"])
    n_wrong = len(valid_indices) - n_correct
    print(f"Saved {len(records)} records to {output_path} ({n_correct} correct, {n_wrong} incorrect)")
    return output_path


# === OOD (Out-of-Distribution) evaluation datasets ===


OOD_DATASETS = {
    "minerva_math": "Minerva Math university-level STEM (272 problems)",
    "math500": "MATH-500 test subset (500 problems)",
    "olympiad_bench": "OlympiadBench text-only English math (674 problems)",
    "gsm8k": "GSM8K grade-school math (1,319 test problems)",
    "aime2024": "AIME 2024 I + II (30 problems)",
    "unanswerable_math": "Synthetic Unanswerable Math (500 answerable + 500 unanswerable)",
}


def _load_ood_minerva_math(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load Minerva Math dataset (math-ai/minervamath). 272 university-level STEM problems."""
    import random
    from datasets import load_dataset

    ds = load_dataset("math-ai/minervamath", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "minerva_math",
            "level": "university",
            "problem": row["question"],
            "answer": row["answer"],
        })
    return primitives


def _load_ood_math500(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load MATH-500 dataset (di-zhang-fdu/MATH500)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("di-zhang-fdu/MATH500", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": row["subject"],
            "level": f"Level {row['level']}",
            "problem": row["problem"],
            "answer": row["answer"],
        })
    return primitives


def _load_ood_olympiad_bench(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load OlympiadBench text-only English math (Hothan/OlympiadBench)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        # final_answer is a list; join with comma for multi-answer problems
        answers = row["final_answer"]
        answer = answers[0] if len(answers) == 1 else ", ".join(answers)

        primitives.append({
            "index": idx,
            "variant": row.get("subfield", "unknown"),
            "level": row.get("difficulty", "unknown"),
            "problem": row["question"],
            "answer": answer,
            "answer_type": row.get("answer_type", "unknown"),
            "is_multiple_answer": row.get("is_multiple_answer", False),
        })
    return primitives


def _load_ood_gsm8k(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load GSM8K test set (openai/gsm8k)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        # Extract numeric answer after "####"
        answer = row["answer"].split("####")[-1].strip()
        primitives.append({
            "index": idx,
            "variant": "gsm8k",
            "level": "grade_school",
            "problem": row["question"],
            "answer": answer,
        })
    return primitives


def _load_ood_aime2024(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load AIME 2024 (HuggingFaceH4/aime_2024)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "aime",
            "level": "competition",
            "problem": row["problem"],
            "answer": str(row["answer"]),
        })
    return primitives


def _load_ood_unanswerable_math(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load Synthetic Unanswerable Math (lime-nlp/Synthetic_Unanswerable_Math).

    Each row contains an answerable and unanswerable version. By default,
    samples 500 rows producing 500 answerable + 500 unanswerable primitives
    (1000 total). If num_problems is given, that many rows are sampled and
    both versions are created from each row.
    """
    import random
    from datasets import load_dataset

    ds = load_dataset("lime-nlp/Synthetic_Unanswerable_Math", data_files="synthetic_unanswerable_math.parquet", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    num_rows = (num_problems or 500)
    rows = rows[:num_rows]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "answerable",
            "level": "unknown",
            "problem": row["answerable_question"],
            "answer": row["ground_truth"],
        })
        primitives.append({
            "index": num_rows + idx,
            "variant": "unanswerable",
            "level": "unknown",
            "problem": row["unanswerable_question"],
            "answer": None,
        })
    return primitives


_OOD_LOADERS = {
    "minerva_math": _load_ood_minerva_math,
    "math500": _load_ood_math500,
    "olympiad_bench": _load_ood_olympiad_bench,
    "gsm8k": _load_ood_gsm8k,
    "aime2024": _load_ood_aime2024,
    "unanswerable_math": _load_ood_unanswerable_math,
}


def create_ood_prompts(
    task_name: str,
    dataset_name: str,
    method_name: str | None = None,
    output_path: Path | None = None,
    num_problems: int | None = None,
    seed: int = 42,
    include_assistant_prefix: bool = True,
) -> Path:
    """
    Create evaluation prompts from an out-of-distribution dataset.

    Loads an external math benchmark, normalizes it to the primitives format,
    then formats prompts using the task's eval template. The resulting file
    can be passed to `evaluate --prompts <path>`.

    Args:
        task_name: Task whose templates/check_correctness to use (e.g., "competition_math")
        dataset_name: OOD dataset key (math500, olympiad_bench, gsm8k, aime2024)
        method_name: Method name for template selection and output path derivation
        output_path: Explicit output path (default: artifacts/{task}/{method}/prompts/ood_{dataset}.json)
        num_problems: Limit number of problems (None = all)
        seed: Random seed for shuffling
        include_assistant_prefix: Whether to include assistant's opening in prompt

    Returns:
        Path to created prompts file
    """
    if dataset_name not in _OOD_LOADERS:
        available = ", ".join(sorted(_OOD_LOADERS.keys()))
        raise ValueError(f"Unknown OOD dataset: {dataset_name}. Available: {available}")

    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Get assistant prefix from method or task
    if method is not None and method.assistant_prefix is not None:
        task.assistant_prefix = method.assistant_prefix

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_path = method.prompts_dir(task_name) / f"ood_{dataset_name}.json"

    # Load template (always use eval template)
    template_variant = method.template_variant if method else None
    if template_variant:
        template_path = TASKS_ROOT / task_name / "templates" / template_variant / "eval.txt"
    else:
        template_path = TASKS_ROOT / task_name / "templates" / "eval.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Load OOD dataset
    print(f"Loading OOD dataset '{dataset_name}' ({OOD_DATASETS[dataset_name]})...")
    primitives = _OOD_LOADERS[dataset_name](num_problems=num_problems, seed=seed)
    print(f"Loaded {len(primitives)} problems")

    # Format prompts
    print(f"Formatting prompts (template: {template_path})...")
    records = []
    for primitive in primitives:
        prompt = task.format_prompt(primitive, template, include_assistant_prefix)
        ground_truth = {
            "answer": primitive["answer"],
            "variant": primitive.get("variant", "unknown"),
            "level": primitive.get("level", "unknown"),
            "problem": primitive["problem"],
        }
        records.append({
            "index": primitive["index"],
            "prompt": prompt,
            "ground_truth": ground_truth,
            "variant": primitive.get("variant", "unknown"),
            "split": f"ood_{dataset_name}",
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, records)
    print(f"Saved {len(records)} prompts to {output_path}")
    return output_path
