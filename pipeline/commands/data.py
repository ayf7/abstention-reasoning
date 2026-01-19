"""
Data commands - primitives and prompts creation.
"""

from pathlib import Path

from pipeline.core.io import load_json, save_json, save_parquet
from pipeline.core.method import Method, get_primitives_path
from pipeline.tasks import get_task


def create_primitives(
    task_name: str,
    output_path: Path | None = None,
    num_puzzles: int | None = None,
    seed: int = 42,
) -> Path:
    """
    Generate raw puzzle data.

    Args:
        task_name: Name of task (e.g., "countdown")
        output_path: Where to save primitives.json (default: artifacts/{task}/primitives.json)
        num_puzzles: Number of puzzles to generate (None = all available)
        seed: Random seed

    Returns:
        Path to created primitives.json
    """
    task = get_task(task_name)

    # Default output path
    if output_path is None:
        output_path = get_primitives_path(task_name)

    count_str = str(num_puzzles) if num_puzzles is not None else "all"
    print(f"Generating {count_str} primitives for task '{task_name}'...")

    primitives = task.create_primitives(num_puzzles, seed)

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
        split_name: Name of split (sft, rl_train, rl_val, classifier, eval, or 'all')
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening

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

    # Handle "all" splits
    if split_name == "all":
        output_dir.mkdir(parents=True, exist_ok=True)
        splits = ["sft", "rl_train", "rl_val", "classifier", "eval"]
        results = {}
        for split in splits:
            # Determine output format based on split
            ext = ".parquet" if split.startswith("rl") else ".json"
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
            )
        return results

    # Single split
    ext = ".parquet" if split_name.startswith("rl") else ".json"
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
) -> Path:
    """Create prompts for a single split."""
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
        records = []
        for primitive in primitives:
            ground_truth = task.get_ground_truth(primitive)
            record = {
                "index": primitive["index"],
                "primitive": primitive,  # Store raw primitive for runtime template application
                "ground_truth": ground_truth,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
                "data_source": task_name,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
            }
            records.append(record)
    else:
        # Resolve template path
        if template_variant:
            template_path = Path(f"pipeline/tasks/{task_name}/templates/{template_variant}/{split_name}.txt")
        else:
            # Legacy fallback (no variant subdirectory)
            template_path = Path(f"pipeline/tasks/{task_name}/templates/{split_name}.txt")

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
