"""
Data commands - primitives and prompts creation.
"""

from pathlib import Path

from pipeline.core.io import load_json, save_json, save_parquet
from pipeline.tasks import get_task


def create_primitives(
    task_name: str,
    output_path: Path,
    num_puzzles: int = 3100,
    seed: int = 42,
) -> Path:
    """
    Generate raw puzzle data.

    Args:
        task_name: Name of task (e.g., "countdown")
        output_path: Where to save primitives.json
        num_puzzles: Number of puzzles to generate
        seed: Random seed

    Returns:
        Path to created primitives.json
    """
    task = get_task(task_name)
    print(f"Generating {num_puzzles} primitives for task '{task_name}'...")

    primitives = task.create_primitives(num_puzzles, seed)

    save_json(output_path, primitives)
    print(f"Saved {len(primitives)} primitives to {output_path}")

    return output_path


def create_prompts(
    task_name: str,
    primitives_path: Path,
    output_dir: Path,
    split_name: str = "all",
    template_path: Path | None = None,
    seed: int = 42,
    include_assistant_prefix: bool = True,
) -> Path | dict[str, Path]:
    """
    Create prompts from primitives for a given split (or all splits).

    The task's get_split_indices() method determines which primitives
    belong to each split.

    Args:
        task_name: Name of task
        primitives_path: Path to primitives.json
        output_dir: Directory to save prompts (or file path for single split)
        split_name: Name of split (sft, rl, classifier, eval) or "all"
        template_path: Path to template file (default: task's template for this split)
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening

    Returns:
        Path to created prompts file, or dict of paths if split="all"
    """
    task = get_task(task_name)

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
                template_path=None,  # Use default for each split
                seed=seed,
                include_assistant_prefix=include_assistant_prefix,
            )
        return results

    # Single split - output_dir is actually the output file path
    return _create_prompts_single(
        task=task,
        task_name=task_name,
        primitives_path=primitives_path,
        output_path=output_dir,
        split_name=split_name,
        template_path=template_path,
        seed=seed,
        include_assistant_prefix=include_assistant_prefix,
    )


def _create_prompts_single(
    task,
    task_name: str,
    primitives_path: Path,
    output_path: Path,
    split_name: str,
    template_path: Path | None,
    seed: int,
    include_assistant_prefix: bool,
) -> Path:
    """Create prompts for a single split."""
    # Load primitives
    primitives = load_json(primitives_path)

    # Get indices for this split from task
    split_indices = task.get_split_indices(len(primitives), split_name, seed)
    index_set = set(split_indices)

    # Filter primitives
    primitives = [p for p in primitives if p["index"] in index_set]

    # Load template
    if template_path is None:
        # rl_train and rl_val both use rl template
        template_split = "rl" if split_name.startswith("rl") else split_name
        template_path = Path(f"pipeline/tasks/{task_name}/templates/{template_split}.txt")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Determine format from output path
    fmt = "parquet" if str(output_path).endswith(".parquet") else "json"

    print(f"Creating {split_name} prompts for {len(primitives)} primitives...")

    # Format prompts
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

        # For RL/verl compatibility, add extra fields
        if fmt == "parquet":
            record["data_source"] = task_name
            record["reward_model"] = {
                "style": "rule",
                "ground_truth": ground_truth,
            }

        records.append(record)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        save_parquet(output_path, records)
    else:
        save_json(output_path, records)

    print(f"Saved {len(records)} prompts to {output_path}")
    return output_path
