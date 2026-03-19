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
                assistant_prefix=assistant_prefix,
                method=method,
                num_hints=num_hints,
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

    # Set hint_source from method config if provided
    if method is not None and method.hint_source is not None:
        task.hint_source = method.hint_source

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
        # Resolve template path (eval_augmented uses the eval template)
        template_split = "eval" if split_name == "eval_augmented" else split_name
        if template_variant:
            template_path = Path(f"pipeline/tasks/{task_name}/templates/{template_variant}/{template_split}.txt")
        else:
            # Legacy fallback (no variant subdirectory)
            template_path = Path(f"pipeline/tasks/{task_name}/templates/{template_split}.txt")

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
