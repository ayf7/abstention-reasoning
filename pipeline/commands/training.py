"""
Training commands - SFT, RL, and checkpoint conversion.
"""

import re
from pathlib import Path

from pipeline.core.io import load_json
from pipeline.core.method import Method
from pipeline.core.utils import model_short_name as _get_model_short_name
from pipeline.tasks import get_task


def _model_project_tag(model_path: str) -> str:
    """Extract model family for wandb project names, stripping experiment suffixes.

    Examples:
        "Qwen/Qwen3-4B"           -> "qwen3-4b"
        ".../sft/qwen3-4b-up4x/model" -> "qwen3-4b"
        ".../sft/qwen2.5-1.5b/model"  -> "qwen2.5-1.5b"
        ".../rl/default/model"         -> "default"
    """
    name = _get_model_short_name(model_path)
    # Truncate after model size indicator (e.g., "qwen3-4b-up4x" -> "qwen3-4b")
    match = re.match(r'(.*?\d+\.?\d*b)', name, re.IGNORECASE)
    return match.group(1) if match else name


def train_sft(
    task_name: str,
    base_model: str,
    method_name: str | None = None,
    run_id: str | None = None,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 4096,
    bf16: bool = True,
    report_to: str = "wandb",
    project_name: str | None = None,
    experiment_name: str | None = None,
    include_abstained: bool = True,
    include_wrong_valid_format: bool = False,
    upsample_hint: int = 1,
    max_correct: int | None = None,
    upsample_abstain: int = 1,
) -> Path:
    """
    Train an SFT model on generated dataset.

    Filters to correct (and optionally abstained) examples, formats as
    prompt/completion pairs, and trains using TRL's SFTTrainer.

    Args:
        task_name: Name of task
        base_model: Base model to fine-tune
        method_name: Method name for auto-derived paths
        run_id: Run identifier for organizing outputs (default: "default")
        dataset_path: Path to generated dataset (default: artifacts/{task}/{method}/datasets/sft_{model}.json)
        output_path: Where to save trained model (default: artifacts/{task}/{method}/models/sft/{run_id}/model)
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_length: Maximum sequence length
        bf16: Use bfloat16 training
        report_to: Reporting integration ("none", "wandb", etc.)
        project_name: Wandb project name (default: {task}-sft)
        experiment_name: Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})
        include_abstained: Include abstained examples in training (default: True)
        include_wrong_valid_format: Include wrong answers with valid format (task-specific, default: False)

    Returns:
        Path to trained model
    """
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default dataset path - look for any sft_*.json in datasets dir
    if dataset_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --dataset must be specified. "
                "Use --method to auto-derive paths, or --dataset for explicit paths."
            )
        datasets_dir = method.datasets_dir(task_name)
        sft_files = list(datasets_dir.glob("sft_*.json"))
        if not sft_files:
            raise FileNotFoundError(
                f"No SFT datasets found in {datasets_dir}. "
                f"Run 'python -m pipeline generate --task {task_name} --method {method_name}' first."
            )
        dataset_path = sft_files[0]  # Use most recent or only one
        if len(sft_files) > 1:
            print(f"Warning: Multiple SFT datasets found, using {dataset_path}")

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        method.ensure_sft_run_dir(task_name, run_id)
        output_path = method.sft_model_path(task_name, run_id)

    # Generate project name if not provided: {task}-sft-{model_short_name}
    if project_name is None:
        project_name = f"{task_name}-sft-{_model_project_tag(base_model)}"

    # Generate experiment name if not provided
    # Format: {method}-{run_id}-{YYYYMMDD}
    # e.g., "simple_abstention-default-20240115"
    if experiment_name is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        method_str = method_name if method_name else "default"
        run_id_str = run_id if run_id else "default"
        experiment_name = f"{method_str}-{run_id_str}-{date_str}"

    run_id_display = run_id or "default"
    print(f"=== SFT Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"Method: {method_name or 'default'}")
    print(f"Run ID: {run_id_display}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Report to: {report_to}")
    print(f"==================================")

    # Load and filter dataset
    print(f"Loading dataset from {dataset_path}")
    data = load_json(dataset_path)

    # Get task for potential custom filtering
    task = get_task(task_name)

    # Helper to check if example is abstained
    def is_abstained(ex):
        return (ex.get("abstained", False)
                or ex.get("metadata", {}).get("abstained", False)
                or "</think>\n\n<abstain>" in ex.get("generation", ""))

    # Use task-specific filter if available, otherwise default logic
    if hasattr(task, 'filter_for_sft'):
        filtered_examples = task.filter_for_sft(
            data,
            include_abstained=include_abstained,
            include_wrong_valid_format=include_wrong_valid_format,
        )
        # Count categories for logging
        num_correct = sum(1 for ex in filtered_examples if ex.get("correct", False))
        num_abstained = sum(1 for ex in filtered_examples if is_abstained(ex))
        num_wrong_valid = len(filtered_examples) - num_correct - num_abstained
        print(f"Loaded {len(data)} examples, keeping {len(filtered_examples)} "
              f"({num_correct} correct, {num_abstained} abstained, {num_wrong_valid} wrong-valid-format)")
    elif include_abstained:
        # Default filtering (no include_wrong_valid_format support)
        filtered_examples = [ex for ex in data if ex.get("correct", False) or is_abstained(ex)]
        num_correct = sum(1 for ex in filtered_examples if ex.get("correct", False))
        num_abstained = sum(1 for ex in filtered_examples if is_abstained(ex))
        print(f"Loaded {len(data)} examples, keeping {len(filtered_examples)} ({num_correct} correct, {num_abstained} abstained)")
    else:
        filtered_examples = [ex for ex in data if ex.get("correct", False)]
        print(f"Loaded {len(data)} examples, {len(filtered_examples)} correct ({100*len(filtered_examples)/len(data):.1f}%)")

    if not filtered_examples:
        raise ValueError("No valid examples found in dataset!")

    # Downsample correct examples if requested
    if max_correct is not None:
        import random
        correct = [ex for ex in filtered_examples if ex.get("correct", False)]
        non_correct = [ex for ex in filtered_examples if not ex.get("correct", False)]
        if len(correct) > max_correct:
            rng = random.Random(42)
            correct = rng.sample(correct, max_correct)
            filtered_examples = correct + non_correct
            rng.shuffle(filtered_examples)
            print(f"Downsampled correct to {max_correct} → {len(filtered_examples)} total ({max_correct} correct, {len(non_correct)} non-correct)")

    # Upsample abstained examples
    if upsample_abstain > 1:
        abstain_examples = [ex for ex in filtered_examples if is_abstained(ex)]
        if abstain_examples:
            extra_copies = abstain_examples * (upsample_abstain - 1)
            filtered_examples = filtered_examples + extra_copies
            print(f"Upsampled {len(abstain_examples)} abstained examples {upsample_abstain}x → "
                  f"{len(abstain_examples) * upsample_abstain} copies (total: {len(filtered_examples)})")

    # Upsample hint-containing examples
    if upsample_hint > 1:
        hint_examples = [ex for ex in filtered_examples if "<request>" in ex.get("generation", "")]
        if hint_examples:
            extra_copies = hint_examples * (upsample_hint - 1)
            filtered_examples = filtered_examples + extra_copies
            print(f"Upsampled {len(hint_examples)} hint examples {upsample_hint}x → "
                  f"{len(hint_examples) * upsample_hint} copies (total: {len(filtered_examples)})")
        else:
            print(f"--upsample-hint={upsample_hint} specified but no hint examples found")

    # Load tokenizer
    print(f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Check if we need response masking
    mask_response_tokens = method.mask_response_tokens if method else False

    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Format data for SFT
    print("Formatting for SFT...")
    formatted = []

    if mask_response_tokens:
        # Pre-tokenize with response masking for clean boundaries
        from pipeline.core.utils import tokenize_with_response_mask

        print("  Using segmented tokenization for response masking")
        print("  Masking \\n<response>...</response>\\n spans")

        for ex in filtered_examples:
            # Apply chat template to prompt messages (excluding assistant prefix)
            messages = ex["prompt"]
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Completion is assistant prefix + generation
            completion = assistant_prefix + ex["generation"]

            # Tokenize prompt (all masked, completion_mask=0)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

            # Tokenize completion with response masking
            completion_tokens, response_mask = tokenize_with_response_mask(
                completion, tokenizer
            )

            # Combine: prompt (mask=0) + completion (mask from response_mask)
            input_ids = prompt_tokens + completion_tokens
            completion_mask = [0] * len(prompt_tokens) + response_mask

            formatted.append({
                "input_ids": input_ids,
                "completion_mask": completion_mask,
            })
    else:
        # Standard prompt/completion format (SFTTrainer handles tokenization)
        for ex in filtered_examples:
            messages = ex["prompt"]
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            completion = assistant_prefix + ex["generation"]

            formatted.append({
                "prompt": prompt,
                "completion": completion,
            })

    dataset = Dataset.from_list(formatted)

    train_dataset = dataset
    print(f"Train: {len(train_dataset)}")

    # Data collator - standard collator handles completion_mask
    data_collator = None

    # Training config. SFT keeps only the final weights: intermediate
    # checkpointing and a held-out eval split were dropped after every recorded
    # run turned out to consume just the final model.
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        bf16=bf16,
        report_to=report_to,
        run_name=experiment_name,
    )

    # Initialize wandb if enabled
    if report_to == "wandb":
        import wandb
        wandb.init(project=project_name, name=experiment_name, reinit=True)

    # Train
    print(f"Starting training: {base_model} -> {output_path}")
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved model to {output_path}")

    return output_path


def _guard_existing_run(
    run_dir: Path | None,
    checkpoints_dir: Path | None,
    output_path: Path | None,
    rollouts_dir: Path | None,
    *,
    overwrite: bool,
    continue_run: bool,
) -> None:
    """Refuse to start a fresh run on top of an existing one.

    Two ways a rerun silently destroys work. A finished run has had its
    checkpoints cleaned up but still holds model/ and rollouts/, so verl finds
    nothing to resume, trains from scratch, and the final conversion overwrites
    the old model/. An interrupted run still has checkpoints, so resume_mode=auto
    picks it up mid-flight under whatever hyperparameters this invocation passed.
    Neither is ever what a fresh run wants, so both require saying so explicitly.
    """
    import re
    import shutil

    ckpts = []
    if checkpoints_dir is not None and checkpoints_dir.is_dir():
        ckpts = sorted(
            (d for d in checkpoints_dir.iterdir()
             if d.is_dir() and re.fullmatch(r"global_step_\d+", d.name)),
            key=lambda d: int(d.name.rsplit("_", 1)[1]),
        )

    model_dirs = [
        d for d in (
            output_path,
            (run_dir / "best") if run_dir is not None else None,
            (run_dir / "last") if run_dir is not None else None,
        )
        if d is not None
    ]

    found = []
    for path in model_dirs:
        if path.is_symlink() or (path.is_dir() and any(path.iterdir())):
            found.append(f"model        {path}")
    if ckpts:
        found.append(f"{len(ckpts)} checkpoint(s)  {checkpoints_dir} (latest {ckpts[-1].name})")
    if rollouts_dir is not None and rollouts_dir.is_dir() and any(rollouts_dir.iterdir()):
        found.append(f"rollouts     {rollouts_dir}")

    if continue_run:
        if not ckpts:
            raise FileNotFoundError(
                f"--continue-run passed but no checkpoints to resume in {checkpoints_dir}.\n"
                f"  A completed run has its checkpoints cleaned up; only --overwrite "
                f"or a new --run-id can proceed."
            )
        print(f"Continuing existing run from {ckpts[-1].name}")
        return

    if not found:
        return

    if overwrite:
        print(f"--overwrite: clearing existing run at {run_dir}")
        for path in [checkpoints_dir, rollouts_dir, *model_dirs]:
            if path is None:
                continue
            if path.is_symlink():
                path.unlink()
                print(f"  removed {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"  removed {path}")
        return

    listing = "\n".join(f"    {item}" for item in found)
    raise FileExistsError(
        f"run directory already holds results: {run_dir}\n{listing}\n"
        f"  Training here would overwrite them. Pick a new --run-id, or pass\n"
        f"  --overwrite to discard them, or --continue-run to resume in place."
    )


def train_rl(
    task_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
    base_model: str | None = None,
    train_prompts_path: Path | None = None,
    val_prompts_path: Path | None = None,
    sft_model_path: Path | None = None,
    output_path: Path | None = None,
    reward_function_path: Path | None = None,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    learning_rate: float = 1e-6,
    total_steps: int = 400,
    kl_coef: float = 0.001,
    n_samples: int = 16,
    save_freq: int | None = None,
    test_freq: int | None = None,
    max_prompt_length: int = 2048,
    max_response_length: int = 2048,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    project_name: str | None = None,
    experiment_name: str | None = None,
    wandb: bool = True,
    resume_path: Path | None = None,
    cleanup_checkpoints: bool = True,
    keep_state: bool = False,
    reward_kwargs_overrides: dict | None = None,
    shuffle_seed: int | None = None,
    overwrite: bool = False,
    continue_run: bool = False,
    save_best: bool = False,
    best_metric: str = "auto",
    max_ckpt_to_keep: int | None = None,
) -> Path:
    """
    Train RL model using verl (GRPO algorithm).

    Args:
        task_name: Name of task
        method_name: Method name for auto-derived paths and reward config
        run_id: Run identifier for organizing outputs (default: "default")
        base_model: Base model for cold-start RL (mutually exclusive with sft_model_path)
        train_prompts_path: Path to RL train prompts parquet file
        val_prompts_path: Path to RL validation prompts parquet file
        sft_model_path: Path to SFT model to start from (mutually exclusive with base_model)
        output_path: Where to save RL model
        reward_function_path: Path to reward function (default: task's reward function)
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        learning_rate: Learning rate
        total_steps: Total training steps
        kl_coef: KL divergence coefficient
        n_samples: Number of samples per prompt
        save_freq: Checkpoint save frequency
        test_freq: Validation/logging frequency (default: same as save_freq)
        max_prompt_length: Maximum prompt length in tokens
        max_response_length: Maximum response length in tokens
        max_model_len: Maximum model context length (default: prompt + response length).
            Set higher than prompt + response for multi-turn hint mode.
        tensor_parallel_size: Tensor parallel size
        gpu_memory_utilization: GPU memory utilization for vLLM
        project_name: Wandb project name (default: {task}-rl)
        experiment_name: Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})
        wandb: Enable wandb logging
        resume_path: Path to resume from existing run (overrides output_path)
        cleanup_checkpoints: Delete checkpoints after training (default: True)
        keep_state: Keep the last optimizer state checkpoint after training (default: False)

    Returns:
        Path to trained model
    """
    import subprocess
    import os

    from pipeline.tasks import get_task

    # Get repo root (assumes we're running from repo root)
    repo_root = Path.cwd()

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Validate mutual exclusion
    if base_model is not None and sft_model_path is not None:
        raise ValueError("Cannot specify both --base-model and --sft-model")

    # Frequencies. With --save-best the periodic save is off by default: the only
    # checkpoints worth keeping are the ones validation picked, and an explicit
    # --save-freq still wins. test_freq must not inherit a disabled save_freq or
    # validation would switch itself off along with it.
    if save_freq is None:
        save_freq = -1 if save_best else 25
    if test_freq is None:
        test_freq = save_freq if save_freq > 0 else 25
    if save_best and test_freq <= 0:
        raise ValueError("--save-best needs validation enabled; --test-freq must be > 0")

    # verl prunes checkpoints by recency. Best-saves only ever happen at
    # increasing steps and the final save is later still, so the two most recent
    # survivors are exactly the best and the last -- but only if room for two.
    if max_ckpt_to_keep is None:
        max_ckpt_to_keep = 2 if save_best else 1
    elif save_best and max_ckpt_to_keep < 2:
        print("Note: --save-best needs room for the best and the final checkpoint; using --max-ckpt-to-keep 2")
        max_ckpt_to_keep = 2

    # Default paths from method
    if method is not None:
        if train_prompts_path is None:
            train_prompts_path = method.prompts_path(task_name, "rl_train")
        if val_prompts_path is None:
            candidate = method.prompts_path(task_name, "rl_val")
            if candidate.exists():
                val_prompts_path = candidate

    # Determine the actor model (base_model or sft_model_path)
    if base_model is not None:
        actor_model = base_model
    elif sft_model_path is not None:
        actor_model = str(sft_model_path)
    elif method is not None:
        actor_model = str(method.sft_model_path(task_name))
    else:
        raise ValueError("Either --base-model or --sft-model is required (or use --method)")

    # Derive run directory structure from method
    run_dir = None
    checkpoints_dir = None
    rollouts_dir = None
    if method is not None and output_path is None:
        method.ensure_rl_run_dir(task_name, run_id)
        run_dir = method.rl_run_dir(task_name, run_id)
        checkpoints_dir = method.rl_checkpoints_dir(task_name, run_id)
        rollouts_dir = method.rl_rollouts_dir(task_name, run_id)
        output_path = method.rl_model_path(task_name, run_id)
    elif output_path is not None:
        # Custom output path - derive subdirectories from it
        run_dir = output_path.parent if output_path.name == "model" else output_path
        checkpoints_dir = run_dir / "checkpoints"
        rollouts_dir = run_dir / "rollouts"
        output_path = run_dir / "model"

    if resume_path is None:
        _guard_existing_run(
            run_dir, checkpoints_dir, output_path, rollouts_dir,
            overwrite=overwrite, continue_run=continue_run,
        )

    # Validate required paths
    if train_prompts_path is None:
        raise ValueError("--train-prompts is required (or use --method)")
    # val_prompts_path is optional — training will skip validation if not provided
    if output_path is None or checkpoints_dir is None or rollouts_dir is None:
        raise ValueError("--output is required (or use --method)")

    # Convert all paths to absolute paths for Ray workers (they run in different working directories)
    train_prompts_path = Path(train_prompts_path).resolve()
    if val_prompts_path is not None:
        val_prompts_path = Path(val_prompts_path).resolve()
    checkpoints_dir = Path(checkpoints_dir).resolve()
    rollouts_dir = Path(rollouts_dir).resolve()
    output_path = Path(output_path).resolve()

    # A prompts file that does not exist used to be handed straight to verl,
    # which then failed much later with an opaque error. Only explicitly-passed
    # paths can be missing here: the auto-derived val path above is assigned
    # solely when it exists.
    for flag, prompts_path in (
        ("--train-prompts", train_prompts_path),
        ("--val-prompts", val_prompts_path),
    ):
        if prompts_path is None or prompts_path.exists():
            continue
        split = prompts_path.stem
        supported = get_task(task_name).supported_splits()
        if split not in supported:
            detail = (
                f"task '{task_name}' does not define the '{split}' split "
                f"(it has: {', '.join(supported)})"
            )
        else:
            detail = (
                f"run: python -m pipeline create_prompts --task {task_name} "
                f"--method {method.name if method else '<method>'} --split {split}"
            )
        raise FileNotFoundError(f"{flag} not found: {prompts_path}\n  {detail}")
    if not actor_model.startswith("/") and "/" in actor_model:
        # Relative path (not a HuggingFace model ID like "Qwen/Qwen2.5-1.5B")
        actor_model = str(Path(actor_model).resolve())

    # Get reward function name and other config from method
    reward_function_name = "compute_score"
    reward_kwargs = {}
    template_content = None
    allow_hint = False
    interaction_name = None
    max_turns = 6
    max_hints = None
    if method is not None:
        reward_function_name = method.reward_function
        reward_kwargs = {**method.reward_kwargs}
        allow_hint = method.allow_hint
        interaction_name = method.interaction_name or (f"{task_name}_{method.name}" if method.multi_turn else None)
        max_turns = method.max_turns
        max_hints = method.max_hints
        template_content = method.load_template(task_name, "rl")

    # Apply CLI overrides to reward kwargs
    if reward_kwargs_overrides:
        reward_kwargs.update(reward_kwargs_overrides)

    # Handle resume path
    if resume_path is not None:
        # Resume uses the checkpoint directory structure
        checkpoints_dir = resume_path
        run_dir = resume_path.parent
        rollouts_dir = run_dir / "rollouts"
        output_path = run_dir / "model"
        if experiment_name is None:
            experiment_name = run_dir.name
        print(f"Resuming from: {resume_path}")

    # Default reward function path
    # Try full task name first, then fall back to base name (e.g., countdown_abstention -> countdown)
    if reward_function_path is None:
        reward_function_path = repo_root / f"verl/recipe/{task_name}/reward_function.py"
        if not reward_function_path.exists():
            # Try base task name (for variants like countdown_abstention)
            base_task_name = task_name.split("_")[0]
            reward_function_path = repo_root / f"verl/recipe/{base_task_name}/reward_function.py"
            if not reward_function_path.exists():
                raise FileNotFoundError(
                    f"Reward function not found at verl/recipe/{task_name}/ or verl/recipe/{base_task_name}/. "
                    f"Please provide --reward-function."
                )

    # Create output directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate project name if not provided: {task}-rl-{model_short_name}
    if project_name is None:
        project_name = f"{task_name}-rl-{_model_project_tag(actor_model)}"

    # Generate experiment name if not provided
    # Format: {method}-{run_id}-{YYYYMMDD}
    if experiment_name is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        method_str = method_name if method_name else "default"
        run_id_str = run_id if run_id else "default"
        experiment_name = f"{method_str}-{run_id_str}-{date_str}"

    # Get task's system message and assistant prefix for runtime template application
    task = get_task(task_name)
    system_message = getattr(task, "system_message", None)
    assistant_prefix = getattr(task, "assistant_prefix", None)

    # Logger config
    logger_config = "['wandb','console']" if wandb else "['console']"

    print(f"=== RL Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"Method: {method_name or 'default'}")
    print(f"Run ID: {run_id or 'default'}")
    print(f"Actor Model: {actor_model}")
    print(f"Train Prompts: {train_prompts_path}")
    print(f"Val Prompts: {val_prompts_path or 'None (no validation)'}")
    print(f"Reward Function: {reward_function_path}:{reward_function_name}")
    print(f"Reward Kwargs: {reward_kwargs}")
    print(f"Multi-Turn: {allow_hint}")
    if interaction_name:
        print(f"Interaction: {interaction_name}")
    print(f"Max Turns: {max_turns}")
    print(f"Run Directory: {run_dir}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Rollouts: {rollouts_dir}")
    print(f"Output Model: {output_path}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Total Steps: {total_steps}")
    print(f"Wandb: {wandb}")
    if template_content:
        print(f"Template: {method.template_variant}/rl.txt")
    print(f"=================================")

    # Build verl command
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"hydra.run.dir={checkpoints_dir}",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_prompts_path}",
        f"data.val_files={val_prompts_path or train_prompts_path}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={val_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"custom_reward_function.path={reward_function_path}",
        f"custom_reward_function.name={reward_function_name}",
        f"actor_rollout_ref.model.path={actor_model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size=8",
        f"actor_rollout_ref.rollout.n={n_samples}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=4",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=4",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        f"trainer.logger={logger_config}",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={checkpoints_dir}",
        f"trainer.n_gpus_per_node={tensor_parallel_size}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.resume_mode={'auto' if (continue_run or resume_path is not None) else 'disable'}",
        f"trainer.max_actor_ckpt_to_keep={max_ckpt_to_keep}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.rollout_data_dir={rollouts_dir}",
    ]

    # Checkpoint selection (vendored verl extensions). save_last guarantees the
    # final step is on disk regardless of the periodic save cadence.
    cmd.append("+trainer.save_last=True")
    if save_best:
        cmd.append("+trainer.save_best=True")
        cmd.append(f"+trainer.best_ckpt_metric={best_metric}")

    # Shuffle seed for training data ordering
    if shuffle_seed is not None:
        cmd.append(f"+data.seed={shuffle_seed}")

    # Add runtime template config if method is specified
    if template_content is not None:
        # Escape the template for shell/hydra (replace newlines, quotes)
        # Use + prefix to add new config keys (they don't exist in base config)
        escaped_template = template_content.replace("\n", "\\n").replace('"', '\\"')
        cmd.append(f'+data.runtime_template="{escaped_template}"')
        if system_message:
            escaped_system = system_message.replace("\n", "\\n").replace('"', '\\"')
            cmd.append(f'+data.runtime_system_message="{escaped_system}"')
        if assistant_prefix:
            escaped_prefix = assistant_prefix.replace("\n", "\\n").replace('"', '\\"')
            cmd.append(f'+data.runtime_assistant_prefix="{escaped_prefix}"')

    # vLLM rollout (verl's default backend and default sync mode; the SGLang
    # backend and the async rollout modes were never used by any recorded run).
    # Add allow_hint flag for multi-turn hint generation
    if allow_hint:
        cmd.append("allow_hint=True")
    # Add max_hints limit for rollout
    if max_hints is not None:
        cmd.append(f"+actor_rollout_ref.rollout.max_hints={max_hints}")
    # Add max_turns for loop bound (model gets extra turns after hints exhausted)
    cmd.append(f"+actor_rollout_ref.rollout.max_turns={max_turns}")

    # Add custom stop strings for rollout if specified in method config
    if method is not None and method.stop_strings is not None:
        # Pass as comma-separated list; verl rollout will parse it
        stop_strings_str = ",".join(method.stop_strings)
        cmd.append(f'+actor_rollout_ref.rollout.stop_strings="{stop_strings_str}"')

    # Add reward kwargs if specified in method config
    # Use + prefix to add new config keys
    if reward_kwargs:
        for key, value in reward_kwargs.items():
            cmd.append(f"+custom_reward_function.reward_kwargs.{key}={value}")

    # Set reward manager type if non-default
    if method is not None and method.reward_manager != "naive":
        cmd.append(f"reward_model.reward_manager={method.reward_manager}")
        # Also pass reward_kwargs to reward_model.reward_kwargs so the manager
        # constructor receives them (e.g., AdaptiveRewardManager needs beta, delta, etc.)
        if reward_kwargs:
            for key, value in reward_kwargs.items():
                cmd.append(f"+reward_model.reward_kwargs.{key}={value}")

    # Set environment variables
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

    print(f"Running verl training...")
    print(f"Command: {' '.join(cmd[:5])}...")

    # Run training
    result = subprocess.run(cmd, env=env, cwd=str(repo_root))

    if result.returncode != 0:
        raise RuntimeError(f"RL training failed with return code {result.returncode}")

    print(f"RL training complete. Checkpoints saved to {checkpoints_dir}")

    # Convert checkpoints to HuggingFace format. last/ is whatever the run
    # ended on and is always written; best/ appears when --save-best recorded
    # a winning step.
    import re
    checkpoint_dirs = [
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and re.match(r"global_step_\d+", d.name)
    ]

    hf_model_path = output_path
    if checkpoint_dirs:
        def get_step(d):
            match = re.search(r"global_step_(\d+)", d.name)
            return int(match.group(1)) if match else 0

        import json
        import shutil

        best_step = None
        best_record = checkpoints_dir / "best_checkpoint.json"
        if best_record.exists():
            try:
                best_step = int(json.loads(best_record.read_text())["step"])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"Warning: could not read {best_record}: {exc}")

        last_ckpt = max(checkpoint_dirs, key=get_step)
        best_ckpt = None
        if best_step is not None:
            best_ckpt = next((d for d in checkpoint_dirs if get_step(d) == best_step), None)
            if best_ckpt is None:
                print(
                    f"Warning: best checkpoint global_step_{best_step} was recorded but is "
                    f"no longer on disk; writing last/ only"
                )
            else:
                print(f"Best checkpoint by validation: {best_ckpt.name}")

        run_root = output_path.parent
        converted: dict[str, Path] = {}

        def _clear(path):
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

        def _convert(label, ckpt):
            actor_path = ckpt / "actor"
            if not actor_path.exists():
                print(f"Warning: actor directory not found in {ckpt}; skipping {label}/")
                return
            dest = run_root / label
            _clear(dest)
            print(f"\nConverting {label} checkpoint: {ckpt.name}")
            converted[label] = Path(convert_checkpoint(actor_path, output_path=dest))
            print(f"HuggingFace model saved to: {converted[label]}")

        if best_ckpt is not None:
            _convert("best", best_ckpt)

        if best_ckpt is not None and best_ckpt == last_ckpt and "best" in converted:
            # The final step also won; copying beats merging the same shards twice.
            dest = run_root / "last"
            _clear(dest)
            shutil.copytree(converted["best"], dest)
            converted["last"] = dest
            print(f"last/ is the same checkpoint as best/ ({last_ckpt.name}); copied")
        else:
            _convert("last", last_ckpt)

        # model/ is what method.rl_model_path resolves to, so point it at the
        # checkpoint the run should be judged on and leave the other beside it.
        primary = "best" if "best" in converted else ("last" if "last" in converted else None)
        if primary is not None:
            _clear(output_path)
            output_path.symlink_to(primary, target_is_directory=True)
            hf_model_path = output_path
            print(f"model/ -> {primary}/")

    # Cleanup checkpoints if requested (rollouts are preserved for analysis)
    if cleanup_checkpoints and not keep_state:
        import shutil
        print("Cleaning up checkpoints...")
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            print(f"  Removed: {checkpoints_dir}")
        print("Cleanup complete.")
    elif cleanup_checkpoints and keep_state:
        # Keep only the last checkpoint (with optimizer state), remove the rest
        import shutil
        import re as _re
        if checkpoints_dir.exists():
            ckpt_dirs = [
                d for d in checkpoints_dir.iterdir()
                if d.is_dir() and _re.match(r"global_step_\d+", d.name)
            ]
            if len(ckpt_dirs) > 1:
                def _get_step(d):
                    m = _re.search(r"global_step_(\d+)", d.name)
                    return int(m.group(1)) if m else 0
                ckpt_dirs.sort(key=_get_step)
                for d in ckpt_dirs[:-1]:
                    shutil.rmtree(d)
                    print(f"  Removed old checkpoint: {d.name}")
            if ckpt_dirs:
                print(f"  Kept last checkpoint: {ckpt_dirs[-1].name}")

    return hf_model_path




def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path | None = None,
    backend: str = "fsdp",
) -> Path:
    """
    Convert FSDP/Megatron checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint (e.g., .../global_step_100/actor)
        output_path: Where to save HF model (default: models/rl/<model>_step<N>)
        backend: Checkpoint backend ("fsdp" or "megatron")

    Returns:
        Path to converted HuggingFace model
    """
    import subprocess
    import re
    import json

    # Find the huggingface config directory
    hf_config_path = checkpoint_path / "huggingface"
    if not hf_config_path.exists():
        raise FileNotFoundError(
            f"HuggingFace config not found at {hf_config_path}. "
            f"Expected structure: {checkpoint_path}/huggingface/config.json"
        )

    # Default output path: models/rl/<model>_step<N>
    if output_path is None:
        # Extract step number from path (e.g., global_step_100)
        step_match = re.search(r"global_step_(\d+)", str(checkpoint_path))
        step_num = step_match.group(1) if step_match else "unknown"

        # Get model type from config
        config_file = hf_config_path / "config.json"
        with open(config_file) as f:
            config = json.load(f)
        model_type = config.get("model_type", "model")

        # Find models/rl directory by walking up from checkpoint
        # Structure: models/rl/<run_name>/global_step_X/actor
        rl_dir = checkpoint_path.parent.parent.parent
        if rl_dir.name != "rl":
            # Fallback: just use parent of checkpoint
            rl_dir = checkpoint_path.parent.parent

        output_path = rl_dir / f"{model_type}_rl_step{step_num}"

    # Get repo root for verl scripts
    repo_root = Path.cwd()
    merger_script = repo_root / "verl/scripts/legacy_model_merger.py"
    if not merger_script.exists():
        raise FileNotFoundError(f"Model merger script not found at {merger_script}")

    print(f"=== Converting Checkpoint ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Backend: {backend}")
    print(f"Output: {output_path}")
    print(f"=============================")

    # Build command
    cmd = [
        "python3", str(merger_script),
        "merge",
        "--backend", backend,
        "--local_dir", str(checkpoint_path),
        "--hf_model_path", str(hf_config_path),
        "--target_dir", str(output_path),
    ]

    # Run conversion
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode != 0:
        raise RuntimeError(f"Checkpoint conversion failed with return code {result.returncode}")

    print(f"Conversion complete. HuggingFace model saved to {output_path}")
    return output_path
