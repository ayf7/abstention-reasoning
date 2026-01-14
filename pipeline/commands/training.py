"""
Training commands - SFT, RL, classifier training, and checkpoint conversion.
"""

from pathlib import Path

from pipeline.core.io import load_json
from pipeline.tasks import get_task


def train_sft(
    task_name: str,
    dataset_path: Path,
    output_path: Path,
    base_model: str,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 4096,
    eval_split: float = 0.05,
    save_steps: int = 100,
    logging_steps: int = 10,
    bf16: bool = True,
    report_to: str = "none",
) -> Path:
    """
    Train an SFT model on generated dataset.

    Filters to correct examples, formats as prompt/completion pairs,
    and trains using TRL's SFTTrainer.

    Args:
        task_name: Name of task
        dataset_path: Path to generated dataset (from generate command)
        output_path: Where to save trained model
        base_model: Base model to fine-tune
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_length: Maximum sequence length
        eval_split: Fraction of data for evaluation
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        bf16: Use bfloat16 training
        report_to: Reporting integration ("none", "wandb", etc.)

    Returns:
        Path to trained model
    """
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer

    # Load and filter dataset
    print(f"Loading dataset from {dataset_path}")
    data = load_json(dataset_path)
    correct_examples = [ex for ex in data if ex.get("correct", False)]
    print(f"Loaded {len(data)} examples, {len(correct_examples)} correct ({100*len(correct_examples)/len(data):.1f}%)")

    if not correct_examples:
        raise ValueError("No correct examples found in dataset!")

    # Load tokenizer
    print(f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Format as prompt/completion pairs
    print("Formatting for SFT...")
    formatted = []
    for ex in correct_examples:
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

        formatted.append({
            "prompt": prompt,
            "completion": completion,
        })

    dataset = Dataset.from_list(formatted)

    # Train/eval split
    split = dataset.train_test_split(test_size=eval_split, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_total_limit=3,
        bf16=bf16,
        report_to=report_to,
    )

    # Train
    print(f"Starting training: {base_model} -> {output_path}")
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    trainer.train()

    # Save final model
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved model to {output_path}")

    return output_path


def train_rl(
    task_name: str,
    train_prompts_path: Path,
    val_prompts_path: Path,
    sft_model_path: Path,
    output_path: Path,
    reward_function_path: Path | None = None,
    train_batch_size: int = 256,
    val_batch_size: int = 256,
    learning_rate: float = 1e-6,
    total_steps: int = 100,
    kl_coef: float = 0.001,
    n_samples: int = 4,
    save_freq: int = 10,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.4,
    project_name: str = "countdown-rl",
    experiment_name: str | None = None,
    wandb: bool = False,
    resume_path: Path | None = None,
) -> Path:
    """
    Train RL model using verl (GRPO algorithm).

    Args:
        task_name: Name of task
        train_prompts_path: Path to RL train prompts parquet file
        val_prompts_path: Path to RL validation prompts parquet file
        sft_model_path: Path to SFT model to start from
        output_path: Where to save RL model
        reward_function_path: Path to reward function (default: task's reward function)
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        learning_rate: Learning rate
        total_steps: Total training steps
        kl_coef: KL divergence coefficient
        n_samples: Number of samples per prompt
        save_freq: Checkpoint save frequency
        tensor_parallel_size: Tensor parallel size
        gpu_memory_utilization: GPU memory utilization for vLLM
        project_name: Wandb project name
        experiment_name: Custom experiment name (default: auto-generated)
        wandb: Enable wandb logging
        resume_path: Path to resume from existing run (overrides output_path)

    Returns:
        Path to trained model
    """
    import subprocess
    import os

    # Get repo root (assumes we're running from repo root)
    repo_root = Path.cwd()

    # Handle resume path
    if resume_path is not None:
        output_path = resume_path
        if experiment_name is None:
            experiment_name = resume_path.name
        print(f"Resuming from: {resume_path}")

    # Default reward function path
    if reward_function_path is None:
        reward_function_path = repo_root / f"verl/recipe/{task_name}/reward_function.py"
        if not reward_function_path.exists():
            raise FileNotFoundError(
                f"Reward function not found at {reward_function_path}. "
                f"Please provide --reward-function-path."
            )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"{task_name}_rl_{learning_rate}_{train_batch_size}"

    # Logger config
    logger_config = "['wandb','console']" if wandb else "['console']"

    print(f"=== RL Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"SFT Model: {sft_model_path}")
    print(f"Train Prompts: {train_prompts_path}")
    print(f"Val Prompts: {val_prompts_path}")
    print(f"Reward Function: {reward_function_path}")
    print(f"Output: {output_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Total Steps: {total_steps}")
    print(f"Wandb: {wandb}")
    print(f"=================================")

    # Build verl command
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"hydra.run.dir={output_path}",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_prompts_path}",
        f"data.val_files={val_prompts_path}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={val_batch_size}",
        "data.max_prompt_length=256",
        "data.max_response_length=2048",
        f"custom_reward_function.path={reward_function_path}",
        "custom_reward_function.name=compute_score",
        f"actor_rollout_ref.model.path={sft_model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size=16",
        f"actor_rollout_ref.rollout.n={n_samples}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=4",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=4",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        f"trainer.logger={logger_config}",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={output_path}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={save_freq}",
        "trainer.resume_mode=auto",
        "trainer.max_actor_ckpt_to_keep=2",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.total_training_steps={total_steps}",
    ]

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

    print(f"RL training complete. Checkpoints saved to {output_path}")

    # Find and convert the final checkpoint
    import re
    checkpoint_dirs = [
        d for d in output_path.iterdir()
        if d.is_dir() and re.match(r"global_step_\d+", d.name)
    ]

    if checkpoint_dirs:
        # Sort by step number and get the latest
        def get_step(d):
            match = re.search(r"global_step_(\d+)", d.name)
            return int(match.group(1)) if match else 0

        latest_checkpoint = max(checkpoint_dirs, key=get_step)
        actor_path = latest_checkpoint / "actor"

        if actor_path.exists():
            print(f"\nConverting final checkpoint: {latest_checkpoint.name}")
            hf_model_path = convert_checkpoint(actor_path)
            print(f"HuggingFace model saved to: {hf_model_path}")
            return hf_model_path
        else:
            print(f"Warning: actor directory not found in {latest_checkpoint}")

    return output_path


def train_classifier(
    task_name: str,
    dataset_path: Path,
    output_path: Path,
    base_model: str,
    mode: str = "binary",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    eval_split: float = 0.1,
) -> Path:
    """
    Train a binary classifier to predict if model generations are correct.

    The classifier takes (prompt + generation) as input and predicts correctness.

    Args:
        task_name: Name of task
        dataset_path: Path to generated dataset (from generate command)
        output_path: Where to save trained classifier
        base_model: Base model for classification
        mode: Classification mode ("binary" or "three_class")
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        eval_split: Fraction of data for evaluation

    Returns:
        Path to trained classifier
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Validate mode
    if mode != "binary":
        raise NotImplementedError(f"Mode '{mode}' not yet implemented. Only 'binary' is supported.")

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    data = load_json(dataset_path)
    print(f"Loaded {len(data)} examples")

    # Prepare classification data
    print("Preparing classification data...")
    formatted = []
    for ex in data:
        # Format prompt as string
        if isinstance(ex["prompt"], list):
            # Chat format - concatenate messages
            prompt_text = ""
            for msg in ex["prompt"]:
                prompt_text += f"{msg['role']}: {msg['content']}\n"
        else:
            prompt_text = ex["prompt"]

        # Input is prompt + generation
        text = prompt_text + ex["generation"]

        # Label is correctness
        label = 1 if ex.get("correct", False) else 0

        formatted.append({"text": text, "label": label})

    # Count label distribution
    from collections import Counter
    label_dist = Counter(ex["label"] for ex in formatted)
    print(f"Label distribution: {dict(label_dist)}")

    dataset = Dataset.from_list(formatted)

    # Train/eval split
    split = dataset.train_test_split(test_size=eval_split, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Load tokenizer and model
    print(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_dataset = split["train"].map(tokenize_fn, batched=True)
    eval_dataset = split["test"].map(tokenize_fn, batched=True)

    # Training arguments
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        bf16=True,
    )

    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = (preds == labels).mean()

        # Compute precision, recall, F1 for positive class
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # Train
    print(f"Starting classifier training: {base_model} -> {output_path}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final model
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved classifier to {output_path}")

    return output_path


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
