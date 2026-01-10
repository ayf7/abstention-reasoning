"""
Shared SFT training wrapper using TRL.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run_sft_training(
    train_path: Path,
    cfg: DictConfig,
    filter_fn=None,
    preprocess_fn=None,
    val_split_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Run SFT training on a JSONL dataset.

    Args:
        train_path: Path to train.jsonl
        cfg: SFT config with keys:
            - model_name: str
            - output_dir: str
            - num_epochs: int
            - batch_size: int
            - learning_rate: float
            - eval_steps: int
            - save_steps: int
        filter_fn: Optional function to filter examples (returns bool)
        preprocess_fn: Optional function to preprocess examples
        val_split_ratio: Fraction for validation split
        seed: Random seed for split

    Returns:
        SFTTrainer instance
    """
    # Lazy imports to avoid CUDA init before VLLM
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"Running SFT on {train_path}...")
    print(f"Model: {cfg.model_name}")
    print(f"Output: {cfg.output_dir}")

    # Load dataset
    dataset = load_dataset("json", data_files=str(train_path))["train"]

    # Apply filter if provided
    if filter_fn is not None:
        original_len = len(dataset)
        dataset = dataset.filter(filter_fn)
        print(f"Filtered: {len(dataset)}/{original_len} examples")

    if len(dataset) == 0:
        raise ValueError("No examples after filtering. Check filter_fn or run create_generations() first.")

    # Apply preprocessing if provided
    if preprocess_fn is not None:
        # Remove old columns to avoid TRL detecting prompt/completion format
        columns_to_remove = [c for c in dataset.column_names if c != "text"]
        dataset = dataset.map(preprocess_fn, remove_columns=columns_to_remove)

    # Print samples
    dry_run = getattr(cfg, "dry_run", False)
    num_samples = 3 if dry_run else 1

    if len(dataset) > 0:
        print("\n" + "=" * 80)
        print(f"Sample training example(s): showing {num_samples}")
        print("=" * 80)
        # Show full text in dry_run, truncate otherwise
        max_len = None if dry_run else 2000
        for i in range(min(num_samples, len(dataset))):
            example = dataset[i]
            if i > 0:
                print("\n" + "-" * 40 + f" Example {i+1} " + "-" * 40)
            if "text" in example:
                text = example["text"]
                if max_len and len(text) > max_len:
                    print(f"{text[:max_len]}...")
                else:
                    print(text)
            elif "prompt" in example:
                print(f"PROMPT:\n{example['prompt'][:500]}..." if len(str(example['prompt'])) > 500 else f"PROMPT:\n{example['prompt']}")
                if "completion" in example:
                    print(f"\nCOMPLETION:\n{example['completion'][:500]}..." if len(str(example['completion'])) > 500 else f"\nCOMPLETION:\n{example['completion']}")
        print("=" * 80 + "\n")

    if dry_run:
        print("DRY RUN: Exiting without training.")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Would train for {cfg.num_epochs} epochs")
        print(f"  Effective batch size: {cfg.batch_size * getattr(cfg, 'gradient_accumulation_steps', 1)}")
        return None

    # Split for validation
    split_dataset = dataset.train_test_split(test_size=val_split_ratio, seed=seed)

    # Setup training args
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=getattr(cfg, "weight_decay", 0.0),
        adam_beta1=getattr(cfg, "adam_beta1", 0.9),
        adam_beta2=getattr(cfg, "adam_beta2", 0.999),
        max_grad_norm=getattr(cfg, "max_grad_norm", 1.0),
        warmup_ratio=getattr(cfg, "warmup_ratio", 0.0),
        lr_scheduler_type=getattr(cfg, "lr_scheduler_type", "linear"),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        logging_steps=10,
        report_to="wandb",
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=cfg.model_name,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Training complete. Model saved to {cfg.output_dir}")
    return trainer
