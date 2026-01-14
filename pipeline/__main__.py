"""
CLI entrypoint for the pipeline.

Usage:
    python -m pipeline <command> [options]

Commands:
    list_tasks                  List available tasks
    create_primitives           Generate raw puzzle data
    create_prompts              Create prompts from primitives (split logic defined by task)
    generate                    Run model on prompts to create dataset
    train_sft                   Train SFT model on generated dataset
    train_rl                    Train RL model using verl (GRPO)
    train_classifier            Train binary classifier for correctness prediction
    convert_checkpoint          Convert FSDP/Megatron checkpoint to HuggingFace format
    evaluate                    Evaluate model and compute metrics

Examples:
    # List available tasks
    python -m pipeline list_tasks

    # Generate primitives
    python -m pipeline create_primitives --task countdown --output artifacts/countdown/primitives.json

    # Create prompts for all splits at once
    python -m pipeline create_prompts --task countdown \\
        --primitives artifacts/countdown/primitives.json \\
        --output artifacts/countdown/prompts

    # Or create prompts for a single split
    python -m pipeline create_prompts --task countdown \\
        --primitives artifacts/countdown/primitives.json \\
        --output artifacts/countdown/prompts/eval.json \\
        --split eval

    # Generate dataset
    python -m pipeline generate --task countdown \\
        --prompts artifacts/countdown/prompts/sft.json \\
        --output artifacts/countdown/datasets/sft_qwen3-14b.json \\
        --model Qwen/Qwen3-14B

    # Train SFT model
    python -m pipeline train_sft --task countdown \\
        --dataset artifacts/countdown/datasets/sft_qwen3-14b.json \\
        --output artifacts/countdown/models/sft \\
        --base-model Qwen/Qwen2.5-1.5B

    # Train RL model
    python -m pipeline train_rl --task countdown \\
        --prompts artifacts/countdown/prompts/rl.parquet \\
        --sft-model artifacts/countdown/models/sft \\
        --output artifacts/countdown/models/rl

    # Train classifier
    python -m pipeline train_classifier --task countdown \\
        --dataset artifacts/countdown/datasets/classifier_sft.json \\
        --output artifacts/countdown/models/classifier \\
        --base-model Qwen/Qwen2.5-1.5B

    # Convert FSDP checkpoint to HuggingFace format
    python -m pipeline convert_checkpoint \\
        --checkpoint artifacts/countdown/models/rl/2026-01-11_.../global_step_100/actor

    # Evaluate
    python -m pipeline evaluate --task countdown \\
        --prompts artifacts/countdown/prompts/eval.json \\
        --output artifacts/countdown/results/eval_sft.json \\
        --model artifacts/countdown/models/sft
"""

import argparse
from pathlib import Path

from pipeline.tasks import list_tasks, get_task
from pipeline import commands


def cmd_list_tasks(args):
    """List available tasks."""
    tasks = list_tasks()
    print("Available tasks:")
    for task in tasks:
        print(f"  - {task}")


def cmd_create_primitives(args):
    """Create primitives."""
    commands.create_primitives(
        task_name=args.task,
        output_path=Path(args.output),
        num_puzzles=args.num_puzzles,
        seed=args.seed,
    )


def cmd_create_prompts(args):
    """Create prompts."""
    # Template path is optional - defaults to task's template for this split
    template_path = Path(args.template) if args.template else None

    commands.create_prompts(
        task_name=args.task,
        primitives_path=Path(args.primitives),
        output_dir=Path(args.output),
        split_name=args.split,
        template_path=template_path,
        seed=args.seed,
        include_assistant_prefix=not args.no_assistant_prefix,
    )


def cmd_generate(args):
    """Generate dataset."""
    commands.generate(
        task_name=args.task,
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def cmd_evaluate(args):
    """Evaluate model."""
    commands.evaluate(
        task_name=args.task,
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def cmd_train_sft(args):
    """Train SFT model."""
    commands.train_sft(
        task_name=args.task,
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        eval_split=args.eval_split,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=not args.no_bf16,
        report_to=args.report_to,
    )


def cmd_train_rl(args):
    """Train RL model using verl."""
    reward_function_path = Path(args.reward_function) if args.reward_function else None
    resume_path = Path(args.resume) if args.resume else None
    commands.train_rl(
        task_name=args.task,
        train_prompts_path=Path(args.train_prompts),
        val_prompts_path=Path(args.val_prompts),
        sft_model_path=Path(args.sft_model),
        output_path=Path(args.output),
        reward_function_path=reward_function_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        kl_coef=args.kl_coef,
        n_samples=args.n_samples,
        save_freq=args.save_freq,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        wandb=args.wandb,
        resume_path=resume_path,
    )


def cmd_train_classifier(args):
    """Train binary classifier."""
    commands.train_classifier(
        task_name=args.task,
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        base_model=args.base_model,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        eval_split=args.eval_split,
    )


def cmd_convert_checkpoint(args):
    """Convert FSDP/Megatron checkpoint to HuggingFace format."""
    output_path = Path(args.output) if args.output else None
    commands.convert_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        output_path=output_path,
        backend=args.backend,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Data pipeline for abstention reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list_tasks
    p = subparsers.add_parser("list_tasks", help="List available tasks")
    p.set_defaults(func=cmd_list_tasks)

    # create_primitives
    p = subparsers.add_parser("create_primitives", help="Generate raw puzzle data")
    p.add_argument("--task", required=True, help="Task name (e.g., countdown)")
    p.add_argument("--output", required=True, help="Output path for primitives.json")
    p.add_argument("--num-puzzles", type=int, default=3100, help="Number of puzzles")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.set_defaults(func=cmd_create_primitives)

    # create_prompts
    p = subparsers.add_parser("create_prompts", help="Create prompts from primitives")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--primitives", required=True, help="Path to primitives.json")
    p.add_argument("--output", required=True, help="Output directory (for --split all) or file path")
    p.add_argument("--split", default="all", help="Split name (sft, rl, classifier, eval, or 'all')")
    p.add_argument("--template", help="Path to template file (default: task's template for split)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    p.add_argument("--no-assistant-prefix", action="store_true", help="Don't include assistant prefix")
    p.set_defaults(func=cmd_create_prompts)

    # generate
    p = subparsers.add_parser("generate", help="Run model on prompts")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--prompts", required=True, help="Path to prompts file")
    p.add_argument("--output", required=True, help="Output path for dataset")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.set_defaults(func=cmd_generate)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate model")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--prompts", required=True, help="Path to eval prompts")
    p.add_argument("--output", required=True, help="Output path for results")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature (0 for greedy)")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.set_defaults(func=cmd_evaluate)

    # train_sft
    p = subparsers.add_parser("train_sft", help="Train SFT model on generated dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--dataset", required=True, help="Path to generated dataset")
    p.add_argument("--output", required=True, help="Output path for trained model")
    p.add_argument("--base-model", required=True, help="Base model to fine-tune")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length")
    p.add_argument("--eval-split", type=float, default=0.05, help="Fraction of data for evaluation")
    p.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    p.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 training")
    p.add_argument("--report-to", default="none", help="Reporting integration (none, wandb)")
    p.set_defaults(func=cmd_train_sft)

    # train_rl
    p = subparsers.add_parser("train_rl", help="Train RL model using verl (GRPO)")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--train-prompts", required=True, help="Path to RL train prompts parquet file")
    p.add_argument("--val-prompts", required=True, help="Path to RL validation prompts parquet file")
    p.add_argument("--sft-model", required=True, help="Path to SFT model")
    p.add_argument("--output", required=True, help="Output path for RL model")
    p.add_argument("--reward-function", help="Path to reward function (default: task's reward function)")
    p.add_argument("--train-batch-size", type=int, default=256, help="Training batch size")
    p.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--total-steps", type=int, default=100, help="Total training steps")
    p.add_argument("--kl-coef", type=float, default=0.001, help="KL divergence coefficient")
    p.add_argument("--n-samples", type=int, default=4, help="Number of samples per prompt")
    p.add_argument("--save-freq", type=int, default=10, help="Checkpoint save frequency")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="GPU memory utilization")
    p.add_argument("--project-name", default="countdown-rl", help="Wandb project name")
    p.add_argument("--experiment-name", help="Custom experiment name (default: auto-generated)")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--resume", help="Path to resume from existing run")
    p.set_defaults(func=cmd_train_rl)

    # train_classifier
    p = subparsers.add_parser("train_classifier", help="Train binary classifier for correctness prediction")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--dataset", required=True, help="Path to generated dataset")
    p.add_argument("--output", required=True, help="Output path for classifier")
    p.add_argument("--base-model", required=True, help="Base model for classification")
    p.add_argument("--mode", default="binary", choices=["binary", "three_class"], help="Classification mode")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    p.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    p.add_argument("--eval-split", type=float, default=0.1, help="Fraction of data for evaluation")
    p.set_defaults(func=cmd_train_classifier)

    # convert_checkpoint
    p = subparsers.add_parser("convert_checkpoint", help="Convert FSDP/Megatron checkpoint to HuggingFace format")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (e.g., .../global_step_100/actor)")
    p.add_argument("--output", help="Output path for HF model (default: {checkpoint}_hf)")
    p.add_argument("--backend", default="fsdp", choices=["fsdp", "megatron"], help="Checkpoint backend")
    p.set_defaults(func=cmd_convert_checkpoint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
