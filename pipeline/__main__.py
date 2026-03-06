"""
CLI entrypoint for the pipeline.

Usage:
    python -m pipeline <command> [options]

Commands:
    list_tasks                  List available tasks
    list_methods                List available methods for a task
    create_primitives           Generate raw puzzle data
    create_prompts              Create prompts from primitives
    generate                    Run model on prompts to create dataset
    train_sft                   Train SFT model on generated dataset
    train_rl                    Train RL model using verl (GRPO)
    train_classifier            Train binary classifier for correctness prediction
    convert_checkpoint          Convert FSDP/Megatron checkpoint to HuggingFace format
    evaluate                    Evaluate model and compute metrics
    evaluate_classifier         Evaluate binary classifier (confusion matrix, precision/recall/F1)

Examples:
    # List available tasks and methods
    python -m pipeline list_tasks
    python -m pipeline list_methods --task countdown

    # Full workflow with --method (auto-derived paths)
    python -m pipeline create_primitives --task countdown --num-puzzles 5000
    python -m pipeline create_prompts --task countdown --method simple_abstention
    python -m pipeline generate --task countdown --method simple_abstention --model Qwen/Qwen3-14B
    python -m pipeline train_sft --task countdown --method simple_abstention --base-model Qwen/Qwen2.5-3B
    python -m pipeline train_rl --task countdown --method simple_abstention
    python -m pipeline evaluate --task countdown --method simple_abstention --model sft
    python -m pipeline evaluate --task countdown --method simple_abstention --model rl

    # Artifacts are organized by method:
    # artifacts/countdown/primitives.json           (shared)
    # artifacts/countdown/simple_abstention/        (method-specific)
    #   prompts/, datasets/, models/, results/
"""

# IMPORTANT: Set VLLM's multiprocessing method before any imports.
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when using VLLM's AsyncLLMEngine which spawns worker processes.
# VLLM reads this env var rather than Python's multiprocessing.set_start_method().
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
from pathlib import Path

from pipeline.tasks import list_tasks, get_task
from pipeline.core.method import Method
from pipeline import commands


def cmd_list_tasks(args):
    """List available tasks."""
    tasks = list_tasks()
    print("Available tasks:")
    for task in tasks:
        print(f"  - {task}")


def cmd_list_methods(args):
    """List available methods for a task."""
    methods = Method.list_methods(args.task)
    if methods:
        print(f"Available methods for '{args.task}':")
        for method in methods:
            print(f"  - {method}")
    else:
        print(f"No methods found for task '{args.task}'")
        print(f"Create method configs in: pipeline/configs/methods/{args.task}/")


def cmd_create_primitives(args):
    """Create primitives."""
    output_path = Path(args.output) if args.output else None
    commands.create_primitives(
        task_name=args.task,
        output_path=output_path,
        num_puzzles=args.num_puzzles,
        seed=args.seed,
    )


def cmd_create_prompts(args):
    """Create prompts."""
    output_dir = Path(args.output) if args.output else None
    primitives_path = Path(args.primitives) if args.primitives else None

    commands.create_prompts(
        task_name=args.task,
        method_name=args.method,
        primitives_path=primitives_path,
        output_dir=output_dir,
        split_name=args.split,
        seed=args.seed,
        include_assistant_prefix=not args.no_assistant_prefix,
        num_hints=args.num_hints,
    )


def cmd_generate(args):
    """Generate dataset."""
    prompts_path = Path(args.prompts) if args.prompts else None
    output_path = Path(args.output) if args.output else None

    # Parse force-hints-distribution string into dict {num_hints: probability}
    force_hints_distribution = None
    dist_str = getattr(args, "force_hints_distribution", None)
    if dist_str:
        force_hints_distribution = {}
        for pair in dist_str.split(","):
            num_hints, prob = pair.strip().split(":")
            force_hints_distribution[int(num_hints.strip())] = float(prob.strip())

    # Parse force-hints-policy string into dict {level: rate}
    force_hints_policy = None
    policy_str = getattr(args, "force_hints_policy", None)
    if policy_str:
        if not force_hints_distribution:
            parser.error("--force-hints-policy requires --force-hints-distribution")
        force_hints_policy = {}
        for pair in policy_str.split(","):
            level, rate = pair.strip().split(":")
            force_hints_policy[level.strip()] = float(rate.strip())

    commands.generate(
        task_name=args.task,
        model_name=args.model,
        method_name=args.method,
        run_id=getattr(args, "run_id", None),
        prompts_path=prompts_path,
        output_path=output_path,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        retry_incorrect=args.retry_incorrect,
        multi_turn=args.multi_turn,  # None = use method config
        max_turns=args.max_turns,  # None = use method config
        use_async=getattr(args, "use_async", False),
        force_hints_distribution=force_hints_distribution,
        force_hints_policy=force_hints_policy,
        hint_selection=getattr(args, "hint_selection", None),
        helper_model=getattr(args, "helper_model", None),
        helper_gpu_memory_utilization=getattr(args, "helper_gpu_memory_utilization", None),
        sample_strategy=getattr(args, "sample_strategy", None),
    )


def cmd_evaluate(args):
    """Evaluate model."""
    prompts_path = Path(args.prompts) if args.prompts else None
    output_path = Path(args.output) if args.output else None

    commands.evaluate(
        task_name=args.task,
        model_name=args.model,
        method_name=args.method,
        run_id=args.run_id,
        split=args.split,
        prompts_path=prompts_path,
        output_path=output_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        multi_turn=args.multi_turn,  # None = use method config
        max_turns=args.max_turns,  # None = use method config
        use_async=getattr(args, "use_async", False),
        hint_selection=getattr(args, "hint_selection", None),
        helper_model=getattr(args, "helper_model", None),
        helper_gpu_memory_utilization=getattr(args, "helper_gpu_memory_utilization", None),
    )


def cmd_analyze(args):
    """Analyze dataset accuracy by variant."""
    commands.analyze(
        dataset_path=Path(args.dataset),
        task_name=getattr(args, "task", None),
    )


def cmd_evaluate_classifier(args):
    """Evaluate binary classifier."""
    classifier_path = Path(args.classifier) if args.classifier else None
    dataset_path = Path(args.dataset) if args.dataset else None
    output_path = Path(args.output) if args.output else None

    commands.evaluate_classifier(
        task_name=args.task,
        classifier_path=classifier_path,
        method_name=args.method,
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def cmd_train_sft(args):
    """Train SFT model."""
    dataset_path = Path(args.dataset) if args.dataset else None
    output_path = Path(args.output) if args.output else None

    commands.train_sft(
        task_name=args.task,
        base_model=args.base_model,
        method_name=args.method,
        run_id=getattr(args, "run_id", None),
        dataset_path=dataset_path,
        output_path=output_path,
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
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        include_abstained=not args.no_abstained,
        include_wrong_valid_format=args.include_wrong_valid_format,
        cleanup_checkpoints=not args.keep_checkpoints,
    )


def cmd_train_rl(args):
    """Train RL model using verl."""
    train_prompts_path = Path(args.train_prompts) if args.train_prompts else None
    val_prompts_path = Path(args.val_prompts) if args.val_prompts else None
    sft_model_path = Path(args.sft_model) if args.sft_model else None
    output_path = Path(args.output) if args.output else None
    reward_function_path = Path(args.reward_function) if args.reward_function else None
    resume_path = Path(args.resume) if args.resume else None

    commands.train_rl(
        task_name=args.task,
        method_name=args.method,
        run_id=args.run_id,
        base_model=args.base_model,
        train_prompts_path=train_prompts_path,
        val_prompts_path=val_prompts_path,
        sft_model_path=sft_model_path,
        output_path=output_path,
        reward_function_path=reward_function_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        kl_coef=args.kl_coef,
        n_samples=args.n_samples,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        wandb=not args.no_wandb,
        resume_path=resume_path,
        cleanup_checkpoints=not args.keep_checkpoints,
        keep_state=args.keep_state,
    )


def cmd_train_classifier(args):
    """Train binary classifier."""
    dataset_path = Path(args.dataset) if args.dataset else None
    output_path = Path(args.output) if args.output else None

    commands.train_classifier(
        task_name=args.task,
        base_model=args.base_model,
        method_name=args.method,
        dataset_path=dataset_path,
        output_path=output_path,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        eval_split=args.eval_split,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        balance=args.balance,
        report_to=args.report_to,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
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

    # list_methods
    p = subparsers.add_parser("list_methods", help="List available methods for a task")
    p.add_argument("--task", required=True, help="Task name")
    p.set_defaults(func=cmd_list_methods)

    # create_primitives
    p = subparsers.add_parser("create_primitives", help="Generate raw puzzle data")
    p.add_argument("--task", required=True, help="Task name (e.g., countdown)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/primitives.json)")
    p.add_argument("--num-puzzles", type=int, default=None, help="Number of puzzles (omit to use all available)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.set_defaults(func=cmd_create_primitives)

    # create_prompts
    p = subparsers.add_parser("create_prompts", help="Create prompts from primitives")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--method", help="Method name for auto-derived paths and templates")
    p.add_argument("--primitives", help="Path to primitives.json (default: artifacts/{task}/primitives.json)")
    p.add_argument("--output", help="Output directory (default: artifacts/{task}/{method}/prompts/)")
    p.add_argument("--split", default="all", help="Split name (sft, rl_train, rl_val, classifier, eval, or 'all')")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    p.add_argument("--no-assistant-prefix", action="store_true", help="Don't include assistant prefix")
    p.add_argument("--num-hints", type=int, default=None,
        help="Number of hints to include from prefix_hints (0-6). Populates 'hints' field on each primitive.")
    p.set_defaults(func=cmd_create_prompts)

    # generate
    p = subparsers.add_parser("generate", help="Run model on prompts to create dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier for model resolution (used with --model sft/rl)")
    p.add_argument("--prompts", help="Path to prompts file (default: artifacts/{task}/{method}/prompts/{split}.json)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/datasets/{split}_{model}.json)")
    p.add_argument("--split", default="sft", help="Which split to generate from (default: sft)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    p.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt (best selected by --sample-strategy)")
    p.add_argument("--sample-strategy", default=None, choices=["shortest_cot", "most_hints", "prefer_abstain"],
        help="Selection strategy when --num-samples > 1 (default: most_hints for multi-turn, shortest_cot otherwise)")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--verbose", action="store_true", help="Print sample prompts during generation")
    p.add_argument("--retry-incorrect", action="store_true", help="Re-run incorrect examples from existing output")
    p.add_argument("--multi-turn", action="store_true", default=None, help="Enable multi-turn generation (auto-detected from method config)")
    p.add_argument("--max-turns", type=int, default=None, help="Maximum turns for multi-turn generation (default: from method config or 6)")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async generation (optimal throughput, processes all prompts concurrently)")
    p.add_argument("--force-hints-distribution", type=str, default=None,
        help="Distribution of forced hint counts, e.g. '1:0.5,2:0.3,3:0.2'. "
             "Maps number of hints to probability. Probabilities must sum to 1.")
    p.add_argument("--force-hints-policy", type=str, default=None,
        help="Per-level force rate, e.g. '1:0.05,2:0.05,3:0.05,4:0.15,5:0.15'. "
             "Only this fraction of examples at each level will get forced hints. "
             "Requires --force-hints-distribution.")
    p.add_argument("--hint-selection", default=None, choices=["sequential", "smart"],
        help="Hint selection strategy (default: from method config or 'sequential')")
    p.add_argument("--helper-model", default=None,
        help="Helper model for smart hint selection (default: from method config)")
    p.add_argument("--helper-gpu-memory-utilization", type=float, default=None,
        help="GPU memory utilization for helper model (default: same as --gpu-memory-utilization)")
    p.set_defaults(func=cmd_generate)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate model")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--model", required=True, help="Model name/path, or 'sft'/'rl' to use method's model")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier for model resolution (used with --model sft/rl)")
    p.add_argument("--split", default="eval", help="Prompts split to evaluate (default: eval)")
    p.add_argument("--prompts", help="Path to eval prompts (default: artifacts/{task}/{method}/prompts/{split}.json)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/results/{split}_{model}.json)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature (0 for greedy)")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--verbose", action="store_true", help="Print sample prompts during generation")
    p.add_argument("--multi-turn", action="store_true", default=None, help="Enable multi-turn generation (auto-detected from method config)")
    p.add_argument("--max-turns", type=int, default=None, help="Maximum turns for multi-turn generation (default: from method config or 6)")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async generation (optimal throughput)")
    p.add_argument("--hint-selection", default=None, choices=["sequential", "smart"],
        help="Hint selection strategy (default: from method config or 'sequential')")
    p.add_argument("--helper-model", default=None,
        help="Helper model for smart hint selection (default: from method config)")
    p.add_argument("--helper-gpu-memory-utilization", type=float, default=None,
        help="GPU memory utilization for helper model (default: same as --gpu-memory-utilization)")
    p.set_defaults(func=cmd_evaluate)

    # analyze
    p = subparsers.add_parser("analyze", help="Analyze dataset accuracy by variant (prints grid)")
    p.add_argument("--dataset", required=True, help="Path to dataset or eval results")
    p.add_argument("--task", help="Task name (for task-specific metrics on raw datasets)")
    p.set_defaults(func=cmd_analyze)

    # evaluate_classifier
    p = subparsers.add_parser("evaluate_classifier", help="Evaluate binary classifier on a dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--classifier", help="Path to classifier model (default: artifacts/{task}/{method}/models/classifier)")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--dataset", help="Path to dataset with 'prompt', 'generation', 'correct' fields")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/results/classifier_eval.json)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    p.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    p.set_defaults(func=cmd_evaluate_classifier)

    # train_sft
    p = subparsers.add_parser("train_sft", help="Train SFT model on generated dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--base-model", required=True, help="Base model to fine-tune")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier for organizing outputs (default: 'default')")
    p.add_argument("--dataset", help="Path to generated dataset (default: auto-detect from method)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/models/sft/{run_id}/model)")
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
    p.add_argument("--report-to", default="wandb", help="Reporting integration (wandb, none)")
    p.add_argument("--project-name", help="Wandb project name (default: {task}-sft)")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})")
    p.add_argument("--no-abstained", action="store_true", help="Exclude abstained examples (by default they're included)")
    p.add_argument("--include-wrong-valid-format", action="store_true", help="Include wrong answers with valid format (task-specific, e.g., valid UCI but wrong move for chess)")
    p.add_argument("--keep-checkpoints", action="store_true", help="Keep intermediate checkpoints after training (by default they are deleted)")
    p.set_defaults(func=cmd_train_sft)

    # train_rl
    p = subparsers.add_parser("train_rl", help="Train RL model using verl (GRPO)")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--method", help="Method name for auto-derived paths, template, and reward config")
    p.add_argument("--run-id", help="Run identifier for organizing outputs (default: 'default')")
    p.add_argument("--base-model", help="Base model for cold-start RL (mutually exclusive with --sft-model)")
    p.add_argument("--train-prompts", help="Path to RL train prompts (default: artifacts/{task}/{method}/prompts/rl_train.parquet)")
    p.add_argument("--val-prompts", help="Path to RL validation prompts (default: artifacts/{task}/{method}/prompts/rl_val.parquet)")
    p.add_argument("--sft-model", help="Path to SFT model (mutually exclusive with --base-model)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/models/rl/{run_id}/model)")
    p.add_argument("--reward-function", help="Path to reward function (default: task's reward function)")
    p.add_argument("--train-batch-size", type=int, default=64, help="Training batch size")
    p.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--total-steps", type=int, default=400, help="Total training steps")
    p.add_argument("--kl-coef", type=float, default=0.001, help="KL divergence coefficient")
    p.add_argument("--n-samples", type=int, default=16, help="Number of samples per prompt (group size)")
    p.add_argument("--save-freq", type=int, default=25, help="Checkpoint save frequency")
    p.add_argument("--test-freq", type=int, default=None, help="Validation/logging frequency (default: same as save-freq)")
    p.add_argument("--max-prompt-length", type=int, default=2048, help="Maximum prompt length in tokens")
    p.add_argument("--max-response-length", type=int, default=2048, help="Maximum response length in tokens")
    p.add_argument("--max-model-len", type=int, default=8192, help="Maximum model context length (default: 8192). Set higher for multi-turn.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="GPU memory utilization")
    p.add_argument("--project-name", help="Wandb project name (default: {task}-rl)")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--resume", help="Path to resume from existing run")
    p.add_argument("--keep-checkpoints", action="store_true", help="Keep checkpoints and rollouts after training (by default they are deleted)")
    p.add_argument("--keep-state", action="store_true", help="Keep the last optimizer state checkpoint after training")
    p.set_defaults(func=cmd_train_rl)

    # train_classifier
    p = subparsers.add_parser("train_classifier", help="Train binary classifier for puzzle solvability prediction")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--base-model", required=True, help="Base model for classification")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--dataset", help="Path to generated dataset (default: auto-detect from method)")
    p.add_argument("--output", help="Output path (default: artifacts/{task}/{method}/models/classifier_{dataset})")
    p.add_argument("--mode", default="binary", choices=["binary", "three_class"], help="Classification mode")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    p.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    p.add_argument("--eval-split", type=float, default=0.1, help="Fraction of data for evaluation")
    p.add_argument("--eval-steps", type=int, default=None, help="Evaluate every N steps (default: once per epoch)")
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    p.add_argument("--balance", default="none", choices=["none", "downsample", "upsample"], help="Class balancing strategy")
    p.add_argument("--report-to", default="wandb", help="Reporting integration (wandb, none)")
    p.add_argument("--project-name", default="binary_classifier", help="Wandb project name")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {task}_{dataset_source}_{model})")
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
