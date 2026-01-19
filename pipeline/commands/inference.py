"""
Inference commands - generation and evaluation.
"""

from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import Generator, GenerationConfig
from pipeline.core.method import Method
from pipeline.core.utils import model_short_name
from pipeline.tasks import get_task


def generate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    split: str = "sft",
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    verbose: bool = False,
    retry_incorrect: bool = False,
) -> Path:
    """
    Generate model outputs on prompts.

    Supports resumption - if output file exists, skips already-completed indices.
    Saves after each batch for crash recovery.

    Args:
        task_name: Name of task
        model_name: Model to use for generation
        method_name: Method name for auto-derived paths
        prompts_path: Path to prompts file (default: artifacts/{task}/{method}/prompts/{split}.json)
        output_path: Where to save dataset (default: artifacts/{task}/{method}/datasets/{split}_{model}.json)
        split: Which split to generate from (default: sft)
        retry_incorrect: If True, re-run incorrect examples
        ... generation config ...

    Returns:
        Path to created dataset file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default prompts path
    if prompts_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --prompts must be specified. "
                "Use --method to auto-derive paths, or --prompts for explicit paths."
            )
        prompts_path = method.prompts_path(task_name, split)

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_path = method.dataset_path(task_name, split, model_name)

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Check for existing output (for resumption)
    records_by_index = {}
    if output_path.exists():
        existing_records = load_json(output_path)
        records_by_index = {r["index"]: r for r in existing_records}

        if retry_incorrect:
            # Identify incorrect indices to retry (but keep all records)
            incorrect_indices = {r["index"] for r in existing_records if not r.get("correct", False)}
            correct_count = len(existing_records) - len(incorrect_indices)
            print(f"Retry mode: {correct_count} correct, retrying {len(incorrect_indices)} incorrect")
            completed_indices = {r["index"] for r in existing_records if r.get("correct", False)}
        else:
            completed_indices = set(records_by_index.keys())
            print(f"Resuming: found {len(completed_indices)} completed examples")
    else:
        completed_indices = set()

    # Filter to remaining prompts
    remaining_prompts = [p for p in prompts_data if p["index"] not in completed_indices]
    if not remaining_prompts:
        print("All prompts already completed!")
        return output_path

    print(f"Generating {len(remaining_prompts)} remaining prompts...")

    # Initialize generator
    config = GenerationConfig(
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
    )
    generator = Generator(config)

    # Process in batches with incremental saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size

    print(f"Generating with {model_name}...")
    for batch_idx in range(0, len(remaining_prompts), batch_size):
        batch_prompts_data = remaining_prompts[batch_idx:batch_idx + batch_size]
        batch_prompts = [p["prompt"] for p in batch_prompts_data]

        # Generate batch
        batch_results = generator.generate(batch_prompts)

        # Process results
        batch_correct = 0
        for prompt_data, gen_result in zip(batch_prompts_data, batch_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            is_correct, meta = task.check_correctness(primitive, gen_result["text"])
            if is_correct:
                batch_correct += 1

            record = {
                "index": prompt_data["index"],
                "variant": prompt_data.get("variant", "unknown"),
                "prompt": prompt_data["prompt"],
                "generation": gen_result["text"],
                "correct": is_correct,
                "finish_reason": gen_result["finish_reason"],
                "token_count": gen_result["token_count"],
                "metadata": meta,
            }
            # Update in-place (replaces old record if retrying)
            records_by_index[prompt_data["index"]] = record

        # Save after each batch (convert dict values to list, sorted by index)
        records = sorted(records_by_index.values(), key=lambda r: r["index"])
        save_json(output_path, records)

        batch_num = batch_idx // batch_size + 1
        print(f"Batch {batch_num}/{total_batches}: {batch_correct}/{len(batch_results)} correct (saved)")

    # Final stats
    correct = sum(1 for r in records if r["correct"])
    print(f"Generated {len(records)} examples: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")

    # Save
    save_json(output_path, records)
    print(f"Saved dataset to {output_path}")

    return output_path


def evaluate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,  # Greedy for eval
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    verbose: bool = False,
) -> Path:
    """
    Evaluate a model on prompts and compute metrics.

    Args:
        task_name: Name of task
        model_name: Model to evaluate (can be "sft" or "rl" to use method's model paths)
        method_name: Method name for auto-derived paths
        prompts_path: Path to eval prompts (default: artifacts/{task}/{method}/prompts/eval.json)
        output_path: Where to save results (default: artifacts/{task}/{method}/results/eval_{model}.json)
        ... generation config ...

    Returns:
        Path to results file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Resolve model shortcuts (sft, rl)
    actual_model_name = model_name
    if method is not None:
        if model_name == "sft":
            actual_model_name = str(method.sft_model_path(task_name))
        elif model_name == "rl":
            actual_model_name = str(method.rl_model_path(task_name))

    # Default prompts path
    if prompts_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --prompts must be specified. "
                "Use --method to auto-derive paths, or --prompts for explicit paths."
            )
        prompts_path = method.prompts_path(task_name, "eval")

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_path = method.results_path(task_name, model_name)

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Initialize generator
    config = GenerationConfig(
        model_name=actual_model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
    )
    generator = Generator(config)

    # Generate
    prompts = [p["prompt"] for p in prompts_data]

    def progress_callback(batch_idx, _results):
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"Batch {batch_idx + 1}/{total_batches} complete")

    print(f"Evaluating {actual_model_name}...")
    generations = generator.generate_batched(prompts, callback=progress_callback)

    # Create result records
    details = []
    for prompt_data, gen_result in zip(prompts_data, generations):
        primitive = {
            "index": prompt_data["index"],
            **prompt_data["ground_truth"],
            "variant": prompt_data.get("variant", "unknown"),
        }

        is_correct, meta = task.check_correctness(primitive, gen_result["text"])
        finish_reason = gen_result.get("finish_reason", "unknown")
        token_count = gen_result.get("token_count", 0)

        detail = {
            "index": prompt_data["index"],
            "variant": prompt_data.get("variant", "unknown"),
            "correct": is_correct,
            "finish_reason": finish_reason,
            "token_count": token_count,
            "generation": gen_result["text"],
            "predicted_answer": meta.get("predicted_answer"),
            "ground_truth": prompt_data["ground_truth"],
            "error": meta.get("error"),
            **{k: v for k, v in meta.items() if k not in ("predicted_answer", "error")},
        }
        details.append(detail)

    # Compute metrics using task-specific logic
    metrics = task.compute_metrics(details)

    # Build results
    results = {
        "model": actual_model_name,
        "model_alias": model_name,  # Keep the alias (sft, rl) if used
        "prompts": str(prompts_path),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        },
        "metrics": metrics,
        "details": details,
    }

    # Print summary using task-specific formatting
    print(task.format_metrics(metrics, model_name))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    return output_path


def analyze(dataset_path: Path, task_name: str | None = None) -> dict:
    """
    Analyze a dataset or eval results and print metrics.

    Delegates display formatting to task.format_metrics() for task-specific output.

    Args:
        dataset_path: Path to dataset.json or results.json
        task_name: Optional task name (required for task-specific formatting)

    Returns:
        Dict with metrics
    """
    data = load_json(dataset_path)
    model_name = data.get("model") if isinstance(data, dict) else None

    # Get task for formatting (if provided)
    task = get_task(task_name) if task_name else None

    # If eval results with stored metrics, use those
    if isinstance(data, dict) and "metrics" in data:
        metrics = data["metrics"]
        if task:
            print(task.format_metrics(metrics, model_name))
        else:
            # Fallback: basic display without task
            print(_format_basic_metrics(metrics, model_name))
        return metrics

    # Otherwise compute from records
    records = data["details"] if isinstance(data, dict) and "details" in data else data

    if task:
        metrics = task.compute_metrics(records)
        print(task.format_metrics(metrics, model_name))
        return metrics

    # Fallback: basic counts without task
    from collections import defaultdict
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        variant = r.get("variant", "unknown")
        counts[variant]["total"] += 1
        if r.get("correct", False):
            counts[variant]["correct"] += 1

    metrics = {"counts_by_variant": dict(counts)}
    print(_format_basic_metrics(metrics, model_name))
    return metrics


def _format_basic_metrics(metrics: dict, model_name: str | None = None) -> str:
    """Basic metrics formatting when no task is specified."""
    lines = ["", "=== Metrics ==="]
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(f"Total: {metrics.get('total', 0)}")
    lines.append(f"Correct: {metrics.get('correct', 0)}")
    lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

    by_variant = metrics.get("counts_by_variant") or metrics.get("by_variant", {})
    if by_variant:
        lines.append("")
        lines.append("By variant:")
        for variant, c in sorted(by_variant.items(), key=lambda x: str(x[0])):
            total = c.get("total", 0)
            correct = c.get("correct", 0)
            acc = correct / total if total > 0 else 0
            lines.append(f"  {variant}: {correct}/{total} ({acc:.0%})")

    return "\n".join(lines)
