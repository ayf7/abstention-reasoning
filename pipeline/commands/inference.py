"""
Inference commands - generation and evaluation.
"""

from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import Generator, GenerationConfig
from pipeline.tasks import get_task


def generate(
    task_name: str,
    prompts_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> Path:
    """
    Generate model outputs on prompts.

    Supports resumption - if output file exists, skips already-completed indices.
    Saves after each batch for crash recovery.

    Args:
        task_name: Name of task
        prompts_path: Path to prompts file
        output_path: Where to save dataset
        model_name: Model to use for generation
        ... generation config ...

    Returns:
        Path to created dataset file
    """
    task = get_task(task_name)

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Check for existing output (for resumption)
    completed_indices = set()
    records = []
    if output_path.exists():
        records = load_json(output_path)
        completed_indices = {r["index"] for r in records}
        print(f"Resuming: found {len(completed_indices)} completed examples")

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
        for prompt_data, gen_result in zip(batch_prompts_data, batch_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            is_correct, meta = task.check_correctness(primitive, gen_result["text"])

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
            records.append(record)

        # Save after each batch
        save_json(output_path, records)

        batch_num = batch_idx // batch_size + 1
        batch_correct = sum(1 for r in records[-len(batch_results):] if r["correct"])
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
    prompts_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,  # Greedy for eval
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> Path:
    """
    Evaluate a model on prompts and compute metrics.

    Args:
        task_name: Name of task
        prompts_path: Path to eval prompts
        output_path: Where to save results
        model_name: Model to evaluate
        ... generation config ...

    Returns:
        Path to results file
    """
    task = get_task(task_name)

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Initialize generator
    config = GenerationConfig(
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    generator = Generator(config)

    # Generate
    prompts = [p["prompt"] for p in prompts_data]

    def progress_callback(batch_idx, _results):
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"Batch {batch_idx + 1}/{total_batches} complete")

    print(f"Evaluating {model_name}...")
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

        detail = {
            "index": prompt_data["index"],
            "variant": prompt_data.get("variant", "unknown"),
            "correct": is_correct,
            "finish_reason": finish_reason,
            "generation": gen_result["text"],
            "predicted_answer": meta.get("predicted_answer"),
            "ground_truth": prompt_data["ground_truth"],
            "error": meta.get("error"),
        }
        details.append(detail)

    # Compute metrics with breakdown by finish reason
    from collections import defaultdict

    # Count by variant: correct, truncated, wrong_output
    variant_counts = defaultdict(lambda: {"correct": 0, "truncated": 0, "wrong_output": 0})
    for d in details:
        variant = d["variant"]
        if d["correct"]:
            variant_counts[variant]["correct"] += 1
        elif d["finish_reason"] == "length":
            variant_counts[variant]["truncated"] += 1
        else:
            variant_counts[variant]["wrong_output"] += 1

    # Compute totals
    total_correct = sum(v["correct"] for v in variant_counts.values())
    total_truncated = sum(v["truncated"] for v in variant_counts.values())
    total_wrong = sum(v["wrong_output"] for v in variant_counts.values())
    total = len(details)

    # Accuracy excludes truncated (they didn't get a chance to answer)
    non_truncated = total_correct + total_wrong
    accuracy = total_correct / non_truncated if non_truncated > 0 else 0.0

    # Per-variant accuracy (excluding truncated)
    accuracy_by_variant = {}
    for variant, counts in variant_counts.items():
        non_trunc = counts["correct"] + counts["wrong_output"]
        accuracy_by_variant[variant] = counts["correct"] / non_trunc if non_trunc > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "accuracy_by_variant": accuracy_by_variant,
        "total": total,
        "correct": total_correct,
        "truncated": total_truncated,
        "wrong_output": total_wrong,
        "counts_by_variant": dict(variant_counts),
    }

    # Build results
    results = {
        "model": model_name,
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

    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Accuracy (excl. truncated): {accuracy:.1%} ({total_correct}/{non_truncated})")
    print(f"  Correct: {total_correct}")
    print(f"  Truncated: {total_truncated}")
    print(f"  Wrong output: {total_wrong}")
    print(f"\nBy variant:")
    for variant in sorted(variant_counts.keys()):
        counts = variant_counts[variant]
        acc = accuracy_by_variant[variant]
        non_trunc = counts["correct"] + counts["wrong_output"]
        print(f"  {variant}: {acc:.1%} ({counts['correct']}/{non_trunc}) "
              f"[correct={counts['correct']}, truncated={counts['truncated']}, wrong={counts['wrong_output']}]")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    return output_path
