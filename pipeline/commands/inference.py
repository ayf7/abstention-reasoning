"""
Inference commands - generation and evaluation.
"""

import re
from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import Generator, GenerationConfig
from pipeline.core.method import Method
from pipeline.core.utils import model_short_name
from pipeline.tasks import get_task


def extract_cot_length(generation: str) -> int:
    """
    Extract the length of chain-of-thought reasoning from generation.

    Looks for content inside <think>...</think> tags.
    Returns character count, or 0 if no CoT found.
    """
    match = re.search(r'<think>(.*?)</think>', generation, re.DOTALL)
    if match:
        return len(match.group(1))
    return 0


def select_best_sample(samples: list[dict], task, primitive: dict) -> dict:
    """
    Select the best sample from multiple generations.

    Strategy:
    1. Check correctness for each sample
    2. Among correct samples, pick shortest CoT
    3. If no correct samples, pick shortest CoT anyway

    Args:
        samples: List of generation results with 'text', 'finish_reason', 'token_count'
        task: Task instance for checking correctness
        primitive: Primitive data with ground truth

    Returns:
        Best sample dict with added 'correct' and 'metadata' fields
    """
    # Check correctness for all samples
    evaluated_samples = []
    for sample in samples:
        is_correct, meta = task.check_correctness(primitive, sample["text"])
        cot_length = extract_cot_length(sample["text"])
        evaluated_samples.append({
            **sample,
            "correct": is_correct,
            "metadata": meta,
            "cot_length": cot_length,
        })

    # Separate correct and incorrect
    correct_samples = [s for s in evaluated_samples if s["correct"]]
    incorrect_samples = [s for s in evaluated_samples if not s["correct"]]

    # Pick shortest CoT among correct, or shortest overall if none correct
    candidates = correct_samples if correct_samples else incorrect_samples
    best_sample = min(candidates, key=lambda s: s["cot_length"])

    # Remove cot_length (internal field)
    del best_sample["cot_length"]

    return best_sample


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
    num_samples: int = 1,
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
        model_name=actual_model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
    )
    generator = Generator(config)

    # Process in batches with incremental saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size

    print(f"Generating with {actual_model_name}...")
    for batch_idx in range(0, len(remaining_prompts), batch_size):
        batch_prompts_data = remaining_prompts[batch_idx:batch_idx + batch_size]
        batch_prompts = [p["prompt"] for p in batch_prompts_data]

        # Generate batch
        batch_results = generator.generate(batch_prompts)

        # Process results
        batch_correct = 0
        for prompt_data, gen_samples in zip(batch_prompts_data, batch_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            # If num_samples > 1, select best sample; otherwise use single sample
            if num_samples > 1:
                best_sample = select_best_sample(gen_samples, task, primitive)
            else:
                # Single sample case - evaluate it
                sample = gen_samples[0]
                is_correct, meta = task.check_correctness(primitive, sample["text"])
                best_sample = {
                    **sample,
                    "correct": is_correct,
                    "metadata": meta,
                }

            if best_sample["correct"]:
                batch_correct += 1

            record = {
                "index": prompt_data["index"],
                "variant": prompt_data.get("variant", "unknown"),
                "prompt": prompt_data["prompt"],
                "generation": best_sample["text"],
                "correct": best_sample["correct"],
                "finish_reason": best_sample["finish_reason"],
                "token_count": best_sample["token_count"],
                "metadata": best_sample["metadata"],
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
    for prompt_data, gen_samples in zip(prompts_data, generations):
        # gen_samples is a list of samples (even if num_samples=1)
        # For evaluation, use the first sample
        gen_result = gen_samples[0]

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


def evaluate_classifier(
    task_name: str,
    classifier_path: Path,
    method_name: str | None = None,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 8,
    max_length: int = 2048,
) -> Path:
    """
    Evaluate a binary classifier on a dataset with ground truth correctness labels.

    Computes confusion matrix, precision, recall, F1 score by comparing
    classifier predictions to actual correctness labels.

    Args:
        task_name: Name of task
        classifier_path: Path to trained classifier model
        method_name: Method name for auto-derived paths
        dataset_path: Path to dataset with 'prompt', 'generation', 'correct' fields
                     (default: uses classifier dataset from method)
        output_path: Where to save results
                    (default: artifacts/{task}/{method}/results/classifier_eval.json)
        batch_size: Batch size for inference
        max_length: Maximum sequence length

    Returns:
        Path to results file
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default classifier path from method
    if classifier_path is None:
        if method is None:
            raise ValueError(
                "Either --classifier or --method must be specified. "
                "Use --method to auto-derive paths, or --classifier for explicit paths."
            )
        classifier_path = method.classifier_model_path(task_name)

    # Default dataset path - look for classifier dataset
    if dataset_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --dataset must be specified. "
                "Use --method to auto-derive paths, or --dataset for explicit paths."
            )
        datasets_dir = method.datasets_dir(task_name)
        classifier_files = list(datasets_dir.glob("classifier_*.json"))
        if not classifier_files:
            raise FileNotFoundError(
                f"No classifier datasets found in {datasets_dir}. "
                f"Run 'python -m pipeline generate --task {task_name} --method {method_name} --split classifier' first."
            )
        dataset_path = classifier_files[0]

    # Default output path - derive from dataset name
    if output_path is None:
        # Extract dataset identifier from filename (e.g., "eval_sft" from "eval_sft.json")
        dataset_stem = dataset_path.stem  # e.g., "eval_sft", "classifier_sft"
        output_name = f"classifier_{dataset_stem}.json"

        if method is not None:
            output_path = method.results_dir(task_name) / output_name
        else:
            # Put results in results dir next to dataset
            output_path = dataset_path.parent.parent / "results" / output_name

    print(f"=== Classifier Evaluation ===")
    print(f"Classifier: {classifier_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"==============================")

    # Load dataset
    raw_data = load_json(dataset_path)

    # Handle eval results format (with "details" array and separate prompts file)
    if isinstance(raw_data, dict) and "details" in raw_data:
        print("Detected eval results format, loading prompts separately...")
        prompts_file = raw_data.get("prompts")
        prompts_by_index = {}
        if prompts_file:
            prompts_path = Path(prompts_file)
            # Try the referenced path first, then look relative to dataset
            if not prompts_path.exists():
                # Try finding eval.json in same method directory
                alt_path = dataset_path.parent.parent / "prompts" / "eval.json"
                if alt_path.exists():
                    prompts_path = alt_path
                    print(f"  Using prompts from: {prompts_path}")
            if prompts_path.exists():
                prompts_data = load_json(prompts_path)
                prompts_by_index = {p["index"]: p["prompt"] for p in prompts_data}
            else:
                print(f"  Warning: prompts file not found at {prompts_file}")

        # Convert to standard format
        data = []
        for detail in raw_data["details"]:
            record = {
                "index": detail["index"],
                "variant": detail.get("variant", "unknown"),
                "prompt": prompts_by_index.get(detail["index"], ""),
                "generation": detail["generation"],
                "correct": detail.get("correct", False),
            }
            data.append(record)
    else:
        # Standard dataset format (list of records with prompt, generation, correct)
        data = raw_data

    print(f"Loaded {len(data)} examples")

    # Count ground truth distribution
    gt_positive = sum(1 for ex in data if ex.get("correct", False))
    gt_negative = len(data) - gt_positive
    print(f"Ground truth: {gt_positive} correct, {gt_negative} incorrect")

    # Load classifier
    print(f"Loading classifier from {classifier_path}")
    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare texts (prompt only - matches training)
    texts = []
    for ex in data:
        if isinstance(ex["prompt"], list):
            # Chat format - concatenate messages
            prompt_text = ""
            for msg in ex["prompt"]:
                prompt_text += f"{msg['role']}: {msg['content']}\n"
        else:
            prompt_text = ex["prompt"]
        texts.append(prompt_text)

    # Run inference
    print(f"Running classifier inference...")
    predictions = []
    probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())

        if (i // batch_size + 1) % 10 == 0:
            print(f"Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    # Compute metrics
    ground_truth = [1 if ex.get("correct", False) else 0 for ex in data]

    tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
    fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)

    accuracy = (tp + tn) / len(data) if len(data) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Build confusion matrix
    confusion_matrix = {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix,
        "total": len(data),
        "ground_truth_positive": gt_positive,
        "ground_truth_negative": gt_negative,
        "predicted_positive": tp + fp,
        "predicted_negative": tn + fn,
    }

    # Print results
    print(f"\n=== Classifier Evaluation Results ===")
    print(f"Total examples: {len(data)}")
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Correct  Incorrect")
    print(f"Actual Correct     {tp:5d}      {fn:5d}")
    print(f"Actual Incorrect   {fp:5d}      {tn:5d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.1f}%)")
    print(f"\nInterpretation:")
    print(f"  - Precision: When classifier predicts 'correct', it's right {precision*100:.1f}% of the time")
    print(f"  - Recall: Classifier finds {recall*100:.1f}% of actually correct examples")
    print(f"======================================")

    # Build detailed results
    details = []
    for i, ex in enumerate(data):
        detail = {
            "index": ex.get("index", i),
            "variant": ex.get("variant", "unknown"),
            "ground_truth": ground_truth[i],
            "prediction": predictions[i],
            "correct_classification": ground_truth[i] == predictions[i],
            "probability_correct": probabilities[i][1],
            "probability_incorrect": probabilities[i][0],
        }
        details.append(detail)

    results = {
        "classifier": str(classifier_path),
        "dataset": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "details": details,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    return output_path
