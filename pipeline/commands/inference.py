"""
Inference commands - generation and evaluation.
"""

import re
from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import Generator, GenerationConfig, AsyncGenerator, run_async_generation
from pipeline.core.hint_selector import HintSelector
from pipeline.core.method import Method
from pipeline.core.utils import model_short_name
from pipeline.tasks import get_task


def extract_generation_hints(text: str) -> list[str]:
    """Extract '(using the <category> of <name>)' hints from generated text.

    Handles nested parentheses in concept names like (a+b)^2.
    """
    hints = []
    pattern_start = re.compile(r'\(using the (\w+) of ', re.IGNORECASE)

    for match in pattern_start.finditer(text):
        category = match.group(1)
        start_pos = match.end()

        # Find matching closing paren by counting depth
        depth = 1
        pos = start_pos
        while pos < len(text) and depth > 0:
            if text[pos] == '(':
                depth += 1
            elif text[pos] == ')':
                depth -= 1
            pos += 1

        if depth == 0:
            name = text[start_pos:pos - 1]
            clean_name = name.strip().replace('**', '').replace('*', '')
            hint = f'{category} of {clean_name}'
            hints.append(hint)

    return hints


def compute_hint_metrics(details: list[dict]) -> dict:
    """
    Compute hint usage metrics for multi-turn evaluation.

    Returns a dict with:
    - counts_by_hints_and_variant: {num_hints: {variant: count}}
    - correct_by_hints_and_variant: {num_hints: {variant: correct_count}}
    - counts_by_hints_and_level: {num_hints: {level: count}}
    - correct_by_hints_and_level: {num_hints: {level: correct_count}}
    - total_hints: total hints used across all examples
    - avg_hints: average hints per example
    """
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    correct_counts = defaultdict(lambda: defaultdict(int))
    counts_by_level = defaultdict(lambda: defaultdict(int))
    correct_by_level = defaultdict(lambda: defaultdict(int))
    total_hints = 0
    max_hints = 0

    for record in details:
        variant = record.get("variant", "unknown")
        level = record.get("level", "unknown")
        num_hints = record.get("num_hints", 0)
        counts[num_hints][variant] += 1
        total_hints += num_hints
        max_hints = max(max_hints, num_hints)
        if level != "unknown":
            counts_by_level[num_hints][level] += 1
        if record.get("correct", False):
            correct_counts[num_hints][variant] += 1
            if level != "unknown":
                correct_by_level[num_hints][level] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    counts_dict = {h: dict(counts[h]) for h in range(max_hints + 1)}
    correct_dict = {h: dict(correct_counts[h]) for h in range(max_hints + 1)}
    counts_level_dict = {h: dict(counts_by_level[h]) for h in range(max_hints + 1)}
    correct_level_dict = {h: dict(correct_by_level[h]) for h in range(max_hints + 1)}

    return {
        "counts_by_hints_and_variant": counts_dict,
        "correct_by_hints_and_variant": correct_dict,
        "counts_by_hints_and_level": counts_level_dict,
        "correct_by_hints_and_level": correct_level_dict,
        "total_hints": total_hints,
        "avg_hints": total_hints / len(details) if details else 0,
        "max_hints": max_hints,
    }


def _format_hint_tables(
    row_labels: list[str],
    counts_map: dict,
    correct_map: dict,
    max_hints: int,
    row_header: str,
    section_title: str,
) -> list[str]:
    """Format counts and accuracy tables for a set of row labels.

    Args:
        row_labels: Sorted list of row label strings (variants or levels).
        counts_map: {num_hints: {label: count}} dict.
        correct_map: {num_hints: {label: correct_count}} dict.
        max_hints: Maximum hint count to display columns for.
        row_header: Column header for the row label (e.g. "Type", "Level").
        section_title: Title prefix (e.g. "TYPE", "DIFFICULTY").
    """

    def get_val(mapping, h, label):
        return mapping.get(str(h), {}).get(label, mapping.get(h, {}).get(label, 0))

    # Compute label column width from longest label
    label_w = max(len(row_header), max((len(l) for l in row_labels), default=8)) + 2
    num_col_w = 10
    acc_col_w = 16

    lines = []

    # --- Counts table ---
    total_w = label_w + num_col_w * (max_hints + 3)
    lines.extend([
        "",
        f"COUNTS BY {section_title} AND HINTS",
        "-" * total_w,
    ])

    header = f"{row_header:<{label_w}}"
    for h in range(max_hints + 1):
        header += f"{h:>{num_col_w}}"
    header += f"{'Total':>{num_col_w}}{'Hints':>{num_col_w}}"
    lines.append(header)
    lines.append("-" * total_w)

    grand_total = 0
    grand_hints = 0
    col_totals = [0] * (max_hints + 1)

    for label in row_labels:
        row = f"{label:<{label_w}}"
        row_total = 0
        row_hints = 0
        for h in range(max_hints + 1):
            c = get_val(counts_map, h, label)
            row_total += c
            row_hints += h * c
            col_totals[h] += c
            row += f"{c:>{num_col_w}}"
        grand_total += row_total
        grand_hints += row_hints
        row += f"{row_total:>{num_col_w}}{row_hints:>{num_col_w}}"
        lines.append(row)

    lines.append("-" * total_w)
    totals_row = f"{'Total':<{label_w}}"
    for h in range(max_hints + 1):
        totals_row += f"{col_totals[h]:>{num_col_w}}"
    totals_row += f"{grand_total:>{num_col_w}}{grand_hints:>{num_col_w}}"
    lines.append(totals_row)

    # --- Accuracy table ---
    acc_total_w = label_w + acc_col_w * (max_hints + 2)
    lines.extend([
        "",
        "",
        f"ACCURACY BY {section_title} AND HINTS",
        "-" * acc_total_w,
    ])

    header = f"{row_header:<{label_w}}"
    for h in range(max_hints + 1):
        header += f"{h:>{acc_col_w}}"
    header += f"{'Total':>{acc_col_w}}"
    lines.append(header)
    lines.append("-" * acc_total_w)

    overall_correct = 0
    overall_total = 0
    col_correct_totals = [0] * (max_hints + 1)
    col_count_totals = [0] * (max_hints + 1)

    for label in row_labels:
        row = f"{label:<{label_w}}"
        row_correct = 0
        row_total = 0
        for h in range(max_hints + 1):
            c = get_val(counts_map, h, label)
            corr = get_val(correct_map, h, label)
            row_correct += corr
            row_total += c
            col_correct_totals[h] += corr
            col_count_totals[h] += c
            if c > 0:
                cell = f"{corr}/{c} ({100*corr/c:.0f}%)"
                row += f"{cell:>{acc_col_w}}"
            else:
                row += f"{'--':>{acc_col_w}}"
        overall_correct += row_correct
        overall_total += row_total
        if row_total > 0:
            cell = f"{row_correct}/{row_total} ({100*row_correct/row_total:.0f}%)"
            row += f"{cell:>{acc_col_w}}"
        else:
            row += f"{'--':>{acc_col_w}}"
        lines.append(row)

    lines.append("-" * acc_total_w)
    totals_row = f"{'Total':<{label_w}}"
    for h in range(max_hints + 1):
        c = col_count_totals[h]
        corr = col_correct_totals[h]
        if c > 0:
            cell = f"{corr}/{c} ({100*corr/c:.0f}%)"
            totals_row += f"{cell:>{acc_col_w}}"
        else:
            totals_row += f"{'--':>{acc_col_w}}"
    if overall_total > 0:
        cell = f"{overall_correct}/{overall_total} ({100*overall_correct/overall_total:.0f}%)"
        totals_row += f"{cell:>{acc_col_w}}"
    lines.append(totals_row)

    return lines


def format_hint_metrics(hint_metrics: dict, details: list[dict]) -> str:
    """Format hint metrics as tables for display.

    Prints tables grouped by variant (problem type) and, if level data is
    available, by difficulty level.
    """
    max_hints = hint_metrics["max_hints"]

    variants = sorted(set(d.get("variant", "unknown") for d in details))
    levels = sorted(set(d.get("level", "unknown") for d in details) - {"unknown"})

    lines = [
        "",
        "=" * 90,
        "HINT USAGE ANALYSIS",
        "=" * 90,
    ]

    # Tables by variant (problem type)
    lines.extend(_format_hint_tables(
        row_labels=variants,
        counts_map=hint_metrics["counts_by_hints_and_variant"],
        correct_map=hint_metrics["correct_by_hints_and_variant"],
        max_hints=max_hints,
        row_header="Type",
        section_title="TYPE",
    ))

    # Tables by difficulty level (if available)
    if levels:
        lines.append("")
        lines.extend(_format_hint_tables(
            row_labels=levels,
            counts_map=hint_metrics["counts_by_hints_and_level"],
            correct_map=hint_metrics["correct_by_hints_and_level"],
            max_hints=max_hints,
            row_header="Level",
            section_title="DIFFICULTY",
        ))

    return "\n".join(lines)


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


def select_best_sample(
    samples: list[dict],
    task,
    primitive: dict,
    strategy: str = "shortest_cot",
) -> dict:
    """
    Select the best sample from multiple generations.

    Strategies:
    - "shortest_cot": Among correct samples, pick shortest CoT. Falls back to
      shortest CoT among incorrect if none correct.
    - "most_hints": Among correct samples, pick the one with most hint requests.
      Falls back to most hints among incorrect if none correct. Useful for
      generating SFT data that teaches models to use hints.
    - "prefer_abstain": If any sample is correct, pick shortest correct CoT.
      Otherwise, prefer an abstaining sample (for hard problems the model should
      learn to abstain on). Falls back to shortest incorrect CoT.

    Args:
        samples: List of generation results with 'text', 'finish_reason', 'token_count'
        task: Task instance for checking correctness
        primitive: Primitive data with ground truth
        strategy: Selection strategy ("shortest_cot", "most_hints", or "prefer_abstain")

    Returns:
        Best sample dict with added 'correct' and 'metadata' fields
    """
    # Check correctness for all samples
    evaluated_samples = []
    for sample in samples:
        is_correct, meta = task.check_correctness(primitive, sample["text"])
        cot_length = extract_cot_length(sample["text"])
        num_hints = sample["text"].count("<request>")
        evaluated_samples.append({
            **sample,
            "correct": is_correct,
            "metadata": meta,
            "cot_length": cot_length,
            "_num_hints": num_hints,
            "_abstained": meta.get("abstained", False),
        })

    # Separate correct and incorrect
    correct_samples = [s for s in evaluated_samples if s["correct"]]
    incorrect_samples = [s for s in evaluated_samples if not s["correct"]]

    if strategy == "prefer_abstain":
        # Abstaining sample always wins if available
        abstained = [s for s in evaluated_samples if s["_abstained"]]
        if abstained:
            best_sample = abstained[0]
        elif correct_samples:
            best_sample = min(correct_samples, key=lambda s: s["cot_length"])
        else:
            best_sample = min(incorrect_samples, key=lambda s: s["cot_length"])
    elif strategy == "most_hints":
        candidates = correct_samples if correct_samples else incorrect_samples
        best_sample = max(candidates, key=lambda s: s["_num_hints"])
    else:
        candidates = correct_samples if correct_samples else incorrect_samples
        best_sample = min(candidates, key=lambda s: s["cot_length"])

    # Remove internal fields
    del best_sample["cot_length"]
    del best_sample["_num_hints"]
    del best_sample["_abstained"]

    return best_sample


def generate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
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
    multi_turn: bool | None = None,
    max_turns: int | None = None,
    use_async: bool = False,
    force_hints_distribution: dict[int, float] | None = None,
    force_hints_policy: dict[str, float] | None = None,
    hint_selection: str | None = None,
    helper_model: str | None = None,
    helper_gpu_memory_utilization: float | None = None,
    sample_strategy: str | None = None,
) -> Path:
    """
    Generate model outputs on prompts.

    Supports resumption - if output file exists, skips already-completed indices.
    Saves after each batch for crash recovery.

    Args:
        task_name: Name of task
        model_name: Model to use for generation
        method_name: Method name for auto-derived paths
        run_id: Run identifier for model resolution (used when model_name="sft" or "rl")
        prompts_path: Path to prompts file (default: artifacts/{task}/{method}/prompts/{split}.json)
        output_path: Where to save dataset (default: artifacts/{task}/{method}/datasets/{split}_{model}.json)
        split: Which split to generate from (default: sft)
        retry_incorrect: If True, re-run incorrect examples
        multi_turn: Enable multi-turn generation with hint injection. If None, uses method config.
        max_turns: Maximum turns for multi-turn generation. If None, uses method config (default: 5).
        use_async: Use async generation for optimal throughput.
        force_hints_distribution: Distribution of forced hint counts. Maps number of hints
            to probability, e.g. {1: 0.5, 2: 0.3, 3: 0.2}. Probabilities must sum to 1.
            If None, no hints are forced.
        force_hints_policy: Per-level probability of forcing hints. Maps level number
            to rate, e.g. {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.15, "5": 0.15}.
            If None but force_hints_distribution is set, all examples get forced hints.
        ... generation config ...

    Returns:
        Path to created dataset file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Resolve multi_turn and max_turns from method config (CLI overrides if explicitly set)
    if multi_turn is None:
        multi_turn = method.multi_turn if method else False
    if force_hints_distribution:
        multi_turn = True  # force_hints requires multi_turn
    if max_turns is None:
        max_turns = method.max_turns if method else 6

    # Resolve hint selection strategy (CLI overrides method config)
    if hint_selection is None:
        hint_selection = method.hint_selection if method else "sequential"
    if helper_model is None:
        helper_model = method.helper_model if method else None

    # Create HintSelector if using smart selection
    hint_selector = None
    if hint_selection == "smart" and multi_turn:
        helper_gpu_util = helper_gpu_memory_utilization if helper_gpu_memory_utilization is not None else gpu_memory_utilization
        hint_selector = HintSelector(
            strategy="smart",
            helper_model=helper_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=helper_gpu_util,
        )
        print(f"Smart hint selection enabled (helper: {helper_model}, gpu_util: {helper_gpu_util})")
    elif hint_selection == "smart" and not multi_turn:
        print("Warning: --hint-selection smart requires multi-turn mode, ignoring")

    # Resolve model shortcuts (sft, rl)
    actual_model_name = model_name
    if method is not None:
        if model_name == "sft":
            actual_model_name = str(method.sft_model_path(task_name, run_id))
        elif model_name == "rl":
            actual_model_name = str(method.rl_model_path(task_name, run_id))

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
            # Preserve abstained examples — they are intentional non-answers
            def _is_done(r):
                if r.get("correct", False):
                    return True
                if r.get("metadata", {}).get("abstained", False):
                    return True
                return False
            incorrect_indices = {r["index"] for r in existing_records if not _is_done(r)}
            done_count = len(existing_records) - len(incorrect_indices)
            print(f"Retry mode: {done_count} done (correct + abstained), retrying {len(incorrect_indices)} incorrect")
            completed_indices = {r["index"] for r in existing_records if _is_done(r)}
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
    if use_async:
        print("Async mode enabled (optimal throughput)")
    if multi_turn:
        print(f"Multi-turn mode enabled: max_turns={max_turns}")
    # Compute per-prompt force_hints based on distribution and policy
    if force_hints_distribution:
        import random
        rng = random.Random(42)

        # Build cumulative distribution for sampling hint counts
        hint_counts = sorted(force_hints_distribution.keys())
        hint_probs = [force_hints_distribution[k] for k in hint_counts]
        cum_probs = []
        cumsum = 0.0
        for p in hint_probs:
            cumsum += p
            cum_probs.append(cumsum)

        def sample_hint_count():
            r = rng.random()
            for count, cp in zip(hint_counts, cum_probs):
                if r < cp:
                    return count
            return hint_counts[-1]

        dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(force_hints_distribution.items()))

        if force_hints_policy:
            # Per-level gating: only force for selected examples
            force_hints_list = []
            level_counts = {}
            for p in remaining_prompts:
                level_str = p.get("ground_truth", {}).get("level", "")
                level_num = level_str.replace("Level ", "") if level_str else ""
                rate = force_hints_policy.get(level_num, 0.0)
                if rng.random() < rate:
                    n = sample_hint_count()
                    force_hints_list.append(n)
                    level_counts[level_str] = level_counts.get(level_str, 0) + 1
                else:
                    force_hints_list.append(0)
            forced_total = sum(1 for f in force_hints_list if f > 0)
            print(f"Force hints policy: {forced_total}/{len(remaining_prompts)} examples will get forced hints")
            print(f"  Distribution: {dist_str}")
            for level in sorted(level_counts.keys()):
                print(f"  {level}: {level_counts[level]} forced")
        else:
            # All examples get forced hints sampled from distribution
            force_hints_list = [sample_hint_count() for _ in remaining_prompts]
            print(f"Force hints enabled for all {len(remaining_prompts)} examples")
            print(f"  Distribution: {dist_str}")
    else:
        force_hints_list = [0] * len(remaining_prompts)

    # For async multi-turn, process all remaining prompts at once
    if multi_turn and use_async:
        import asyncio

        all_prompts = [p["prompt"] for p in remaining_prompts]
        all_ground_truths = [p["ground_truth"] for p in remaining_prompts]

        print(f"Running async generation on {len(all_prompts)} prompts...")
        async_generator = AsyncGenerator(config)
        all_results = asyncio.run(
            async_generator.generate_with_hints_async(
                all_prompts,
                all_ground_truths,
                max_turns=max_turns,
                force_hints=force_hints_list,
                hint_selector=hint_selector,
            )
        )

        # Process all results
        sample_strategy = sample_strategy or ("most_hints" if multi_turn else "shortest_cot")
        for prompt_data, gen_samples in zip(remaining_prompts, all_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            if num_samples > 1:
                best_sample = select_best_sample(gen_samples, task, primitive, strategy=sample_strategy)
            else:
                sample = gen_samples[0]
                is_correct, meta = task.check_correctness(primitive, sample["text"])
                best_sample = {
                    **sample,
                    "correct": is_correct,
                    "metadata": meta,
                }

            record = {
                "index": prompt_data["index"],
                "variant": prompt_data.get("variant", "unknown"),
                "prompt": prompt_data["prompt"],
                "generation": best_sample["text"],
                "correct": best_sample["correct"],
                "finish_reason": best_sample["finish_reason"],
                "token_count": best_sample["token_count"],
                "metadata": best_sample["metadata"],
                "extracted_hints": extract_generation_hints(best_sample["text"]),
            }

            if "num_hints" in best_sample:
                record["num_hints"] = best_sample["num_hints"]
            if "turns" in best_sample:
                record["turns"] = best_sample["turns"]
            if "total_token_count" in best_sample:
                record["total_token_count"] = best_sample["total_token_count"]

            records_by_index[prompt_data["index"]] = record

        # Save final results
        records = sorted(records_by_index.values(), key=lambda r: r["index"])
        save_json(output_path, records)

        correct = sum(1 for r in records if r["correct"])
        print(f"Async generation complete: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
        print(f"Saved dataset to {output_path}")
        if hint_selector is not None:
            hint_selector.shutdown()
        return output_path

    # Async regular generation (non-multi-turn)
    if use_async and not multi_turn:
        import asyncio

        all_prompts = [p["prompt"] for p in remaining_prompts]

        print(f"Running async generation on {len(all_prompts)} prompts...")
        async_generator = AsyncGenerator(config)
        all_results = asyncio.run(
            async_generator.generate_async(all_prompts, num_samples=num_samples)
        )

        # Process all results
        for prompt_data, gen_samples in zip(remaining_prompts, all_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            if num_samples > 1:
                best_sample = select_best_sample(gen_samples, task, primitive)
            else:
                sample = gen_samples[0]
                is_correct, meta = task.check_correctness(primitive, sample["text"])
                best_sample = {
                    **sample,
                    "correct": is_correct,
                    "metadata": meta,
                }

            record = {
                "index": prompt_data["index"],
                "variant": prompt_data.get("variant", "unknown"),
                "prompt": prompt_data["prompt"],
                "generation": best_sample["text"],
                "correct": best_sample["correct"],
                "finish_reason": best_sample["finish_reason"],
                "token_count": best_sample["token_count"],
                "metadata": best_sample["metadata"],
                "extracted_hints": extract_generation_hints(best_sample["text"]),
            }

            records_by_index[prompt_data["index"]] = record

        # Save final results
        records = sorted(records_by_index.values(), key=lambda r: r["index"])
        save_json(output_path, records)

        correct = sum(1 for r in records if r["correct"])
        print(f"Async generation complete: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
        print(f"Saved dataset to {output_path}")
        return output_path

    # Standard batch processing (sync)
    for batch_idx in range(0, len(remaining_prompts), batch_size):
        batch_prompts_data = remaining_prompts[batch_idx:batch_idx + batch_size]
        batch_prompts = [p["prompt"] for p in batch_prompts_data]

        # Generate batch
        if multi_turn:
            # Multi-turn generation with hint injection (sync)
            # Duplicate prompts for num_samples > 1 (generate_with_hints produces 1 trajectory each)
            if num_samples > 1:
                expanded_prompts = [p for p in batch_prompts for _ in range(num_samples)]
                expanded_gts = [p["ground_truth"] for p in batch_prompts_data for _ in range(num_samples)]
                expanded_force = [fh for fh in force_hints_list[batch_idx:batch_idx + batch_size] for _ in range(num_samples)]
                expanded_results = generator.generate_with_hints(
                    expanded_prompts,
                    expanded_gts,
                    max_turns=max_turns,
                    force_hints=expanded_force,
                    hint_selector=hint_selector,
                )
                # Group back: every num_samples consecutive results belong to the same prompt
                batch_results = [
                    [r[0] for r in expanded_results[i * num_samples:(i + 1) * num_samples]]
                    for i in range(len(batch_prompts_data))
                ]
            else:
                batch_ground_truths = [p["ground_truth"] for p in batch_prompts_data]
                batch_force_hints = force_hints_list[batch_idx:batch_idx + batch_size]
                batch_results = generator.generate_with_hints(
                    batch_prompts,
                    batch_ground_truths,
                    max_turns=max_turns,
                    force_hints=batch_force_hints,
                    hint_selector=hint_selector,
                )
        else:
            batch_results = generator.generate(batch_prompts)

        # Process results
        sample_strategy = sample_strategy or ("most_hints" if multi_turn else "shortest_cot")
        batch_correct = 0
        for prompt_data, gen_samples in zip(batch_prompts_data, batch_results):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            # If num_samples > 1, select best sample; otherwise use single sample
            if num_samples > 1:
                best_sample = select_best_sample(gen_samples, task, primitive, strategy=sample_strategy)
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
                "extracted_hints": extract_generation_hints(best_sample["text"]),
            }

            # Add multi-turn metadata if available
            if "num_hints" in best_sample:
                record["num_hints"] = best_sample["num_hints"]
            if "turns" in best_sample:
                record["turns"] = best_sample["turns"]
            if "total_token_count" in best_sample:
                record["total_token_count"] = best_sample["total_token_count"]

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

    if hint_selector is not None:
        hint_selector.shutdown()

    return output_path


def evaluate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
    split: str = "eval",
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,  # Greedy for eval
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    verbose: bool = False,
    multi_turn: bool | None = None,
    max_turns: int | None = None,
    use_async: bool = False,
    hint_selection: str | None = None,
    helper_model: str | None = None,
    helper_gpu_memory_utilization: float | None = None,
) -> Path:
    """
    Evaluate a model on prompts and compute metrics.

    Args:
        task_name: Name of task
        model_name: Model to evaluate (can be "sft" or "rl" to use method's model paths)
        method_name: Method name for auto-derived paths
        run_id: Run identifier for model resolution (used when model_name="sft" or "rl")
        prompts_path: Path to eval prompts (default: artifacts/{task}/{method}/prompts/eval.json)
        output_path: Where to save results (default: artifacts/{task}/{method}/results/eval_{model}.json)
        multi_turn: Enable multi-turn generation with hint injection. If None, uses method config.
        max_turns: Maximum turns for multi-turn generation. If None, uses method config (default: 5).
        use_async: Use async generation for optimal throughput.
        ... generation config ...

    Returns:
        Path to results file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Resolve multi_turn and max_turns from method config (CLI overrides if explicitly set)
    if multi_turn is None:
        multi_turn = method.multi_turn if method else False
    if max_turns is None:
        max_turns = method.max_turns if method else 6

    # Resolve hint selection strategy (CLI overrides method config)
    if hint_selection is None:
        hint_selection = method.hint_selection if method else "sequential"
    if helper_model is None:
        helper_model = method.helper_model if method else None

    # Create HintSelector if using smart selection
    hint_selector = None
    if hint_selection == "smart" and multi_turn:
        helper_gpu_util = helper_gpu_memory_utilization if helper_gpu_memory_utilization is not None else gpu_memory_utilization
        hint_selector = HintSelector(
            strategy="smart",
            helper_model=helper_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=helper_gpu_util,
        )
        print(f"Smart hint selection enabled (helper: {helper_model}, gpu_util: {helper_gpu_util})")

    # Resolve model shortcuts (sft, rl)
    actual_model_name = model_name
    if method is not None:
        if model_name == "sft":
            actual_model_name = str(method.sft_model_path(task_name, run_id))
        elif model_name == "rl":
            actual_model_name = str(method.rl_model_path(task_name, run_id))

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
        # Include run_id in output filename for SFT/RL models
        output_model_name = model_name
        if model_name in ("sft", "rl") and run_id and run_id != "default":
            output_model_name = f"{model_name}_{run_id}"
        from pipeline.core.method import model_short_name
        model_slug = model_short_name(output_model_name)
        output_path = method.results_dir(task_name) / f"{split}_{model_slug}.json"

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

    print(f"Evaluating {actual_model_name}...")
    if use_async:
        print("Async mode enabled (optimal throughput)")
    if multi_turn:
        print(f"Multi-turn mode enabled: max_turns={max_turns}")

    # Async multi-turn generation
    if multi_turn and use_async:
        import asyncio

        ground_truths = [p["ground_truth"] for p in prompts_data]

        print(f"Running async multi-turn generation on {len(prompts)} prompts...")
        async_generator = AsyncGenerator(config)
        generations = asyncio.run(
            async_generator.generate_with_hints_async(
                prompts,
                ground_truths,
                max_turns=max_turns,
                hint_selector=hint_selector,
            )
        )

    # Async regular generation
    elif use_async and not multi_turn:
        import asyncio

        print(f"Running async generation on {len(prompts)} prompts...")
        async_generator = AsyncGenerator(config)
        generations = asyncio.run(
            async_generator.generate_async(prompts, num_samples=1)
        )

    # Sync multi-turn generation
    elif multi_turn:
        ground_truths = [p["ground_truth"] for p in prompts_data]
        generations = generator.generate_with_hints_batched(
            prompts,
            ground_truths,
            max_turns=max_turns,
            hint_selector=hint_selector,
        )

    # Sync regular generation
    else:
        def progress_callback(batch_idx, _results):
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            print(f"Batch {batch_idx + 1}/{total_batches} complete")

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
            "level": prompt_data.get("ground_truth", {}).get("level", "unknown"),
            "correct": is_correct,
            "finish_reason": finish_reason,
            "token_count": token_count,
            "generation": gen_result["text"],
            "predicted_answer": meta.get("predicted_answer"),
            "ground_truth": prompt_data["ground_truth"],
            "error": meta.get("error"),
            **{k: v for k, v in meta.items() if k not in ("predicted_answer", "error")},
        }

        # Add multi-turn specific fields
        if "num_hints" in gen_result:
            detail["num_hints"] = gen_result["num_hints"]
        if "turns" in gen_result:
            detail["turns"] = gen_result["turns"]
        if "total_token_count" in gen_result:
            detail["total_token_count"] = gen_result["total_token_count"]
        if "hint_selections" in gen_result:
            detail["hint_selections"] = gen_result["hint_selections"]

        details.append(detail)

    # Compute metrics using task-specific logic
    metrics = task.compute_metrics(details)

    # Compute hint metrics if multi-turn
    hint_metrics = None
    if multi_turn:
        hint_metrics = compute_hint_metrics(details)

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
            "multi_turn": multi_turn,
            "max_turns": max_turns if multi_turn else None,
        },
        "metrics": metrics,
        "details": details,
    }

    # Add hint metrics if available
    if hint_metrics:
        results["hint_metrics"] = hint_metrics

    # Print summary using task-specific formatting
    print(task.format_metrics(metrics, model_name))

    # Print hint analysis if multi-turn
    if hint_metrics:
        print(format_hint_metrics(hint_metrics, details))

    # Print per-problem hint selection details if smart selection was used
    if hint_selection == "smart" and multi_turn:
        print("\n--- Hint Selection Details ---")
        for d in details:
            sels = d.get("hint_selections", [])
            if sels:
                level = d.get("level", "?")
                correct = "Y" if d.get("correct") else "N"
                idx = d.get("index", "?")
                sel_strs = [
                    f"turn {s['turn']}: {s['prev_last_given']}->{s['new_last_given']} ({s['type']})"
                    for s in sels
                ]
                print(f"  [{idx}] level={level} correct={correct} hints={d.get('num_hints', 0)} | {', '.join(sel_strs)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    if hint_selector is not None:
        hint_selector.shutdown()

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
