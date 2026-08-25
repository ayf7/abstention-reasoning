"""
Comparison commands - multi-sample comparison between models.

Takes two models (simple + abstention), generates N samples per eval problem
from each using VLLM, then plots per-problem accuracy vs abstention rate.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import AsyncGenerator, GenerationConfig
from pipeline.core.method import ARTIFACTS_ROOT, Method
from pipeline.core.utils import model_short_name
from pipeline.tasks import get_task


async def _generate_all_samples_async(config, prompts_data, task, label="model"):
    """Generate N samples per problem using async engine and check correctness."""
    generator = AsyncGenerator(config)

    prompts = [p["prompt"] for p in prompts_data]
    print(f"  [{label}] Generating {config.num_samples} samples for {len(prompts)} problems...")

    all_samples = await generator.generate_async(prompts, num_samples=config.num_samples)

    # Shut down engine to free GPU memory for next model
    if generator._engine is not None:
        generator._engine = None
    generator._tokenizer = None

    all_records = []
    for prompt_data, samples in zip(prompts_data, all_samples):
        primitive = prompt_data.get("ground_truth", prompt_data)

        sample_records = []
        for sample in samples:
            is_correct, meta = task.check_correctness(primitive, sample["text"])
            sample_records.append({
                "correct": is_correct,
                "abstained": meta.get("abstained", False) if isinstance(meta, dict) else False,
                "finish_reason": sample["finish_reason"],
                "token_count": sample["token_count"],
            })

        all_records.append({
            "index": prompt_data["index"],
            "variant": prompt_data.get("variant", "unknown"),
            "ground_truth": primitive if "ground_truth" not in prompt_data else prompt_data["ground_truth"],
            "n_samples": len(samples),
            "n_correct": sum(1 for s in sample_records if s["correct"]),
            "n_abstained": sum(1 for s in sample_records if s["abstained"]),
            "samples": sample_records,
        })

    total_correct = sum(r["n_correct"] for r in all_records)
    total_samples = sum(r["n_samples"] for r in all_records)
    print(f"  [{label}] Done: {total_correct}/{total_samples} correct samples")

    return all_records


def _resolve_model_path(model_name, method, task_name, run_id):
    """Resolve 'sft'/'rl' aliases to actual paths."""
    if model_name in ("sft", "rl") and method:
        if model_name == "sft":
            return str(method.sft_model_path(task_name, run_id))
        else:
            return str(method.rl_model_path(task_name, run_id))
    return model_name


def compare_sampling(
    task_name: str,
    simple_model: str,
    abstention_model: str,
    simple_prompts_path: Path,
    abstention_prompts_path: Path,
    output_path: Path | None = None,
    simple_method: str | None = None,
    abstention_method: str | None = None,
    simple_run_id: str | None = None,
    abstention_run_id: str | None = None,
    num_samples: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    group_by: str = "variant",
) -> Path:
    """
    Generate N samples from two models on their respective prompts, then compare.

    Produces a scatter plot of per-problem simple accuracy vs abstention rate,
    plus a JSON file with all per-problem statistics.

    Args:
        task_name: Task name
        simple_model: Simple/baseline model name or path (or 'sft'/'rl')
        abstention_model: Abstention model name or path (or 'sft'/'rl')
        simple_prompts_path: Path to prompts for simple model
        abstention_prompts_path: Path to prompts for abstention model
        output_path: Base path for output files (.json and .pdf)
        simple_method: Method name for resolving simple model path
        abstention_method: Method name for resolving abstention model path
        simple_run_id: Run ID for simple model
        abstention_run_id: Run ID for abstention model
        num_samples: Samples per problem per model
        max_new_tokens: Max new tokens
        temperature: Sampling temperature
        top_p: Top-p
        tensor_parallel_size: Tensor parallel size
        gpu_memory_utilization: GPU memory utilization
        group_by: Field to group/color scatter points by

    Returns:
        Path to results JSON
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = get_task(task_name)

    # Resolve model paths
    simple_method_obj = Method.load(simple_method, task_name) if simple_method else None
    abstention_method_obj = Method.load(abstention_method, task_name) if abstention_method else None

    actual_simple = _resolve_model_path(simple_model, simple_method_obj, task_name, simple_run_id)
    actual_abstention = _resolve_model_path(abstention_model, abstention_method_obj, task_name, abstention_run_id)

    # Default output path
    if output_path is None:
        simple_tag = f"{simple_method}_{model_short_name(actual_simple)}" if simple_method else model_short_name(actual_simple)
        abstention_tag = f"{abstention_method}_{model_short_name(actual_abstention)}" if abstention_method else model_short_name(actual_abstention)
        output_path = ARTIFACTS_ROOT / task_name / f"comparison_{simple_tag}_vs_{abstention_tag}.json"

    print(f"=== Sampling Comparison ===")
    print(f"Task: {task_name}")
    print(f"Simple model: {actual_simple}")
    print(f"Abstention model: {actual_abstention}")
    print(f"Simple prompts: {simple_prompts_path}")
    print(f"Abstention prompts: {abstention_prompts_path}")
    print(f"Samples per problem: {num_samples}")
    print(f"Output: {output_path}")
    print(f"===========================")

    # Load prompts
    simple_prompts_data = load_json(simple_prompts_path)
    abstention_prompts_data = load_json(abstention_prompts_path)
    print(f"Loaded {len(simple_prompts_data)} simple prompts, {len(abstention_prompts_data)} abstention prompts")

    # Verify same indices
    simple_indices = {p["index"] for p in simple_prompts_data}
    abstention_indices = {p["index"] for p in abstention_prompts_data}
    if simple_indices != abstention_indices:
        missing_in_abst = simple_indices - abstention_indices
        missing_in_simple = abstention_indices - simple_indices
        if missing_in_abst:
            print(f"Warning: {len(missing_in_abst)} indices in simple prompts missing from abstention prompts")
        if missing_in_simple:
            print(f"Warning: {len(missing_in_simple)} indices in abstention prompts missing from simple prompts")

    import asyncio

    base_config = GenerationConfig(
        model_name="",  # placeholder
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=None,
    )

    # Generate from simple model (async)
    print(f"\n--- Simple model ---")
    simple_config = GenerationConfig(**{**base_config.__dict__, "model_name": actual_simple})
    simple_records = asyncio.run(
        _generate_all_samples_async(simple_config, simple_prompts_data, task, label="simple")
    )

    # Clean up VLLM engine and free GPU memory before loading second model
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Generate from abstention model (async)
    print(f"\n--- Abstention model ---")
    abstention_config = GenerationConfig(**{**base_config.__dict__, "model_name": actual_abstention})
    abstention_records = asyncio.run(
        _generate_all_samples_async(abstention_config, abstention_prompts_data, task, label="abstention")
    )

    # Build per-problem comparison
    simple_by_index = {r["index"]: r for r in simple_records}
    abstention_by_index = {r["index"]: r for r in abstention_records}

    records = []
    for idx in sorted(simple_by_index.keys()):
        s = simple_by_index[idx]
        a = abstention_by_index[idx]

        gt = s.get("ground_truth", {})
        group = s.get(group_by, gt.get(group_by, "unknown"))
        a_answered = a["n_samples"] - a["n_abstained"]

        record = {
            "index": idx,
            group_by: group,
            "simple_accuracy": s["n_correct"] / s["n_samples"],
            "abstention_rate": a["n_abstained"] / a["n_samples"],
            "abstention_accuracy_when_answering": (
                a["n_correct"] / a_answered if a_answered > 0 else None
            ),
            "simple_n": s["n_samples"],
            "simple_correct": s["n_correct"],
            "abstention_n": a["n_samples"],
            "abstention_correct": a["n_correct"],
            "abstention_abstained": a["n_abstained"],
        }
        # Store extra ground_truth fields (e.g. level) for grouping
        for extra_key in ("level",):
            if extra_key in gt and extra_key not in record:
                record[extra_key] = gt[extra_key]
        records.append(record)

    # Summary by group
    by_group = defaultdict(list)
    for r in records:
        by_group[r[group_by]].append(r)

    print(f"\nPer-{group_by} summary:")
    print(f"  {group_by:<25s}  {'N':>4s}  {'SimpleAcc':>9s}  {'AbstRate':>8s}  {'AbstAcc':>7s}")
    for group, group_records in sorted(by_group.items()):
        n = len(group_records)
        avg_simple = sum(r["simple_accuracy"] for r in group_records) / n
        avg_abstain = sum(r["abstention_rate"] for r in group_records) / n
        answering = [r for r in group_records if r["abstention_accuracy_when_answering"] is not None]
        avg_abstain_acc = sum(r["abstention_accuracy_when_answering"] for r in answering) / len(answering) if answering else float("nan")
        print(f"  {str(group):<25s}  {n:4d}  {avg_simple:9.3f}  {avg_abstain:8.3f}  {avg_abstain_acc:7.3f}")

    avg_simple = sum(r["simple_accuracy"] for r in records) / len(records)
    avg_abstain = sum(r["abstention_rate"] for r in records) / len(records)
    print(f"\n  Overall simple accuracy:  {avg_simple:.3f}")
    print(f"  Overall abstention rate:  {avg_abstain:.3f}")

    # Heatmap: count problems at each (accuracy, abstention_rate) cell
    import numpy as np
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    n_bins = num_samples + 1  # 0/N through N/N
    ticks = np.arange(n_bins) / num_samples

    def _build_heatmap(recs):
        h = np.zeros((n_bins, n_bins), dtype=int)
        for r in recs:
            xi = round(r["simple_accuracy"] * num_samples)
            yi = round(r["abstention_rate"] * num_samples)
            h[yi, xi] += 1
        return h

    def _plot_heatmap(ax, heatmap, title, global_vmax, show_xlabel=True, show_ylabel=True, annotate_fontsize=5):
        heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
        im = ax.imshow(heatmap_masked, origin="lower", cmap="YlOrRd", aspect="equal",
                       norm=LogNorm(vmin=1, vmax=global_vmax),
                       extent=[-0.5 / num_samples, 1 + 0.5 / num_samples,
                               -0.5 / num_samples, 1 + 0.5 / num_samples])
        for yi in range(n_bins):
            for xi in range(n_bins):
                if heatmap[yi, xi] > 0:
                    ax.text(xi / num_samples, yi / num_samples, str(heatmap[yi, xi]),
                            ha="center", va="center", fontsize=annotate_fontsize,
                            color="white" if heatmap[yi, xi] > global_vmax * 0.3 else "black")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ticks[::4])
        ax.set_yticks(ticks[::4])
        ax.set_xticklabels([f"{v:.0%}" for v in ticks[::4]], fontsize=7)
        ax.set_yticklabels([f"{v:.0%}" for v in ticks[::4]], fontsize=7)
        if show_xlabel:
            ax.set_xlabel(f"Simple Accuracy ({num_samples}s)", fontsize=8)
        if show_ylabel:
            ax.set_ylabel(f"Abstention Rate ({num_samples}s)", fontsize=8)
        return im

    # Build heatmaps
    heatmap_all = _build_heatmap(records)
    groups = sorted(by_group.keys())
    group_heatmaps = {g: _build_heatmap(by_group[g]) for g in groups}
    global_vmax = heatmap_all.max()

    # Layout: main plot on top, per-group subplots on bottom (3 per row)
    n_groups = len(groups)
    n_rows_bottom = (n_groups + 2) // 3  # ceil(n_groups / 3)

    fig = plt.figure(figsize=(12, 8 + 4 * n_rows_bottom))
    gs = gridspec.GridSpec(1 + n_rows_bottom, 3, figure=fig,
                           height_ratios=[2] + [1] * n_rows_bottom,
                           hspace=0.35, wspace=0.3)

    # Main plot (spans all 3 columns)
    ax_main = fig.add_subplot(gs[0, :])
    _plot_heatmap(ax_main, heatmap_all, f"Accuracy vs Abstention Rate — {task_name} (all)", global_vmax)
    ax_main.set_xticks(ticks[::2])
    ax_main.set_yticks(ticks[::2])
    ax_main.set_xticklabels([f"{v:.0%}" for v in ticks[::2]], fontsize=8)
    ax_main.set_yticklabels([f"{v:.0%}" for v in ticks[::2]], fontsize=8)
    ax_main.set_xlabel(f"Simple Model Accuracy ({num_samples} samples)")
    ax_main.set_ylabel(f"Abstention Rate ({num_samples} samples)")

    # Per-group subplots
    for i, g in enumerate(groups):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        n_in_group = len(by_group[g])
        _plot_heatmap(ax, group_heatmaps[g], f"{g} (N={n_in_group})", global_vmax,
                      show_xlabel=(row == n_rows_bottom), show_ylabel=(col == 0),
                      annotate_fontsize=4)

    plot_path = output_path.with_suffix(".pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")

    # Save JSON
    results = {
        "task": task_name,
        "simple_model": actual_simple,
        "abstention_model": actual_abstention,
        "simple_prompts": str(simple_prompts_path),
        "abstention_prompts": str(abstention_prompts_path),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "group_by": group_by,
        "details": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"Saved results to {output_path}")

    return output_path
