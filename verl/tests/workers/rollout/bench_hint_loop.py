#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Round vs continuous hint scheduling on a real model and a real GPU.

Needs a GPU; not collected by pytest. It drives the same HintLoop and the same
_VLLMLoopEngine the trainer uses, so what it measures is the shipped path.

    python tests/workers/rollout/bench_hint_loop.py \\
        --model artifacts/competition_math_v2/hint/models/sft/qwen2.5-3b-up4x/model \\
        --prompts artifacts/competition_math_v2/hint/prompts/eval.json \\
        --num-prompts 64 --n 8 --nested

Two questions, two modes:

  --temperature 0  Do the drivers agree? Greedy makes sampling deterministic,
                   so a run repeated under the same driver measures the
                   numerical floor (batch composition changes the kernels'
                   reduction order) and the cross-driver rate is read against
                   it. Scheduling is not supposed to move behaviour at all.

  --temperature 1  Does the throughput hold? Reports policy tokens decoded per
                   wall-clock second, which is the number the change exists to
                   move, plus the turn-level distributions that must not move.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.workers.rollout.hint_loop import HintLoop, HintLoopConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _VLLMLoopEngine


def build_prompts(path, tokenizer, num_prompts, n):
    """Prompt token ids and hint lists, built the way RLHFDataset builds them."""
    with open(path) as f:
        rows = json.load(f)
    rows = rows[:num_prompts]
    prompt_ids, hints = [], []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=False, continue_final_message=True, tokenize=False
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        row_hints = list(row["ground_truth"].get("hint_exprs", []))
        for _ in range(n):
            prompt_ids.append(list(ids))
            hints.append(row_hints)
    return prompt_ids, hints


def summarize(states, tokenizer, elapsed, engine):
    """Everything the drivers must agree on, plus the throughput they need not."""
    decoded = 0
    for st in states:
        for seg, seg_mask in zip(st.partials[1:], st.masks):
            decoded += sum(seg_mask)
    texts = [
        tokenizer.decode([t for seg in st.partials[1:] for t in seg]) for st in states
    ]
    lengths = [st.response_len for st in states]
    return {
        "wall_s": round(elapsed, 2),
        "policy_tokens": decoded,
        "decode_tok_s": round(decoded / elapsed, 1),
        "rollouts": len(states),
        "generate_calls": engine.calls,
        "batch_sizes": engine.batch_sizes,
        "num_hints_mean": round(statistics.fmean(st.num_hints for st in states), 4),
        "num_hints_hist": dict(sorted(Counter(int(st.num_hints) for st in states).items())),
        "asked_frac": round(statistics.fmean(st.num_hints > 0 for st in states), 4),
        "truncated_frac": round(statistics.fmean(st.truncated for st in states), 4),
        "forced_frac": round(statistics.fmean(st.forced for st in states), 4),
        "malformed_frac": round(statistics.fmean(float(st.malformed) for st in states), 4),
        "turns_mean": round(statistics.fmean(st.turns for st in states), 4),
        "response_len_mean": round(statistics.fmean(lengths), 1),
        "response_len_p90": sorted(lengths)[int(0.9 * (len(lengths) - 1))],
        "texts": texts,
    }


class _CountingEngine(_VLLMLoopEngine):
    """_VLLMLoopEngine plus the call trace the benchmark reports."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.batch_sizes = []
        self.live = 0

    def generate_batch(self, prompts, max_tokens, mode="turn"):
        if mode == "turn":
            self.calls += 1
            self.batch_sizes.append(len(prompts))
        return super().generate_batch(prompts, max_tokens, mode=mode)

    def submit(self, request_id, prompt, max_tokens):
        super().submit(request_id, prompt, max_tokens)
        self.live += 1
        self.batch_sizes.append(self.live)

    def step(self):
        self.calls += 1
        finished = super().step()
        self.live -= len(finished)
        return finished


def run_once(llm, tokenizer, prompt_ids, hints, driver, config, turn_params, answer_params):
    # Prefix-cache hits from an earlier run would flatter whichever driver goes
    # second, and by a lot -- every turn resubmits an accumulated sequence.
    llm.llm_engine.reset_prefix_cache()

    engine = _CountingEngine(llm, turn_params, answer_params)
    loop = HintLoop(engine=engine, tokenizer=tokenizer, config=config, log=None)
    states = loop.make_states(prompt_ids, hints)

    start = time.perf_counter()
    if driver == "continuous":
        loop.run_continuous(states)
    else:
        loop.run_rounds(states)
    loop.force_answers(states)
    elapsed = time.perf_counter() - start

    return summarize(states, tokenizer, elapsed, engine)


def compare(label, a, b, keys):
    agree = sum(x == y for x, y in zip(a["texts"], b["texts"]))
    print(f"\n{label}: {agree}/{len(a['texts'])} rollouts byte-identical "
          f"({100 * agree / len(a['texts']):.1f}%)")
    for key in keys:
        if a[key] != b[key]:
            print(f"  {key}: {a[key]} -> {b[key]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--n", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--max-hints", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--nested", action="store_true", help="requests live inside <think>")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--think-budget", type=int, default=2048)
    ap.add_argument("--answer-budget", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--drivers", default="rounds,continuous,rounds,continuous",
                    help="comma-separated run order; repeats measure run-to-run spread")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids, hints = build_prompts(args.prompts, tokenizer, args.num_prompts, args.n)
    print(f"{len(prompt_ids)} rollouts, prompt tokens "
          f"{min(len(p) for p in prompt_ids)}-{max(len(p) for p in prompt_ids)}")

    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )

    # Mirrors vLLMRollout.__init__ for allow_hint.
    turn_params = SamplingParams(
        n=1,
        logprobs=0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.think_budget,
        include_stop_str_in_output=True,
        detokenize=True,
        stop=["</answer>", "</think>\n\n<abstain>", "</request>", "</response>"],
    )
    answer_params = SamplingParams(
        n=1,
        logprobs=0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.answer_budget,
        include_stop_str_in_output=False,
        detokenize=True,
        stop=["</answer>"],
    )
    config = HintLoopConfig(
        max_hints=args.max_hints,
        max_turns=args.max_turns,
        max_model_len=args.max_model_len,
        max_response_len=args.think_budget,
        answer_budget=args.answer_budget,
        nested_request=args.nested,
    )

    keys = ["wall_s", "decode_tok_s", "policy_tokens", "generate_calls", "num_hints_mean",
            "num_hints_hist", "asked_frac", "truncated_frac", "forced_frac",
            "malformed_frac", "turns_mean", "response_len_mean", "response_len_p90"]

    runs = []
    for driver in args.drivers.split(","):
        driver = driver.strip()
        result = run_once(llm, tokenizer, prompt_ids, hints, driver,
                          config, turn_params, answer_params)
        result["driver"] = driver
        runs.append(result)
        print(f"\n=== run {len(runs)}: {driver} ===")
        for key in keys:
            print(f"  {key:20s} {result[key]}")

    print("\n" + "=" * 70)
    print(f"{'run':>4} {'driver':>12} {'wall_s':>9} {'decode_tok/s':>13} {'calls':>7} "
          f"{'hints':>7} {'trunc':>7} {'forced':>7}")
    for i, r in enumerate(runs, 1):
        print(f"{i:>4} {r['driver']:>12} {r['wall_s']:>9} {r['decode_tok_s']:>13} "
              f"{r['generate_calls']:>7} {r['num_hints_mean']:>7} "
              f"{r['truncated_frac']:>7} {r['forced_frac']:>7}")

    by_driver = {}
    for r in runs:
        by_driver.setdefault(r["driver"], []).append(r)
    for driver, group in by_driver.items():
        rates = [r["decode_tok_s"] for r in group]
        print(f"{driver:>12}: decode tok/s {rates} mean {statistics.fmean(rates):.1f}")
    if "rounds" in by_driver and "continuous" in by_driver:
        base = statistics.fmean(r["decode_tok_s"] for r in by_driver["rounds"])
        new = statistics.fmean(r["decode_tok_s"] for r in by_driver["continuous"])
        print(f"\ncontinuous / rounds decode throughput: {new / base:.3f}x")

    if args.temperature == 0:
        for driver, group in by_driver.items():
            if len(group) > 1:
                compare(f"{driver} run 1 vs run 2 (numerical floor)", group[0], group[1], keys)
        if "rounds" in by_driver and "continuous" in by_driver:
            compare("rounds vs continuous", by_driver["rounds"][0],
                    by_driver["continuous"][0], keys)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
