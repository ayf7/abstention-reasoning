# abstention-reasoning

Training language models to recognize when they cannot solve a problem and ask for a hint, rather than committing to an answer every time.

Models are trained in two stages: supervised fine-tuning (SFT) on chain-of-thought traces, then reinforcement learning with a reward function that pays for correct answers and charges for each hint taken. RL runs on a modified copy of [verl](https://github.com/volcengine/verl), vendored in `verl/`.

Everything is driven by one CLI:

```bash
python -m pipeline <command> [--task TASK] [--method METHOD] ...
```

## Install

Requires Python >= 3.10.12 and a CUDA GPU — vLLM backs all generation.

```bash
pip install -e .        # pipeline + deps (torch, transformers, vllm, trl, ...)
pip install -e verl/    # RL trainer
```

`verl/` is vendored in-tree, not a submodule. It carries local modifications (custom reward functions, multi-turn rollout hooks) and should not be swapped for an upstream checkout.

## Quickstart

The full `baseline` → `method_b` pipeline on `countdown` with a 1.5B student and Qwen3-14B as the teacher. Every path is written out: `--method` plus `--run-id` would derive the same ones, but spelling them makes the data flow between stages visible.

### 0. Problems — shared by every method

```bash
python -m pipeline create_primitives --task countdown --num-puzzles 5000 --seed 42 \
    --output artifacts/countdown/problems/primitives.json

python -m pipeline create_partitions --task countdown --seed 42 \
    --primitives artifacts/countdown/problems/primitives.json \
    --output artifacts/countdown/problems
```

`create_partitions` writes one file per split (`sft_whole`, `sft_train`, `sft_val`, `rl_train`, `rl_val`, `eval`). Every method formats those same files, so the split is fixed once and recorded rather than recomputed per method.

### 1. Baseline — answer directly, never ask for a hint

```bash
# Apply the baseline template to each partition
python -m pipeline create_prompts --task countdown --method baseline --split all \
    --primitives artifacts/countdown/problems/primitives.json \
    --output artifacts/countdown/problems_with_format

# Teacher rollouts -> SFT data
python -m pipeline generate --task countdown --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_whole \
    --prompts artifacts/countdown/problems_with_format/sft_whole__baseline.json \
    --output artifacts/countdown/sft_datasets/sft_whole__baseline.json

# SFT the student on those rollouts
python -m pipeline train_sft --task countdown --method baseline --run-id qwen2.5-1.5b \
    --base-model Qwen/Qwen2.5-1.5B \
    --dataset artifacts/countdown/sft_datasets/sft_whole__baseline.json \
    --output artifacts/countdown/models/baseline_sft/qwen2.5-1.5b/model

# RL from that SFT checkpoint -> the baseline model
python -m pipeline train_rl --task countdown --method baseline --run-id qwen2.5-1.5b \
    --sft-model artifacts/countdown/models/baseline_sft/qwen2.5-1.5b/model \
    --train-prompts artifacts/countdown/problems_with_format/rl_train__baseline.parquet \
    --val-prompts artifacts/countdown/problems_with_format/rl_val__baseline.parquet \
    --output artifacts/countdown/models/baseline_models/qwen2.5-1.5b/model

python -m pipeline evaluate --task countdown --method baseline --model rl --async \
    --run-id qwen2.5-1.5b --split eval \
    --prompts artifacts/countdown/problems_with_format/eval__baseline.json \
    --output artifacts/countdown/models/baseline_models/qwen2.5-1.5b/evals/eval.json
```

### 2. Method B — request hints mid-reasoning

Method B starts from the *baseline model*, not from the base checkpoint: reasoning is sharpened first, hint-seeking second.

```bash
# Apply the method_b template to the same partitions
python -m pipeline create_prompts --task countdown --method method_b --split all \
    --primitives artifacts/countdown/problems/primitives.json \
    --output artifacts/countdown/problems_with_format

# Probe: how often does the teacher solve each problem unaided?
python -m pipeline generate --task countdown --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_whole --no-hints --num-samples 8 \
    --prompts artifacts/countdown/problems_with_format/sft_whole__baseline.json \
    --output artifacts/countdown/.scratch/sft_whole__baseline__probe.json

# Schedule hints from that probe -- harder problems get more -- and oversample
# until half the set is correct. The profile is derived in memory; only the
# dataset is written.
python -m pipeline generate --task countdown --method method_b --async \
    --model Qwen/Qwen3-14B --split sft_whole \
    --hint-schedule artifacts/countdown/.scratch/sft_whole__baseline__probe.json \
    --hint-target-fraction 0.5 --max-hints 4 --target-correct-rate 0.5 \
    --prompts artifacts/countdown/problems_with_format/sft_whole__method_b.json \
    --output artifacts/countdown/sft_datasets/sft_whole__method_b.json

# Hint SFT on top of the baseline model
python -m pipeline train_sft --task countdown --method method_b --run-id qwen2.5-1.5b \
    --base-model artifacts/countdown/models/baseline_models/qwen2.5-1.5b/model \
    --dataset artifacts/countdown/sft_datasets/sft_whole__method_b.json \
    --output artifacts/countdown/models/method_b_sft/qwen2.5-1.5b/model

# RL with the quadratic hint penalty. alpha is the swept parameter and is not
# in the method config, so it must be passed here -- and it is the only thing
# distinguishing the runs, which is why it is in the run id.
python -m pipeline train_rl --task countdown --method method_b \
    --run-id qwen2.5-1.5b__quad-a0.5 \
    --sft-model artifacts/countdown/models/method_b_sft/qwen2.5-1.5b/model \
    --train-prompts artifacts/countdown/problems_with_format/rl_train__method_b.parquet \
    --val-prompts artifacts/countdown/problems_with_format/rl_val__method_b.parquet \
    --reward-kwargs hint_penalty=0.1 hint_penalty_shape=quadratic hint_penalty_alpha=0.5 \
    --override algorithm.norm_adv_by_std_in_grpo=True \
    --output artifacts/countdown/models/method_b_models/qwen2.5-1.5b__quad-a0.5/model

python -m pipeline evaluate --task countdown --method method_b --model rl --async \
    --run-id qwen2.5-1.5b__quad-a0.5 --split eval \
    --prompts artifacts/countdown/problems_with_format/eval__method_b.json \
    --output artifacts/countdown/models/method_b_models/qwen2.5-1.5b__quad-a0.5/evals/eval.json
```

Pass `--async` to `generate` and `evaluate` for the batched async vLLM path; it is the standard mode here. `--multi-turn` is read from the method config and does not need passing.

## Tasks

| Task | Problem |
|---|---|
| `countdown` | Reach a target number by combining given operands with `+ - * /` |
| `competition_math` | Competition math problems (HuggingFace MATH) |
| `code_output` | Predict the stdout of a short program |

Each lives in `pipeline/tasks/{task}/` and implements the `BaseTask` interface: `create_primitives`, `format_prompt`, `check_correctness`.

## Methods

A *method* bundles a prompt-template variant, a reward function, and its RL settings into one named config (`pipeline/configs/methods/{task}/{method}.yaml`). The config name is also the name its artifacts carry on disk, so renaming a method renames its files. Four methods exist on countdown and competition math, three on code output (`method_b` is math-and-countdown only).

**Evaluated methods:**

| Method | Behavior |
|---|---|
| `baseline` | Answer directly, never ask for a hint. The starting point for every other method. |
| `hint_encourage` | Multi-turn: ask for hints, with a bonus for admitting a wrong answer. |
| `method_b` | Multi-turn hint-seeking inside a single `<think>` block, swept over the hint-penalty weight α. Countdown and competition math only. |

**SFT parents** — not evaluated on their own, but required to produce the hint methods above. Do not delete them:

| Method | Parent of |
|---|---|
| `hint` | `hint_encourage`, `method_b` |

Model sizes used throughout: Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B.

Run `python -m pipeline list_methods --task <task>` to list what a task actually has.

Reward functions themselves live with the trainer, in `verl/recipe/{task}/reward_function.py`; a method config selects one by name and passes it `reward_kwargs`.

## Commands

**Discovery**

| Command | Purpose |
|---|---|
| `list_tasks` | List registered tasks |
| `list_methods` | List method configs for a task |

**Data**

| Command | Purpose |
|---|---|
| `create_primitives` | Generate raw puzzle data (shared across methods) |
| `create_partitions` | Split primitives into per-split problem files under `problems/` |
| `create_prompts` | Apply a method's template to a partition, or `all` of them |
| `create_ood_prompts` | Build OOD eval prompts (`aime2024`, `gsm8k`, `math500`, `minerva_math`, `olympiad_bench`, `unanswerable_math`) |

**Inference**

| Command | Purpose |
|---|---|
| `generate` | Run a model over prompts to produce a dataset (`--async`, `--no-hints`, `--hint-schedule`, `--target-correct-rate`) |
| `evaluate` | Evaluate a model (`--model sft`/`rl`, `--run-id`, `--num-samples`, `--async`) |
| `analyze` | Report accuracy by variant for a dataset or results file |

**Training**

| Command | Purpose |
|---|---|
| `train_sft` | SFT (`--epochs` 3, `--batch-size` 4, `--max-length` 4096) |
| `train_rl` | RL (`--total-steps` 400, `--train-batch-size` 64, `--save-freq` 25, `--reward-kwargs`, `--override`) |
| `convert_checkpoint` | Convert an FSDP/Megatron checkpoint to HuggingFace format |

`python -m pipeline <command> --help` documents every flag.

Every command that touches a model path requires `--run-id`; nothing is derived from the base checkpoint. Commands that read or write artifacts take an explicit `--primitives` / `--prompts` / `--dataset` / `--output`, and fall back to the method-derived path when omitted.

## Data model

Each stage is a pure transformation that writes new files and never edits existing ones:

```
problems/primitives.json                      Raw puzzle data (index, variant, task fields)
       │
       │  create_partitions
       ▼
problems/{split}.json                         The seeded split, written down once
       │
       │  create_prompts        + a method's template
       ▼
problems_with_format/{split}__{method}.json   Model-ready inputs + ground truth
       │                        .parquet for rl_train / rl_val
       │  generate              + a model's rollouts
       ▼
sft_datasets/{split}__{method}.json           Generations + correctness labels
       │
       │  train_sft -> train_rl
       ▼
models/{method}_{sft,models}/{run_id}/model/
       │
       │  evaluate
       ▼
models/{method}_{sft,models}/{run_id}/evals/{split}.json
```

### Splits

Splits are disjoint slices of a seeded shuffle of the primitives, so no problem appears in both training and evaluation. **Tasks declare their own layout**, and not every task defines every split:

| Split | `countdown`, `competition_math` | `code_output` |
|---|---|---|
| `sft_whole` | 0–30% | 0–19.2% |
| `sft_train` | 0–27% | 0–17.28% |
| `sft_val` | 27–30% | 17.28–19.2% |
| `rl_train` | 30–65% | 19.2–67.2% |
| `rl_val` | 65–70% | — |
| `eval` | 70–100% | 67.2–100% |

`sft_whole` is the one overlap, and it is an identity rather than an exception: it is exactly `sft_train` + `sft_val`, index for index, because the three share a left edge and `sft_val` is carved off the tail. `BaseTask.__init_subclass__` checks that at import time, so a task cannot override the table into a layout where the name lies. Materialize whichever of the three a stage needs — train on `sft_train`, early-stop on `sft_val`, or train on `sft_whole` when there is nothing to early-stop against.

`code_output` allocates its whole range across three regions, so it has no `rl_val`.

`--split all` creates exactly the splits the task defines — a task's `SPLITS` table in `pipeline/tasks/{task}/task.py` is the single source of truth.

## Artifacts

Generated data and model weights are written under `artifacts/`. The tree is organized by **stage**, not by method: each layer is one directory per task, and a method's files are told apart by name.

```
artifacts/{task}/
├── problems/                              problems. no template.
│   ├── primitives.json                    raw, shared by every method
│   └── {split}.json                       one per split, written by create_partitions
├── problems_with_format/                  + a method's template. model-ready, nothing generated.
│   ├── {split}__{method}.json             sft_whole, sft_train, sft_val, eval
│   └── rl_{train,val}__{method}.parquet   verl rolls out itself, so no generations here
├── sft_datasets/                          + generations and correctness labels
│   └── {split}__{method}.json
├── models/{method}_{sft,models}/{run_id}/
│   ├── model/                             HuggingFace weights, or a symlink to last/ or best/
│   ├── last/  best/  rollouts/            RL only
│   └── evals/{split}.json                 results live inside the run that produced them
└── .scratch/                              intermediates read back by a later step,
    └── {split}__{method}__probe.json      never trained on
```

The three data layers are distinguished by **what is in a file**, not by which stage reads it. That is what decides the two cases people get wrong: the RL parquet holds prompts and ground truth and no generations, so it belongs in `problems_with_format/`; and `sft_datasets/` is SFT-stage-only by construction rather than by convention, since nothing else ever carries generations.

Names come from the method's config filename, so `baseline.yaml` produces `sft_whole__baseline.json` and `models/baseline_sft/`. Fields are separated by `__` throughout — `{split}__{method}` and `{split}__{method}__{desc}` — because method names contain single underscores (`method_b`, `method_ac`) and a single-underscore separator would make `sft_whole__method_b_probe` ambiguous. Hyphens stay legal *inside* a field, which is what carries model slugs (`qwen3-4b-base`) and run descriptions (`quad-a0.5`).

`--run-id` names the run directory verbatim and is **required** wherever a model path is derived. A run is `{model_slug}` or `{model_slug}__{desc}`, where `{desc}` marks a deviation from the default recipe — so a default run is the bare slug (`qwen2.5-1.5b`) and only the varying part is named (`qwen2.5-1.5b__quad-a0.5`, `qwen3-4b-base__s2`). Nothing is inferred from the base checkpoint; a name guessed in code would only drift from the convention.

## Repository layout

```
pipeline/
├── __main__.py              # CLI entrypoint
├── commands/                # data, inference, training
├── configs/methods/{task}/  # method configs
├── core/                    # io, generator (vLLM), method paths
└── tasks/{task}/            # task class + prompt templates
verl/                        # modified verl trainer + reward functions
```

## Adding a task

1. Create `pipeline/tasks/{task}/` with a class extending `BaseTask`, implementing `create_primitives`, `format_prompt`, and `check_correctness`.
2. Override `SPLITS` if the default split layout does not suit the data.
3. Add templates under `pipeline/tasks/{task}/templates/{variant}/`.
4. Add method configs under `pipeline/configs/methods/{task}/`.
5. Register the class in `pipeline/tasks/__init__.py`.
