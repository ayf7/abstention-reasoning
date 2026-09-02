# abstention-reasoning

Training language models to recognize when they cannot solve a problem and perform alternative action (abstaining or hint-seeking), rather than committing to an answer every time.

Models are trained in two stages: supervised fine-tuning (SFT) on chain-of-thought traces, then reinforcement learning with a reward function that pays for correct answers *and* for well-calibrated abstention. RL runs on a modified copy of [verl](https://github.com/volcengine/verl), vendored in `verl/`.

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

End-to-end on `countdown` with the `simple` baseline:

```bash
# 1. Generate raw puzzles (shared by every method for this task)
python -m pipeline create_primitives --task countdown --num-puzzles 5000

# 2. Render prompts for every split the task defines
python -m pipeline create_prompts --task countdown --method simple --split all

# 3. Sample SFT training data from a strong teacher model
python -m pipeline generate --task countdown --method simple --model Qwen/Qwen3-4B --split sft --async

# 4. Supervised fine-tune
python -m pipeline train_sft --task countdown --method simple --base-model Qwen/Qwen2.5-1.5B

# 5. RL from the SFT checkpoint
python -m pipeline train_rl --task countdown --method simple

# 6. Evaluate. --model takes 'sft'/'rl' to resolve the method's own checkpoint
python -m pipeline evaluate --task countdown --method simple --model rl --async
```

`--method` auto-derives every artifact path, so steps rarely need explicit `--prompts` / `--dataset` / `--output`. Pass `--async` to `generate` and `evaluate` for the batched async vLLM path; it is the standard mode here.

## Tasks

| Task | Problem |
|---|---|
| `countdown` | Reach a target number by combining given operands with `+ - * /` |
| `competition_math` | Competition math problems (HuggingFace MATH) |
| `code_output` | Predict the stdout of a short program |

Each lives in `pipeline/tasks/{task}/` and implements the `BaseTask` interface: `create_primitives`, `format_prompt`, `check_correctness`.

## Methods

A *method* bundles a prompt-template variant, a reward function, and its RL settings into one named config (`pipeline/configs/methods/{task}/{method}.yaml`). Six methods exist for all three tasks.

**Evaluated methods:**

| Method | Behavior |
|---|---|
| `simple` | Baseline. Answer directly, never abstain. |
| `verify` | Answer, then verify your own answer. |
| `abstention_verify` | Verify, then either `<commit>` or `<abstain>`. |
| `hint_encourage` | Multi-turn: ask for hints, with a bonus for admitting a wrong answer. |

**SFT parents** — not evaluated on their own, but required to produce the two above. Do not delete them:

| Method | Parent of |
|---|---|
| `simple_abstention` | `abstention_verify` |
| `hint` | `hint_encourage` |

Model sizes used throughout: Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B.

Remaining configs in `pipeline/configs/methods/` are ablations and retired variants. Run `python -m pipeline list_methods --task countdown` to list them.

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
| `create_prompts` | Render prompts from primitives for a split, or `all` |
| `create_verify_prompts` | Build `verify` / `abstention_verify` prompts from an existing dataset |
| `create_ood_prompts` | Build OOD eval prompts (`aime2024`, `gsm8k`, `math500`, `minerva_math`, `olympiad_bench`, `unanswerable_math`) |

**Inference**

| Command | Purpose |
|---|---|
| `generate` | Run a model over prompts to produce a dataset (`--async`, `--multi-turn`) |
| `evaluate` | Evaluate a model (`--run-id`, `--num-samples`, `--async`) |
| `analyze` | Report accuracy by variant for a dataset or results file |

**Training**

| Command | Purpose |
|---|---|
| `train_sft` | SFT (`--epochs` 3, `--batch-size` 4, `--max-length` 4096) |
| `train_rl` | RL (`--total-steps` 400, `--train-batch-size` 64, `--save-freq` 25) |
| `convert_checkpoint` | Convert an FSDP/Megatron checkpoint to HuggingFace format |

`python -m pipeline <command> --help` documents every flag.

## Data model

Each stage is a pure transformation that writes new files and never edits existing ones:

```
primitives.json     Raw puzzle data (index, variant, task-specific fields)
       │
       ├── template ───► prompts/*.json     Model-ready inputs + ground truth
       │
       └── model ──────► datasets/*.json    Generations + correctness labels
                              │
                              └─────────────► results/*.json   Metrics + details
```

### Splits

Splits are disjoint slices of a seeded shuffle of the primitives, so no problem appears in both training and evaluation. **Tasks declare their own layout**, and not every task defines every split:

| Split | `countdown`, `competition_math` | `code_output` |
|---|---|---|
| `sft` | 0–30% | 0–19.2% |
| `rl_train` | 30–65% | 19.2–67.2% |
| `rl_val` | 65–70% | — |
| `eval` | 90–100% | 67.2–100% |
| `eval_augmented` | 70–100% | — |

`eval_augmented` is the deliberate exception to disjointness: it is a superset of `eval` drawn from otherwise-unused indices, and it is what most evaluations actually read. `code_output` allocates its whole range across three splits, so it has no spare region for `eval_augmented` and no `rl_val`.

`--split all` creates exactly the splits the task defines — a task's `SPLITS` table in `pipeline/tasks/{task}/task.py` is the single source of truth.

## Artifacts

Generated data and model weights are written under `artifacts/`, which is typically a symlink to shared storage:

```
artifacts/{task}/
├── primitives.json              # shared across methods
└── {method}/
    ├── prompts/                 # sft.json, rl_train.parquet, eval.json, ...
    ├── datasets/                # generations + correctness labels
    ├── models/
    │   ├── sft/{run_id}/model/
    │   └── rl/{run_id}/model/   # + checkpoints/, rollouts/
    └── results/                 # evaluation metrics
```

Use `--run-id` to keep multiple runs of the same (task, method) side by side.

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
