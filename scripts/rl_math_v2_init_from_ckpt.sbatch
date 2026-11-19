#!/bin/bash
# ============================================================================
# competition_math / hint_ablations - p=0.05 arm started from a mixed policy
# ============================================================================
#   sbatch scripts/rl_math_v2_init_from_ckpt.sbatch
#
# Same hyperparameters as the 4B arms of scripts/rl_math_v2_sweep.sbatch in the
# feat-rollout-optimization worktree (mean-only advantages, continuous hints,
# reward without the exhausted-request penalty), with one change: the initial
# policy is the p=0.0125 arm at global step 50 instead of the SFT parent.
#
# Why. GRPO only gets a signal about asking for hints from groups in which some
# rollouts asked and some did not. The SFT parent asks on ~7% of rollouts, one
# hint each, and at p=0.05 that behaviour was pruned to 1% within 20 steps,
# before the model found that 3-5 hints on hard prompts pay. The p=0.0125 arm
# after 50 updates asks on 37% of rollouts with 1-6 hints and 89% of its groups
# are mixed, so the target reward can prune from the other side.
#
# The reward is pinned to what the running arms use so the initial policy is
# the only difference: exhausted_penalty=0 disables the newer penalty that the
# method yaml now carries by default.
# ============================================================================
#SBATCH --job-name=rl-mth-4b-p0.05-init
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/v2_init_%j.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/v2_init_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=genai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayf7@cornell.edu

set -euo pipefail

REPO_ROOT="/home/ayf7/repos/abstention-reasoning"
CONDA_ENV="/home/ayf7/env-b200"

P=0.05
INIT_RUN="qwen3-4b-math-v2-hint-p0.0125"
INIT_STEP=50
# HF export of ${INIT_RUN}/checkpoints/global_step_${INIT_STEP}/actor, written
# by `python -m pipeline convert_checkpoint`. Kept outside that run's
# checkpoints/ so its pruning cannot touch it.
INIT_MODEL="artifacts/competition_math/hint_ablations/models/rl_init/${INIT_RUN}-step${INIT_STEP}/model"

RUN_ID="qwen3-4b-math-v2-hint-p${P}-init-p0.0125s${INIT_STEP}"
EXP_NAME="v2-4b-p${P}-init-p0.0125s${INIT_STEP}"
PROJECT="math_new_hint_4b"

# Verbatim from rl_math_v2_new_hint.sh (4B branch).
N_SAMPLES=8
TRAIN_BATCH_SIZE=64
LR=1e-6
TOTAL_STEPS=400
TEST_FREQ=10
SAVE_FREQ=25
MAX_CKPT_TO_KEEP=4
MAX_PROMPT_LENGTH=2048
THINK_BUDGET=2048
ANSWER_BUDGET=64
MAX_RESPONSE_LENGTH=2816
MAX_MODEL_LEN=8192
NORM_ADV_BY_STD=False

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"

unset RAY_ADDRESS || true
# Short on purpose: the plasma store socket path is capped at 107 bytes.
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-init}"
mkdir -p "$RAY_TMPDIR"

if [ ! -d "$INIT_MODEL" ]; then
    echo "FATAL: init model not found: $INIT_MODEL" >&2
    exit 1
fi

VAL_PARQUET="artifacts/competition_math/hint_ablations/prompts/rl_val.parquet"
DATA_SOURCE=$(python -c "
import sys, pandas as pd
v = pd.read_parquet(sys.argv[1])['data_source'].unique()
assert len(v) == 1, f'mixed data_source: {list(v)}'
print(v[0])
" "$VAL_PARQUET")
BEST_METRIC="val-aux/${DATA_SOURCE}/correct/mean"

# Requeue after preemption resumes from the last checkpoint; a run dir with
# nothing checkpointed is discarded instead of blocking the relaunch.
RUN_DIR="artifacts/competition_math/hint_ablations/models/rl/${RUN_ID}"
RESUME_FLAG=""
if compgen -G "${RUN_DIR}/checkpoints/global_step_*" > /dev/null; then
    RESUME_FLAG="--continue-run"
elif [ -d "$RUN_DIR" ]; then
    RESUME_FLAG="--overwrite"
fi

echo "=============================================="
echo " v2 hint_penalty arm from a mixed init policy"
echo "=============================================="
echo " hint_penalty:  ${P}"
echo " init policy:   ${INIT_MODEL}"
echo " run id:        ${RUN_ID}"
echo " steps:         ${TOTAL_STEPS} (validate every ${TEST_FREQ}, save every ${SAVE_FREQ}, keep ${MAX_CKPT_TO_KEEP})"
echo " resume:        ${RESUME_FLAG:-fresh run}"
echo " adv by std:    ${NORM_ADV_BY_STD}"
echo " wandb:         off"
echo " job / restart: ${SLURM_JOB_ID:-none} / ${SLURM_RESTART_COUNT:-0}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

python -m pipeline train_rl ${RESUME_FLAG} \
    --task competition_math \
    --method hint_ablations \
    --run-id "${RUN_ID}" \
    --sft-model "${INIT_MODEL}" \
    --reward-kwargs "hint_penalty=${P}" "exhausted_penalty=0" \
    --n-samples "${N_SAMPLES}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --learning-rate "${LR}" \
    --total-steps "${TOTAL_STEPS}" \
    --test-freq "${TEST_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
    --max-ckpt-to-keep "${MAX_CKPT_TO_KEEP}" \
    --save-best \
    --best-metric "${BEST_METRIC}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --force-answer "{\"think_tokens\": ${THINK_BUDGET}, \"answer_tokens\": ${ANSWER_BUDGET}}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --project-name "${PROJECT}" \
    --experiment-name "${EXP_NAME}" \
    --no-wandb \
    --override "ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}" \
               "actor_rollout_ref.rollout.continuous_hints=true" \
               "algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD}"

echo "Run ${RUN_ID} complete."
