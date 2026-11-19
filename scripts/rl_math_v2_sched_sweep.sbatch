#!/bin/bash
# ============================================================================
# competition_math / hint_ablations - hint_penalty sweep from the scheduled-hint
# SFT parent, one arm per submission
# ============================================================================
#   sbatch --partition=genai-goyal-highpri scripts/rl_math_v2_sched_sweep.sbatch 0.025
#   sbatch scripts/rl_math_v2_sched_sweep.sbatch 0.1
#   sbatch scripts/rl_math_v2_sched_sweep.sbatch 0.05 <sft run id> <tag>   # another parent
#
# Same hyperparameters as the 4B arms of rl_math_v2_sweep.sbatch in the
# feat-rollout-optimization worktree (mean-only advantages, continuous hints,
# n=8, batch 64, lr 1e-6, 400 steps). What changes is the SFT parent:
# qwen3-4b-sched-n8-tok-fullloss, trained on sft_scheduled_n8_qwen3-14b.json
# (48% of rows hinted, 1-5 hints each, no "let me request a hint" phrasing).
# On eval_augmented it asks on 57% of prompts (parent: 16%) and scores 0.760
# (parent: 0.688) with hints counted, so GRPO starts from mixed groups instead
# of pruning a 7% ask rate to nothing within 20 steps.
#
# The reward is the method yaml default, which the earlier arms predate:
# requests after the 5 hints run out cost exhausted_penalty=0.2 each and more
# than 2 of them score 0. The new parent has a 6+ hint tail (2% of prompts,
# up to 91 requests), which is what that penalty is for.
#
# The earlier arms' run dirs were moved to <run_id>.bak, so the run ids below
# are reused; the wandb names carry "sched" so the two sweeps do not overlap.
#
# Partition: genai-goyal-highpri (tier 30) preempts genai (tier 20) youngest
# first, so submit highpri arms before genai ones. --requeue plus the resume
# logic below makes a preemption cost at most save_freq (25) steps.
# ============================================================================
#SBATCH --job-name=rl-mth-4b-sched
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/v2_sched_%j.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/v2_sched_%j.err
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

P="${1:?usage: sbatch scripts/rl_math_v2_sched_sweep.sbatch <hint_penalty> [sft_run] [tag]}"
# Optional: start from a different scheduled-hint SFT parent, with a tag that
# keeps the run id, job name and wandb name apart from the sweep's
# (sched_chain_math_4b.sbatch uses this for its regenerated parent).
SFT_RUN="${2:-qwen3-4b-sched-n8-tok-fullloss}"
TAG="${3:+-$3}"

REPO_ROOT="/home/ayf7/repos/abstention-reasoning"
CONDA_ENV="/home/ayf7/env-b200"

SFT_MODEL="artifacts/competition_math/hint_ablations/models/sft/${SFT_RUN}/model"
RUN_ID="qwen3-4b-math-v2-hint${TAG}-p${P}"
EXP_NAME="v2-4b-sched${TAG}-p${P}"
PROJECT="math_new_hint_4b"

# squeue truncates the name column, so carry only what tells arms apart. Set
# here because #SBATCH lines cannot see the script's arguments.
scontrol update JobId="${SLURM_JOB_ID}" JobName="rl-mth-4b-sched${TAG}-p${P}" 2>/dev/null || true

# Verbatim from rl_math_v2_new_hint.sh (4B branch), except TEST_FREQ.
N_SAMPLES=8
TRAIN_BATCH_SIZE=64
LR=1e-6
TOTAL_STEPS=400
# Was 10. Validation on the 294 rl_val prompts costs ~3-5 min against a ~2 min
# step, so every 10 steps was ~14% of wall clock. Every 25 lines it up with the
# periodic saves, so the best checkpoint is always one that is on disk anyway.
# The arms were requeued onto this value from their step-25 checkpoints.
TEST_FREQ=25
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
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-schedp${P}}"
mkdir -p "$RAY_TMPDIR"

if ! compgen -G "${SFT_MODEL}/*.safetensors" > /dev/null; then
    echo "FATAL: SFT parent not found: $SFT_MODEL" >&2
    exit 1
fi

# verl names validation metrics after the parquet's data_source column, which
# predates the hint_ablations rename; read it rather than spell it.
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
echo " v2 hint_penalty sweep arm - scheduled-hint SFT parent"
echo "=============================================="
echo " hint_penalty:  ${P}"
echo " sft parent:    ${SFT_MODEL}"
echo " run id:        ${RUN_ID}"
echo " wandb:         ${PROJECT} / ${EXP_NAME}"
echo " best metric:   ${BEST_METRIC}"
echo " steps:         ${TOTAL_STEPS} (validate every ${TEST_FREQ}, save every ${SAVE_FREQ}, keep ${MAX_CKPT_TO_KEEP})"
echo " resume:        ${RESUME_FLAG:-fresh run}"
echo " adv by std:    ${NORM_ADV_BY_STD}"
echo " job / restart: ${SLURM_JOB_ID:-none} / ${SLURM_RESTART_COUNT:-0}"
echo " partition:     ${SLURM_JOB_PARTITION:-none}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

python -m pipeline train_rl ${RESUME_FLAG} \
    --task competition_math \
    --method hint_ablations \
    --run-id "${RUN_ID}" \
    --sft-model "${SFT_MODEL}" \
    --reward-kwargs "hint_penalty=${P}" \
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
    --override "ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}" \
               "actor_rollout_ref.rollout.continuous_hints=true" \
               "algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD}"

echo "Run ${RUN_ID} complete."
