#!/bin/bash
# ============================================================================
# competition_math / hint - hint_penalty ablation on Qwen2.5-3B
# ============================================================================
# Three arms differing only in hint_penalty, all starting from the same SFT
# parent. Submitted as a job array so one sbatch launches the whole sweep:
#
#   sbatch scripts/rl_math_hint_penalty_sweep.sh
#
# To add a single arm to a sweep already in flight, override the array with
# just that index, e.g. `sbatch --array=1 ...` for p=0.05. Arms are only
# comparable when they ran the same rollout code, so check git before doing
# that: the v2 arms all postdate the forced-answer fixes.
#
# Each array task takes one B200 on the high-priority genai partition
# (PriorityTier=30, so these preempt the tier-20 genai/goyal jobs rather than
# queue behind them, and nothing outranks them in return). Budget 48h: the
# v2 rollouts are longer than v1's (hint-users are no longer truncated early,
# and the forced-answer pass adds a decode round), so the measured rate is
# ~4.8 min/step on a B200, i.e. ~31h for 400 steps.
#
# All arms deliberately share one GPU type. Splitting an ablation across
# H100 and B200 would put a hardware difference inside the comparison.
#
# Unlike the older per-method scripts, this one never runs `ray start` or
# `ray stop`. verl calls ray.init() with no address, so each job builds its
# own in-process cluster. Starting a shared head on a fixed port would make
# concurrent array tasks collide, and `ray stop` would kill their siblings.
# ============================================================================
#SBATCH --job-name=cmath_hint_psweep
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/hint_psweep_%A_%a.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/hint_psweep_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --partition=genai-goyal-highpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayf7@cornell.edu

set -euo pipefail

REPO_ROOT="/home/ayf7/repos/abstention-reasoning"
CONDA_ENV="/home/ayf7/env-b200"
SFT_MODEL="artifacts/competition_math/hint/models/sft/qwen2.5-3b-up4x/model"
# v2: the v1 curves were confounded by truncation being scored 0 (see
# logs/diagnosis.md), so they are kept in a separate project rather than
# plotted alongside runs that no longer share their reward geometry.
PROJECT="qwen2.5-3b-math-hint-ablation-v2"

# Arm selection -------------------------------------------------------------
PENALTIES=(0 0.05 0.1)
P="${PENALTIES[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="qwen2.5-3b-math-hint-p${P}"

# Hyperparameters -----------------------------------------------------------
N_SAMPLES=8            # GRPO group size; larger so groups contain hint-users
TRAIN_BATCH_SIZE=64
LR=1e-6
TOTAL_STEPS=400
TEST_FREQ=10           # validate every 10 steps; --save-best keys off this
MAX_PROMPT_LENGTH=2048

# Budget split. THINK_BUDGET caps the model's own decoded tokens; ANSWER_BUDGET
# is held back for a forced answer when a rollout thinks right up to that cap,
# so running out of room yields a gradable answer instead of a 0-scored
# malformed response. MAX_RESPONSE_LENGTH is the response *tensor* width and
# must additionally cover injected hint text, which no longer counts against
# the decode budget: measured worst case is 597 tokens for all 5 hints, so
# 2048 + 64 + 704 headroom.
THINK_BUDGET=2048
ANSWER_BUDGET=64
MAX_RESPONSE_LENGTH=2816
MAX_MODEL_LEN=8192

# Select the checkpoint by accuracy, not reward: reward is 1 - p*hints, so it
# is not comparable across arms and would favour whichever arm penalises least.
BEST_METRIC="val-aux/competition_math/correct/mean"

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"

# Each task builds its own Ray instance; a stale address would hijack that.
unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

if [ ! -d "$SFT_MODEL" ]; then
    echo "FATAL: SFT parent not found: $SFT_MODEL" >&2
    exit 1
fi

echo "=============================================="
echo " competition_math / hint - penalty ablation"
echo "=============================================="
echo " array task:    ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"
echo " hint_penalty:  ${P}"
echo " run id:        ${RUN_ID}"
echo " sft parent:    ${SFT_MODEL}"
echo " group size:    ${N_SAMPLES}"
echo " steps:         ${TOTAL_STEPS} (validate every ${TEST_FREQ})"
echo " budgets:       think ${THINK_BUDGET} + answer ${ANSWER_BUDGET}, tensor ${MAX_RESPONSE_LENGTH}"
echo " best metric:   ${BEST_METRIC}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================================="

python -m pipeline train_rl \
    --task competition_math \
    --method hint \
    --run-id "${RUN_ID}" \
    --sft-model "${SFT_MODEL}" \
    --reward-kwargs "hint_penalty=${P}" \
    --n-samples "${N_SAMPLES}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --learning-rate "${LR}" \
    --total-steps "${TOTAL_STEPS}" \
    --test-freq "${TEST_FREQ}" \
    --save-best \
    --best-metric "${BEST_METRIC}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --force-answer "{\"think_tokens\": ${THINK_BUDGET}, \"answer_tokens\": ${ANSWER_BUDGET}}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --project-name "${PROJECT}" \
    --experiment-name "p${P}" \
    --override "ray_init.num_cpus=${SLURM_CPUS_PER_TASK}"

echo "Arm p=${P} complete."
