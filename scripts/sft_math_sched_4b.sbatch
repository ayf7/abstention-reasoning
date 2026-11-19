#!/bin/bash
# ============================================================================
# competition_math / hint_ablations - SFT Qwen3-4B on the scheduled-hint data
# ============================================================================
#   sbatch scripts/sft_math_sched_4b.sbatch
#
# A/B against the RL parent qwen3-4b-up4x-tok-fullloss: same base model
# (Qwen3-4B-Base-plaintags), same epochs / lr / batch / max_length, same
# full-sequence loss. Two things change, both about the data:
#   - dataset: sft_scheduled_n8_qwen3-14b.json (1682 rows, all correct, 48%
#     with hints spread over 1-5 requests) instead of sft_qwen3-14b.json
#     (1219 correct rows, 15% with hints).
#   - no --upsample-hint 4. The parent used it to lift hinted rows to ~42% of
#     the training mix; the scheduled data is already there (48%), so the
#     hinted share matches without duplicating rows.
# Needs a B200 like the parent: batch 4 x 4096 tokens without gradient
# checkpointing OOMs a 94 GB H100 at step 1 (job 214801), and with it too.
# ============================================================================
#SBATCH --job-name=sft-mth-4b-sched
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/sft_sched4b_%j.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/sft_sched4b_%j.err
#SBATCH --time=6:00:00
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
RUN_ID="qwen3-4b-sched-n8-tok-fullloss"
BASE_MODEL="/share/goyal/ayf7/artifacts/base_models/Qwen3-4B-Base-plaintags"
DATASET="artifacts/competition_math/hint_ablations/datasets/sft_scheduled_n8_qwen3-14b.json"
OUT_MODEL="artifacts/competition_math/hint_ablations/models/sft/${RUN_ID}/model"

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================="
echo " hint_ablations SFT - Qwen3-4B on scheduled hints"
echo "=============================================="
echo " run id:        ${RUN_ID}"
echo " base model:    ${BASE_MODEL}"
echo " dataset:       ${DATASET}"
echo " output:        ${OUT_MODEL}"
echo " job / restart: ${SLURM_JOB_ID} / ${SLURM_RESTART_COUNT:-0}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

# SFT keeps only the final weights (save_strategy=no): nothing to resume after
# a preemption, but nothing to redo either if the model is already written.
if [ -d "$OUT_MODEL" ] && compgen -G "${OUT_MODEL}/*.safetensors" > /dev/null; then
    echo "SFT model already present, nothing to do: ${OUT_MODEL}"
    exit 0
fi

python -m pipeline train_sft \
    --task competition_math \
    --method hint_ablations \
    --base-model "${BASE_MODEL}" \
    --run-id "${RUN_ID}" \
    --dataset "${DATASET}" \
    --project-name math_new_hint \
    --experiment-name "sft-v2-${RUN_ID}"

echo "SFT ${RUN_ID} complete."
