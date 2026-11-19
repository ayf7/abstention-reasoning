#!/bin/bash
# ============================================================================
# competition_math / hint_ablations - eval the scheduled-hint SFT model
# ============================================================================
#   sbatch --dependency=afterok:<sft job> scripts/eval_sft_math_sched_4b.sbatch
#
# Same protocol as results/eval_augmented_sft_qwen3-4b-up4x-tok-fullloss.json
# (the RL parent): eval_augmented split, T=1.0, top_p=1.0, one sample,
# max_new_tokens 2816, multi-turn hints. Output lands next to it so the two
# are directly comparable.
# ============================================================================
#SBATCH --job-name=eval-sft-4b-sched
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/eval_sched4b_%j.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/eval_sched4b_%j.err
#SBATCH --time=4:00:00
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
TASK="competition_math"
METHOD="hint_ablations"
SPLIT="eval_augmented"
RUN_ID="qwen3-4b-sched-n8-tok-fullloss"
MAX_NEW_TOKENS=2816
TEMPERATURE=1.0
TOP_P=1.0
NUM_SAMPLES=1

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"

MODEL="artifacts/${TASK}/${METHOD}/models/sft/${RUN_ID}/model"
OUT="artifacts/${TASK}/${METHOD}/results/${SPLIT}_sft_${RUN_ID}.json"

if ! compgen -G "${MODEL}/*.safetensors" > /dev/null; then
    echo "FATAL: SFT model not found: ${MODEL}" >&2
    exit 1
fi
if [ -f "$OUT" ]; then
    echo "Results already present, nothing to do: ${OUT}"
    exit 0
fi

echo "=============================================="
echo " hint_ablations eval - scheduled-hint SFT"
echo "=============================================="
echo " model:         ${MODEL}"
echo " split:         ${SPLIT}"
echo " sampling:      T=${TEMPERATURE} top_p=${TOP_P} n=${NUM_SAMPLES} max_new=${MAX_NEW_TOKENS}"
echo " output:        ${OUT}"
echo " job / restart: ${SLURM_JOB_ID} / ${SLURM_RESTART_COUNT:-0}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

python -m pipeline evaluate \
    --task "${TASK}" \
    --method "${METHOD}" \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --output "${OUT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"

echo "Eval ${RUN_ID} complete -> ${OUT}"
