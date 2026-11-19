#!/bin/bash
# ============================================================================
# competition_math - eval the SFT and RL baselines on the Method B protocol
# ============================================================================
#   sbatch scripts/eval_competition_math_baselines.sbatch
#
# The RL baselines have never been evaluated, and the existing SFT evals
# (evals/eval_narrow.json) ran on a 500-problem split at T=0.7 / 4096 tokens,
# so they cannot be read next to the Method B numbers. This re-runs both on
# the Method B protocol: the 1764-problem eval split, T=1.0, top_p=1.0, one
# sample, 2048 tokens, single turn (the baselines have no hint channel).
#
# Paths are explicit because this artifacts tree is a reorganised copy and does
# not follow the {task}/{method}/models/{sft,rl}/{run_id}/ layout that --method
# derives. eval_narrow.json is left alone; output lands at evals/eval.json,
# matching what the Method B runs are called.
# ============================================================================
#SBATCH --job-name=eval-math-baselines
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/eval_math_baseline_%A_%a.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/eval_math_baseline_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=4:00:00
#SBATCH --partition=genai-goyal-highpri
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
ARTIFACTS="/share/goyal/ayf7/artifacts/competition_math"

TASK="competition_math"
PROMPTS="${ARTIFACTS}/problems_with_format/eval__baseline.json"
MAX_NEW_TOKENS=2048
TEMPERATURE=1.0
TOP_P=1.0
NUM_SAMPLES=1

# (family, model dir) per array index. The family is also the directory the
# weights live under: baseline_sft/ for SFT, baseline_models/ for RL.
FAMILIES=(baseline_sft    baseline_sft  baseline_sft   baseline_models baseline_models baseline_models)
MODELS=(  qwen2.5-1.5b    qwen2.5-3b    qwen3-4b-base  qwen2.5-1.5b    qwen2.5-3b      qwen3-4b-base)

FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${ARTIFACTS}/models/${FAMILY}/${MODEL_NAME}/model"
OUT="${ARTIFACTS}/models/${FAMILY}/${MODEL_NAME}/evals/eval.json"

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"

if ! compgen -G "${MODEL}/*.safetensors" > /dev/null; then
    echo "FATAL: model weights not found: ${MODEL}" >&2
    exit 1
fi
if [ ! -f "$PROMPTS" ]; then
    echo "FATAL: prompts not found: ${PROMPTS}" >&2
    exit 1
fi
if [ -f "$OUT" ]; then
    echo "Results already present, nothing to do: ${OUT}"
    exit 0
fi
mkdir -p "$(dirname "$OUT")"

echo "=============================================="
echo " competition_math baseline eval"
echo "=============================================="
echo " family / model: ${FAMILY} / ${MODEL_NAME}"
echo " weights:        ${MODEL}"
echo " prompts:        ${PROMPTS}"
echo " sampling:       T=${TEMPERATURE} top_p=${TOP_P} n=${NUM_SAMPLES} max_new=${MAX_NEW_TOKENS} single-turn"
echo " output:         ${OUT}"
echo " job / task:     ${SLURM_ARRAY_JOB_ID} / ${SLURM_ARRAY_TASK_ID} (restart ${SLURM_RESTART_COUNT:-0})"
echo " node / gpu:     $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

python -m pipeline evaluate \
    --task "${TASK}" \
    --model "${MODEL}" \
    --prompts "${PROMPTS}" \
    --output "${OUT}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"

echo "Eval complete: ${FAMILY}/${MODEL_NAME} -> ${OUT}"
