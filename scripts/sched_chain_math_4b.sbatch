#!/bin/bash
# ============================================================================
# competition_math / hint_ablations - regenerate the scheduled-hint SFT data on
# this cluster, SFT Qwen3-4B-Base on it, then RL. One stage per submission,
# chained with afterok so each waits for the previous to finish cleanly:
#
#   G=$(sbatch --parsable -t 12:00:00 scripts/sched_chain_math_4b.sbatch generate)
#   S=$(sbatch --parsable -t 6:00:00 --dependency=afterok:$G scripts/sched_chain_math_4b.sbatch sft)
#   sbatch -p genai-goyal-highpri --dependency=afterok:$S \
#       scripts/rl_math_v2_sched_sweep.sbatch 0.05 qwen3-4b-sched-n8-r2-tok-fullloss r2
#
# The generate stage reproduces sft_scheduled_n8_qwen3-14b.json, which was made
# on the empireai cluster with the code now in pipeline/core/hint_schedule.py
# and the --hint-schedule flags of `pipeline generate`:
#   1. probe     Qwen3-14B on the sft split, hint requests banned, 8 samples per
#                problem. Pass rate per problem = the difficulty profile.
#   2. schedule  rank problems hardest first (pass rate, then reasoning tokens);
#                the hardest 50% get hints, in five equal bands of 5..1 hints.
#   3. generate  8 rollouts per problem with the scheduled hints injected; drop
#                samples that fire requests back to back or answer straight off
#                a hint (--min-words-after-hint), then keep the correct sample
#                with the most hints (most_hints); drop a problem when every
#                sample ran the 5-hint pool dry or rapid-fired.
#   4. filter    correct rows only -> the SFT dataset.
# Flags are read off the remote artifacts (n=8, max_new_tokens 4096, 50% hinted,
# up to 5 hints, no exhausted rows). Temperature / top_p were not recorded and
# use generate's defaults (0.7 / 0.9).
#
# One deliberate difference from the remote dataset: the rapid-fire filter. There,
# 275 of 808 hinted rows were the teacher requesting all five hints back to back
# and answering off the last one (no 2-4 hint row did this), and the SFT student
# learned that as its main hinted behaviour: RL from it converged to "try, then
# dump five hints, then answer".
#
# Outputs carry "-r2" so nothing overwrites the remote-made files, and the SFT
# run id / RL run id stay apart from the running sweep. Every stage skips work
# whose output already exists, which is what --requeue after a preemption
# relies on (generate saves only at the end of each phase, so a preemption
# mid-phase redoes that phase).
# ============================================================================
#SBATCH --job-name=sched-chain-4b
#SBATCH --output=/home/ayf7/repos/abstention-reasoning/logs/sched_chain_%j.out
#SBATCH --error=/home/ayf7/repos/abstention-reasoning/logs/sched_chain_%j.err
#SBATCH --time=12:00:00
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

STAGE="${1:?usage: sbatch scripts/sched_chain_math_4b.sbatch <generate|sft>}"

REPO_ROOT="/home/ayf7/repos/abstention-reasoning"
CONDA_ENV="/home/ayf7/env-b200"
TASK="competition_math"
METHOD="hint_ablations"
TAG="r2"

TEACHER="Qwen/Qwen3-14B"
BASE_MODEL="/share/goyal/ayf7/artifacts/base_models/Qwen3-4B-Base-plaintags"
DATASETS="artifacts/${TASK}/${METHOD}/datasets"
PROBE="${DATASETS}/probe_nohint_qwen3-14b-${TAG}.json"
PROFILE="${DATASETS}/difficulty_qwen3-14b-${TAG}.json"
RAW="${DATASETS}/sft_scheduled_n8_qwen3-14b-${TAG}.with_incorrect.json"
DATASET="${DATASETS}/sft_scheduled_n8_qwen3-14b-${TAG}.json"
SFT_RUN="qwen3-4b-sched-n8-${TAG}-tok-fullloss"
OUT_MODEL="artifacts/${TASK}/${METHOD}/models/sft/${SFT_RUN}/model"

NUM_SAMPLES=8
MAX_NEW_TOKENS=4096
HINT_FRACTION=0.5
MAX_HINTS=5
# Reasoned rows in the remote data have 100+ words between hints; rapid-fired
# ones have 0-1. Anything in between is rare, so the cut is not delicate.
MIN_WORDS_AFTER_HINT=20
SEED=42

scontrol update JobId="${SLURM_JOB_ID}" JobName="sched-chain-${STAGE}" 2>/dev/null || true

cd "$REPO_ROOT"
source /share/apps/software/anaconda3/bin/activate "$CONDA_ENV"
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"
# Qwen3-14B is resolved through the hub cache; set it here rather than rely on
# the submitting shell's environment.
export HF_HOME="${HF_HOME:-/share/goyal/ayf7/huggingface}"

echo "=============================================="
echo " scheduled-hint chain - stage: ${STAGE}"
echo "=============================================="
echo " teacher:       ${TEACHER}"
echo " profile:       ${PROFILE}"
echo " dataset:       ${DATASET}"
echo " sft run:       ${SFT_RUN}"
echo " job / restart: ${SLURM_JOB_ID} / ${SLURM_RESTART_COUNT:-0}"
echo " partition:     ${SLURM_JOB_PARTITION:-none}"
echo " node / gpu:    $(hostname) / ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

case "$STAGE" in
generate)
    if [ -f "$DATASET" ]; then
        echo "Dataset already present, nothing to do: ${DATASET}"
        exit 0
    fi

    if [ -f "$PROFILE" ]; then
        echo "Difficulty profile present, skipping probe: ${PROFILE}"
    else
        echo "--- phase 1: no-hint probe, ${NUM_SAMPLES} samples per problem"
        python -m pipeline generate \
            --task "${TASK}" --method "${METHOD}" --split sft \
            --model "${TEACHER}" \
            --output "${PROBE}" \
            --no-hints \
            --num-samples "${NUM_SAMPLES}" \
            --difficulty-profile "${PROFILE}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --seed "${SEED}"
    fi

    echo "--- phase 2+3: scheduled hints (${HINT_FRACTION} hinted, <=${MAX_HINTS} hints), ${NUM_SAMPLES} rollouts per problem, >=${MIN_WORDS_AFTER_HINT} words after each hint"
    python -m pipeline generate \
        --task "${TASK}" --method "${METHOD}" --split sft \
        --model "${TEACHER}" \
        --output "${RAW}" \
        --hint-schedule "${PROFILE}" \
        --hint-target-fraction "${HINT_FRACTION}" \
        --max-hints "${MAX_HINTS}" \
        --num-samples "${NUM_SAMPLES}" \
        --sample-strategy most_hints \
        --drop-exhausted \
        --min-words-after-hint "${MIN_WORDS_AFTER_HINT}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}"

    echo "--- phase 4: keep correct rows"
    python - "${RAW}" "${DATASET}" "${MIN_WORDS_AFTER_HINT}" <<'PY'
import sys
from collections import Counter
from pipeline.core.io import load_json, save_json
from pipeline.commands.inference import min_words_after_hints

raw, out, min_words = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = load_json(raw)
keep = [r for r in rows if r.get("correct")]
save_json(out, keep)
hints = Counter(r.get("num_hints", 0) for r in keep)
print(f"{len(keep)}/{len(rows)} correct -> {out}")
print("num_hints:", " ".join(f"{h}:{hints[h]}" for h in sorted(hints)))
rapid = sum(1 for r in keep
            if (w := min_words_after_hints(r["generation"])) is not None and w < min_words)
print(f"rows with < {min_words} words after a hint (should be 0): {rapid}")
PY
    ;;

sft)
    # SFT keeps only the final weights (save_strategy=no): nothing to resume,
    # nothing to redo if the model is already written.
    if compgen -G "${OUT_MODEL}/*.safetensors" > /dev/null; then
        echo "SFT model already present, nothing to do: ${OUT_MODEL}"
        exit 0
    fi
    if [ ! -f "$DATASET" ]; then
        echo "FATAL: dataset missing, run the generate stage first: ${DATASET}" >&2
        exit 1
    fi
    # Same recipe as sft_math_sched_4b.sbatch (the running sweep's parent):
    # train_sft defaults, no --upsample-hint, full-sequence loss.
    python -m pipeline train_sft \
        --task "${TASK}" \
        --method "${METHOD}" \
        --base-model "${BASE_MODEL}" \
        --run-id "${SFT_RUN}" \
        --dataset "${DATASET}" \
        --project-name math_new_hint \
        --experiment-name "sft-v2-${SFT_RUN}"
    ;;

*)
    echo "FATAL: unknown stage '${STAGE}' (generate|sft)" >&2
    exit 2
    ;;
esac

echo "Stage ${STAGE} complete."
