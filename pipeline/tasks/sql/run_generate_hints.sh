#!/usr/bin/env bash
# Run TRAPI hint generation across all Text-to-SQL datasets and splits.
#
# Usage:
#   # Run conceptual hints on dev splits:
#   bash pipeline/tasks/sql/run_generate_hints.sh --style conceptual --split dev
#
#   # Run partial_sql hints on all splits (overnight):
#   bash pipeline/tasks/sql/run_generate_hints.sh --style partial_sql --split all
#
#   # Run both styles across all splits:
#   bash pipeline/tasks/sql/run_generate_hints.sh --style both --split all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/sql"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Defaults
STYLE="both"      # conceptual, partial_sql, both
SPLIT="all"       # dev, train, all
BATCH_SIZE=20
FILTER="complex"  # complex, multi_step, all
MODEL="gpt-5.6-sol_2026-07-09"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --style)
            STYLE="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --filter|-f)
            FILTER="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--style conceptual|partial_sql|both] [--split dev|train|all] [--batch-size N] [--filter complex|multi_step|all]"
            exit 1
            ;;
    esac
done

cd "${REPO_ROOT}"

# Define files to process based on SPLIT
DEV_FILES=(
    "primitives_spider_dev.json"
    "primitives_bird_dev.json"
    "primitives_sparc_dev.json"
    "primitives_cosql_dev.json"
)

TRAIN_FILES=(
    "primitives_spider_train.json"
    "primitives_spider_train_others.json"
    "primitives_bird_train.json"
    "primitives_sparc_train.json"
    "primitives_cosql_train.json"
)

TARGET_FILES=()
if [[ "${SPLIT}" == "dev" ]]; then
    TARGET_FILES=("${DEV_FILES[@]}")
elif [[ "${SPLIT}" == "train" ]]; then
    TARGET_FILES=("${TRAIN_FILES[@]}")
elif [[ "${SPLIT}" == "all" ]]; then
    TARGET_FILES=("${DEV_FILES[@]}" "${TRAIN_FILES[@]}")
else
    echo "Invalid --split: ${SPLIT}. Choose dev, train, or all."
    exit 1
fi

STYLES_TO_RUN=()
if [[ "${STYLE}" == "conceptual" ]]; then
    STYLES_TO_RUN=("conceptual")
elif [[ "${STYLE}" == "partial_sql" ]]; then
    STYLES_TO_RUN=("partial_sql")
elif [[ "${STYLE}" == "both" ]]; then
    STYLES_TO_RUN=("conceptual" "partial_sql")
else
    echo "Invalid --style: ${STYLE}. Choose conceptual, partial_sql, or both."
    exit 1
fi

echo "================================================================================"
echo "TRAPI Hint Generation"
echo "Styles:      ${STYLES_TO_RUN[*]}"
echo "Splits:      ${SPLIT} (${#TARGET_FILES[@]} files)"
echo "Filter:      ${FILTER}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Model:       ${MODEL}"
echo "Log dir:     ${LOG_DIR}"
echo "================================================================================"

for s in "${STYLES_TO_RUN[@]}"; do
    echo ""
    echo ">>> Starting Hint Style: ${s}"
    echo ""
    for file in "${TARGET_FILES[@]}"; do
        in_path="${DATA_DIR}/${file}"
        base_name="${file%.json}"
        out_path="${DATA_DIR}/${base_name}_hints_${s}.json"
        log_path="${LOG_DIR}/${base_name}_hints_${s}.log"

        echo "--------------------------------------------------------------------------------"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing: ${file} (style=${s})"
        echo "  Input:  ${in_path}"
        echo "  Output: ${out_path}"
        echo "  Log:    ${log_path}"

        python -m pipeline.tasks.sql.generate_hints \
            --input "${in_path}" \
            --output "${out_path}" \
            --model "${MODEL}" \
            --hint-style "${s}" \
            --filter "${FILTER}" \
            --batch-size "${BATCH_SIZE}" \
            2>&1 | tee -a "${log_path}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${file} (style=${s})"
    done
done

echo ""
echo "================================================================================"
echo "All runs completed successfully at $(date '+%Y-%m-%d %H:%M:%S')!"
echo "================================================================================"
