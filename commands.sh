#!/bin/bash
# =============================================================================
# commands.sh — Organized pipeline commands reference
# =============================================================================

# =============================================================================
# SLURM LOGIN NODES
# =============================================================================

## Public GPU nodes
# srun --partition=default_partition --mem=32G --gres=gpu:nvidia_rtx_6000_ada_generation:1 --ntasks=1 --cpus-per-task=8 --time=48:00:00 --pty /bin/bash
# srun -p gpu --pty --job-name=job-name --nodes=1 --gres gpu:nvidia_h100_nvl:1 --cpus-per-task=4 --mem-per-cpu=16G --time=48:00:00 /bin/bash
# srun -p gpu --pty --job-name=job-name --nodes=1 --gres gpu:nvidia_h200_nvl:1 --cpus-per-task=4 --mem-per-cpu=16G --time=48:00:00 /bin/bash
# srun -p gpu --pty --job-name=job-name --nodes=1 --gres gpu:nvidia_b200:1 --cpus-per-task=4 --mem-per-cpu=16G --time=48:00:00 /bin/bash

## H100 / B200 (Goyal partition)
# srun -p goyal-interactive --pty --nodes=1 --cpus-per-task=16 --mem=100G --time=48:00:00 --gres=gpu:1 /bin/bash
# srun -p genai-goyal-highpri --pty --job-name=job-name --nodes=1 --gres gpu:nvidia_b200:1 --cpus-per-task=16 --mem=100G --time=48:00:00 /bin/bash
# srun -p genai --pty --job-name=job-name --nodes=1 --gres gpu:nvidia_b200:1 --cpus-per-task=4 --mem-per-cpu=16G --time=48:00:00 /bin/bash

## A6000 (Abdelfattah partition)
# srun --partition=abdelfattah-interactive --account=abdelfattah --nodelist=abdelfattah-compute-01 --mem=100000 --gres=gpu:0 --ntasks=4 --pty /bin/bash -l


# =============================================================================
# COMPETITION MATH
# =============================================================================

# --- Data Generation --------------------------------------------------------

python -m pipeline generate --task competition_math --method hint \
    --model Qwen/Qwen3-14B --split sft \
    --force-hints-distribution "1:0.5,2:0.3,3:0.2" \
    --force-hints-policy "1:0.05,2:0.05,3:0.05,4:0.15,5:0.15" --async

# Hint analysis: create prompts with 0-5 hints baked in
for i in 0 1 2 3 4 5; do
    python -m pipeline create_prompts --task competition_math --method hint_analysis \
        --split eval --num-hints $i \
        --output artifacts/competition_math/hint_analysis/prompts_${i}hints/
done

python create_abstention_from_hint.py \
    --input artifacts/competition_math/hint/datasets/sft_qwen3-14b.json \
    --output artifacts/competition_math/simple_abstention/datasets/sft_from_hint.json

# --- SFT Training -----------------------------------------------------------

python -m pipeline train_sft --task competition_math --method hint \
    --base-model Qwen/Qwen2.5-1.5B

python -m pipeline train_sft --task competition_math --method simple_abstention \
    --dataset artifacts/competition_math/simple_abstention/datasets/sft_from_hint.json \
    --base-model Qwen/Qwen2.5-1.5B \
    --run-id from_hint --include-abstained

# --- RL Training -------------------------------------------------------------

# Simple baseline
python -m pipeline train_rl --task competition_math --method simple \
    --total-steps 400 \
    --sft-model artifacts/competition_math/simple/models/sft/qwen2.5-1.5b/model \
    --train-batch-size 64 --n-samples 4

# Hint (standard penalty)
python -m pipeline train_rl --task competition_math --method hint \
    --sft-model artifacts/competition_math/hint/models/sft/default/model \
    --train-batch-size 64 --n-samples 16 --total-steps 800 --keep-state --run-id new2

# Hint encourage (reward=1)
python -m pipeline train_rl --task competition_math --method hint_encourage \
    --sft-model artifacts/competition_math/hint/models/sft/default/model \
    --train-batch-size 64 --n-samples 16 --total-steps 800 --keep-state --run-id new2

# Hint encourage 2 (reward=2)
python -m pipeline train_rl --task competition_math --method hint_encourage_2 \
    --sft-model artifacts/competition_math/hint/models/sft/default/model \
    --train-batch-size 64 --n-samples 16 --total-steps 800 --keep-state --run-id new2

# Simple abstention
# from_hint: trained with from_hint SFT data
python -m pipeline train_rl --task competition_math --method simple_abstention \
    --sft-model artifacts/competition_math/simple_abstention/models/sft/from_hint/model \
    --run-id fixed_metric \
    --train-batch-size 64 --n-samples 16 --total-steps 400

# Hint adjusted (group-level hint normalization)
python -m pipeline train_rl --task competition_math --method hint_adjusted \
    --sft-model artifacts/competition_math/hint/models/sft/default/model \
    --train-prompts artifacts/competition_math/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/competition_math/hint/prompts/rl_val.parquet \
    --train-batch-size 64 --n-samples 16 --total-steps 800 --keep-state --run-id default

# --- Evaluation --------------------------------------------------------------

python -m pipeline evaluate --task competition_math --method idea_1 \
    --model Qwen/Qwen3-14B --async

# Hint analysis: evaluate with 0-5 hints (SFT model)
for i in 0 1 2 3 4 5; do
    python -m pipeline evaluate --task competition_math \
        --prompts artifacts/competition_math/hint_analysis/prompts_${i}hints/eval.json \
        --output artifacts/competition_math/hint_analysis/results/eval_${i}hints.json \
        --model artifacts/competition_math/simple/models/sft/qwen2.5-1.5b/model --async
done

# Hint analysis: evaluate with 0-5 hints (Qwen3-14B)
for i in 0 1 2 3 4 5; do
    python -m pipeline evaluate --task competition_math \
        --prompts artifacts/competition_math/hint_analysis/prompts_${i}hints/eval.json \
        --output artifacts/competition_math/hint_analysis/results_14b/eval_${i}hints.json \
        --model Qwen/Qwen3-14B --async
done

# Hint smart evaluation
python -m pipeline evaluate --task competition_math --method hint_smart \
    --model artifacts/competition_math/hint/models/sft/default/model \
    --output artifacts/competition_math/hint_smart/results/eval_sft_hint_smart.json \
    --gpu-memory-utilization 0.4 --helper-gpu-memory-utilization 0.4

# Hint RL model evaluation (standard penalty, run new2)  # 49%
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint/results/eval_augmented_new.json \
    --model artifacts/competition_math/hint/models/rl/new2/model \
    --multi-turn --async

# Hint RL intermediate checkpoint (400 steps)  # 48%
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint/results/eval_augmented_new.json \
    --model artifacts/competition_math/hint/models/rl/new2/temp/global_step_400/actor/huggingface \
    --multi-turn --async

# Hint encourage RL (old run, reward=1)  # 75%
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval.json \
    --output artifacts/competition_math/hint_encourage/results/eval_new.json \
    --model artifacts/competition_math/hint_encourage/models/rl/new/model \
    --multi-turn --async

# Hint encourage RL (new2, reward=1)  # 51%
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint_encourage/results/eval_augmented_new.json \
    --model artifacts/competition_math/hint_encourage/models/rl/new2/model \
    --multi-turn --async

# Hint encourage 2 RL (new2, reward=2)
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint_encourage_2/results/eval_augmented_new2.json \
    --model artifacts/competition_math/hint_encourage_2/models/rl/new2/model \
    --multi-turn --async

# Hint encourage 2 RL intermediate (new2, 400 steps)
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint_encourage_2/results/eval_augmented_new2_400.json \
    --model artifacts/competition_math/hint_encourage_2/models/rl/new2/temp/actor/huggingface \
    --multi-turn --async

# Simple hint -> eval augmented
python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/hint/prompts/eval_augmented.json \
    --output artifacts/competition_math/hint/results/eval_augmented_new2.json \
    --model artifacts/competition_math/hint/models/rl/new2/model \
    --multi-turn --async

# Simple abstention RL
python -m pipeline evaluate --task competition_math --method simple_abstention \
    --model artifacts/competition_math/simple_abstention/models/rl/default/model \
    --split eval --async

python -m pipeline evaluate --task competition_math \
      --prompts artifacts/competition_math/simple_abstention/prompts/eval_augmented.json \
      --output artifacts/competition_math/simple_abstention/results/eval_augmented_fixed_metric.json \
      --model artifacts/competition_math/simple_abstention/models/rl/fixed_metric/model \
      --async

python -m pipeline evaluate --task competition_math \
    --prompts artifacts/competition_math/simple/prompts/eval_augmented.json \
    --output artifacts/competition_math/simple/results/eval_augmented_default.json \
    --model artifacts/competition_math/simple/models/rl/default/model \
    --async
# --- Checkpoint Conversion ---------------------------------------------------

python -m pipeline convert_checkpoint \
    --checkpoint artifacts/competition_math/hint_encourage_2/models/rl/new2/temp/actor \
    --output artifacts/competition_math/hint_encourage_2/models/rl/new2/temp/actor/huggingface

python -m pipeline convert_checkpoint \
    --checkpoint artifacts/competition_math/hint_encourage/models/rl/new2/temp/actor \
    --output artifacts/competition_math/hint_encourage/models/rl/new2/temp/actor/huggingface

python -m pipeline convert_checkpoint \
    --checkpoint artifacts/competition_math/hint/models/rl/new2/temp/global_step_400/actor \
    --output artifacts/competition_math/hint/models/rl/new2/temp/global_step_400/actor/huggingface


# =============================================================================
# COUNTDOWN
# =============================================================================

# --- RL Training -------------------------------------------------------------

# Hint (standard penalty)
python -m pipeline train_rl --task countdown --method hint \
    --run-id new2 \
    --max-response-length 2048 --max-model-len 8192

# Hint encourage (reward=1)
python -m pipeline train_rl --task countdown --method hint_encourage \
    --sft-model artifacts/countdown/hint/models/sft/default/model \
    --train-prompts artifacts/countdown/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/countdown/hint/prompts/rl_val.parquet \
    --total-steps 400 --run-id default

# Hint adjusted (group-level hint normalization)
python -m pipeline train_rl --task countdown --method hint_adjusted \
    --train-prompts artifacts/countdown/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/countdown/hint/prompts/rl_val.parquet \
    --max-response-length 2048 --max-model-len 8192 \
    --run-id default

# --- Evaluation --------------------------------------------------------------

# Hint analysis: evaluate with 0-5 hints (Qwen3-14B)
for i in 0 1 2 3 4 5; do
    python -m pipeline evaluate --task countdown \
        --prompts artifacts/countdown/hint_analysis/prompts_${i}hints/eval.json \
        --output artifacts/countdown/hint_analysis/results/eval_${i}hints.json \
        --model Qwen/Qwen3-14B --async
done

# Hint encourage RL evaluation
python -m pipeline evaluate --task countdown --method hint_encourage \
    --model artifacts/countdown/hint_encourage/models/rl/default/model \
    --prompts artifacts/countdown/hint/prompts/eval.json \
    --multi-turn --async

# Hint RL evaluation (run new2)
python -m pipeline evaluate --task countdown --method hint \
    --model artifacts/countdown/hint/models/rl/new2/model \
    --prompts artifacts/countdown/hint/prompts/eval.json \
    --multi-turn --async


# =============================================================================
# CODE OUTPUT
# =============================================================================

# --- Data Generation --------------------------------------------------------

# Create primitives
python -m pipeline create_primitives --task code_output --num-puzzles 5000

# Create all prompts (simple)
python -m pipeline create_prompts --task code_output --method simple --split all

# Create all prompts (simple_abstention)
python -m pipeline create_prompts --task code_output --method simple_abstention --split all

# Create all prompts (hint)
python -m pipeline create_prompts --task code_output --method hint --split all

# Generate CoT dataset (simple, for SFT)
python -m pipeline generate --task code_output --method simple \
    --model Qwen/Qwen3-14B --split sft --async

# Generate CoT dataset (simple abstain for SFT)
python -m pipeline generate --task code_output --method simple_abstention \
    --model Qwen/Qwen3-14B --split sft --async

# Generate CoT dataset (hint, for SFT)
python -m pipeline generate --task code_output --method hint \
    --model Qwen/Qwen3-14B --split sft --multi-turn --async

# Create abstention SFT data from hint dataset
python create_abstention_from_hint.py \
    --input artifacts/code_output/hint/datasets/sft_qwen3-14b.json \
    --output artifacts/code_output/simple_abstention/datasets/sft_from_hint.json

# --- SFT Training -----------------------------------------------------------

# Simple baseline
python -m pipeline train_sft --task code_output --method simple \
    --dataset artifacts/code_output/simple/datasets/sft_qwen3-14b.json \
    --base-model Qwen/Qwen2.5-1.5B

# Simple abstention (from hint data)
python -m pipeline train_sft --task code_output --method simple_abstention \
    --dataset artifacts/code_output/simple_abstention/datasets/sft_from_hint.json \
    --base-model Qwen/Qwen2.5-1.5B \
    --run-id from_hint --include-abstained

# Hint
python -m pipeline train_sft --task code_output --method hint \
    --dataset artifacts/code_output/hint/datasets/sft_qwen3-14b.json \
    --base-model Qwen/Qwen2.5-1.5B

# --- RL Training -------------------------------------------------------------

# Simple baseline
python -m pipeline train_rl --task code_output --method simple \
    --train-batch-size 64 --n-samples 16 --total-steps 400 \
    --max-prompt-length 2560

# Simple abstention (from hint SFT)
python -m pipeline train_rl --task code_output --method simple_abstention \
    --sft-model artifacts/code_output/simple_abstention/models/sft/from_hint/model \
    --train-batch-size 64 --n-samples 16 --total-steps 400 \
    --max-prompt-length 2560 \
    --run-id from_hint

# Hint
python -m pipeline train_rl --task code_output --method hint \
    --train-batch-size 64 --n-samples 16 --total-steps 400 \
    --max-prompt-length 2560 --max-model-len 8192

# --- Evaluation --------------------------------------------------------------

# SFT model
python -m pipeline evaluate --task code_output --method simple --model sft --async

# Simple RL model
python -m pipeline evaluate --task code_output --method simple --model rl --async

# Simple abstention RL model
python -m pipeline evaluate --task code_output --method simple_abstention \
    --model artifacts/code_output/simple_abstention/models/rl/from_hint/model --async

# Hint RL model
python -m pipeline evaluate --task code_output --method hint \
    --model rl --multi-turn --async

# Base model (Qwen3-14B)
python -m pipeline evaluate --task code_output --method simple \
    --model Qwen/Qwen3-14B --async


# =============================================================================
# CHESS PUZZLES
# =============================================================================

# --- Data Generation --------------------------------------------------------

python -m pipeline generate --task chess_puzzles --method hint \
    --model Qwen/Qwen3-32B --split sft --multi-turn --async \
    --force-hints-distribution "0:0.66,1:0.11,2:0.11,3:0.12" \
    --num-samples 16

# --- SFT Training -----------------------------------------------------------

python -m pipeline train_sft --task chess_puzzles --method hint \
    --dataset artifacts/chess_puzzles/hint/datasets/sft_qwen3-32b.json \
    --base-model Qwen/Qwen2.5-1.5B --include-wrong-valid-format

# --- RL Training -------------------------------------------------------------

# Hint (standard penalty)
python -m pipeline train_rl --task chess_puzzles --method hint \
    --train-batch-size 64 --n-samples 16 \
    --total-steps 400 --save-freq 25 --test-freq 10 --keep-checkpoints \
    --max-model-len 4096

# Hint encourage
python -m pipeline train_rl --task chess_puzzles --method hint_encourage \
    --sft-model artifacts/chess_puzzles/hint/models/sft/default/model \
    --train-prompts artifacts/chess_puzzles/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/chess_puzzles/hint/prompts/rl_val.parquet \
    --train-batch-size 64 --n-samples 16 \
    --total-steps 400 --save-freq 25 --test-freq 10 --keep-checkpoints \
    --max-model-len 4096

# Hint 1 only (default)
python -m pipeline train_rl --task chess_puzzles --method hint_1_only \
    --run-id default_0.1 \
    --sft-model artifacts/chess_puzzles/hint/models/sft/default/model \
    --train-prompts artifacts/chess_puzzles/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/chess_puzzles/hint/prompts/rl_val.parquet \
    --train-batch-size 64 --n-samples 16 \
    --total-steps 400 --save-freq 25 --test-freq 10 --keep-checkpoints \
    --max-model-len 4096

# Hint 1 only (min diversity 0.1)
python -m pipeline train_rl --task chess_puzzles --method hint_1_only \
    --run-id minDiv02 \
    --sft-model artifacts/chess_puzzles/hint/models/sft/default/model \
    --train-prompts artifacts/chess_puzzles/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/chess_puzzles/hint/prompts/rl_val.parquet \
    --train-batch-size 64 --n-samples 16 \
    --total-steps 400 --save-freq 25 --test-freq 10 --keep-checkpoints \
    --max-model-len 4096 \
    --min-diversity 0.1

# Hint adjusted (group-level hint normalization)
python -m pipeline train_rl --task chess_puzzles --method hint_adjusted \
    --sft-model artifacts/chess_puzzles/hint/models/sft/default/model \
    --train-prompts artifacts/chess_puzzles/hint/prompts/rl_train.parquet \
    --val-prompts artifacts/chess_puzzles/hint/prompts/rl_val.parquet \
    --train-batch-size 64 --n-samples 16 \
    --total-steps 400 --save-freq 25 --test-freq 10 --keep-checkpoints \
    --max-model-len 4096
