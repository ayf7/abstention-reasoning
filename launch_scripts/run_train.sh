#!/bin/bash
# Generic train_sft + train_rl runner for one (task, model) combo, using the
# baseline method and the math_o1 data/models namespace.
#
# Usage: bash run_train.sh <task> <model_short_name> <hf_base_model>
#   e.g. bash run_train.sh math qwen2.5-1.5b Qwen/Qwen2.5-1.5B

set -euo pipefail

TASK="$1"
MODEL="$2"
HF_MODEL="$3"
RUN_ID="${MODEL}"

# python -m pipeline train_sft \
#   --task "${TASK}" \
#   --method baseline \
#   --run-id "${RUN_ID}" \
#   --base-model "${HF_MODEL}" \
#   --data-name math_o1

python -m pipeline train_rl \
  --task "${TASK}" \
  --method baseline \
  --run-id "${RUN_ID}_seqmean" \
  --sft-model "models/math_o1/baseline_sft/${RUN_ID}/model" \
  --data-name math_o1 \
  --overwrite  \
  --override actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  --override actor_rollout_ref.rollout.val_kwargs.temperature=1 \
  --override actor_rollout_ref.rollout.val_kwargs.n=8 #\#--override actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean


