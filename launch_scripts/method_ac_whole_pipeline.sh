#EVRYTHING MUST BE DONE PARALLELY, THIS IS JUST INFO

#CREATE THE FORMATS
for TASK in countdown math; do
  python -m pipeline create_prompts --task "$TASK" --method baseline --split all --num-hints 0 \
    --output "artifacts/${TASK}/problems_with_format_v3"
done

#GENERATE THE SFT DATA
for TASK in countdown math; do
  python -m pipeline generate --task "${TASK}" --method method_ac --model Qwen/Qwen3-14B \
  --split sft_train --num-samples 8 --sample-strategy random_correct --async \
  --prompts artifacts/${TASK}/problems_with_format_v3/sft_train__method_ac.json \
  --output artifacts/${TASK}/sft_datasets/sft_train__method_ac_v3.json
done

#TRAIN SFT AND RL MODELS FOR GENERATOR/SOLVER
for TASK in countdown math; do
  for MODEL in qwen2.5-1.5b qwen2.5-3b qwen3-4b-base; do

    python -m pipeline train_sft \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --base-model artifacts/${TASK}/models/baseline_models/${MODEL}/model \
      --dataset artifacts/${TASK}/sft_datasets/sft_train__method_ac_v3.json


    python -m pipeline train_rl \
      --task "${TASK}" \
      --method method_ac \
      --run-id ${MODEL} \
      --sft-model artifacts/${TASK}/models/method_ac_sft/${MODEL}/model \
      --train-prompts artifacts/${TASK}/problems_with_format_v3/rl_train__method_ac.parquet \
      --val-prompts artifacts/${TASK}/problems_with_format_v3/rl_val__method_ac.parquet \
      --overwrite
  
  done
done

## RUN 10 GENERATIONS / datapoint using the SOLVER