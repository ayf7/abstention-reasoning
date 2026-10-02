python -m pipeline generate --task sql --method baseline --async \
    --model Qwen/Qwen3-32B --split sft_train \
    --data-name sql_conceptual --num-samples 10 \
  --sample-strategy random 

python -m pipeline generate --task sql --method baseline --async \
    --model Qwen/Qwen3-32B --split sft_val \
    --data-name sql_conceptual --num-samples 10 \
  --sample-strategy random

