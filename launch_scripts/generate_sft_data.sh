python -m pipeline generate --task math --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_train \
    --data-name math_o1 --num-samples 10 \
  --sample-strategy random 

python -m pipeline generate --task math --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_val \
    --data-name math_o1 --num-samples 10 \
  --sample-strategy random

