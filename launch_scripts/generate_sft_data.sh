python -m pipeline generate --task math --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_train \
    --data-name math_o1 --num-samples 10 \
  --sample-strategy random 

python -m pipeline generate --task math --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_val \
    --data-name math_o1 --num-samples 10 \
  --sample-strategy random

# The repo clone this script runs in is ephemeral (pod-local), so generated
# data under data/ must be pushed back to git or it's lost when the pod exits.
git add data
git commit -m "Generated data"
git push 