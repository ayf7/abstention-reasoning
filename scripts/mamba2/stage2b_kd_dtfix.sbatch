#!/bin/bash
# Stage-2b main run, Mamba2 arm: 294M-token length-extension KD @32k on stage2b_mix_v1,
# initialised from this arm's stage-2a checkpoint. 4xH100-80 on cornell.
#
# No new code runs here. src/training/stage2_kd.py is the same file stage 2a used, unedited;
# 2b is that trainer with (micro_batch, seq_len) = (1, 32768) and the LR halved. The path has
# now executed five times across three arms, so what is new is the seq_len value for THIS
# mixer and nothing else.
#
# Preconditions, and how each is known:
#   * memory -- PROJECTED at ~53.4 GB / 80, not measured. The Qwen GDN arm is a tight control
#     (same parameter class, same trainer, same 4 ranks, same mb4x4096 -> mb1x32768 step) and it
#     moved 46.6 -> 54.3 GB with no offload; this arm's 2a measured 45.7 GB, 0.9 GB under GDN's
#     at an identical config. ~26 GB spare. This is deliberately not gated by a separate smoke:
#     peak is reached inside the first two optimizer steps (2a measured 37.4 GB at step 1, 45.7
#     at step 2), so an underestimate aborts in ~2 minutes and costs queue time, not GPU hours.
#     If it does OOM, resubmit with `--override micro_batch=1 --override offload_optimizer=true`
#     -- MiMo's 7B 2b ran that way at 65.3 GB.
#   * throughput -- PROJECTED at ~8.5k tok/s from the GDN arm's 2a->2b ratio (25,071 -> 7,982,
#     x0.318) applied to this arm's measured 26,795. Corroborated by 2b throughput being nearly
#     size-independent across arms: GDN 4B hit 7,982 and MiMo 7B hit 8,101, which says 32k step
#     time is dominated by machinery both arms share (attention in the 9 retained layers, plus
#     fp32 logit chunking over a 151,936 vocab) rather than by the mixer.
#   * LR -- 1e-4/1e-5 is 2a's probe-confirmed 2e-4/2e-5 halved per the recipe. No probe needed.
#
# Walltime: 294e6/8.5k = 9.61 h compute, plus ~14 evals at 524k val tokens each, 5 optimizer-shard
# saves and the cold 4-rank load = ~10.3 h, x1.2 = 12.4 -> 13 h by the house rule. Requesting 14:
# the extra hour buys against the throughput figure being projected rather than measured, and a
# walltime cut costs a requeue plus a cold reload, which is dearer than an idle hour.
#
# Cuts are recoverable: resume: true with save_every 25 writes trainer_state.pt (optimizer,
# scheduler, step, wandb id) about every 52M tokens, and resubmitting THIS SAME SCRIPT continues
# the same cosine schedule rather than restarting it. Do NOT delete output_dir between attempts
# -- that is the resume state.
#
# QOS: `test` has MaxWall 02:00:00 with DenyOnLimit and will reject this outright. `standard` is
# the default association QOS (MaxWall 2 days, priority 500). If it will not queue, resubmit
# unchanged with `sbatch --qos=long ...` (MaxWall 7 days, priority 200).
#
# GPU TYPE UNPINNED 2026-08-28: the gres was `gpu:nvidia_h100_80gb_hbm3:4`, which excludes the six
# H200 nodes (alphagpu19-24) and left job 51983 pending with an estimated start 33 h out -- no node
# in cornell had 4 free GPUs of any type, and the best H200 node had 3 free against the best H100
# node's 2. Untyped `gpu:4` keeps `-p cornell`, so the rtx6000_gpu_governance rewrite cannot pull
# this onto the Blackwell pool (alphagpu51-54, a different partition); `sbatch --test-only` confirmed
# it stays in cornell and moved the estimate 7 h earlier onto alphagpu22. An H200 has 141 GB against
# the measured 45.7 GB per-rank peak, so it is strictly more headroom, and it is faster, so the
# walltime below stays conservative. Data order, seeds and world size 4 are unchanged, so the run
# is still comparable with the GDN reference, which ran on H100s.
#SBATCH -J m2_s2b_dtfix
#SBATCH -p cornell
#SBATCH --qos=standard
#SBATCH --gres=gpu:4   # untyped on purpose -- see GPU TYPE UNPINNED note above
#SBATCH --cpus-per-task=12   # 4 ranks x OMP_NUM_THREADS=3; must move together or the cgroup oversubscribes
#SBATCH --mem=512G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/home/afei1/Linearization_afei/logs/slurm_output/m2_s2b_dtfix_%j.out
#SBATCH --error=/mnt/home/afei1/Linearization_afei/logs/slurm_output/m2_s2b_dtfix_%j.err
set -u

ROOT=/mnt/home/afei1/Linearization_afei
cd "$ROOT"

module load Python/3.10.15
export LD_LIBRARY_PATH=/mnt/lustre/cornell/afei1/lib:${LD_LIBRARY_PATH:-}
source /mnt/lustre/cornell/afei1/pip_env_mamba2/bin/activate   # mamba_ssm; see stage1_smoke.sbatch
export HF_HOME=/mnt/lustre/cornell/afei1/hf_cache
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/mnt/lustre/cornell/afei1/triton_cache
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR=$ROOT/logs/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
mkdir -p "$WANDB_DIR"

DATA=/mnt/lustre/cornell/afei1/data/stage2b_mix_v1
[ -f "$DATA/meta.json" ] || { echo "ABORT: $DATA not built"; exit 1; }
[ -f "$DATA/.build_done" ] || { echo "ABORT: $DATA build incomplete"; exit 1; }
[ -f /mnt/lustre/cornell/afei1/checkpoints/qwen3-4b-mamba2-u4-stage2a-dtfix/config.json ] \
  || { echo "ABORT: stage-2a student checkpoint missing"; exit 1; }

# The packed memmap must be 32k-native: stage2_mix_v1 is 4k and would silently train the
# wrong regime at seq_len 32768.
python - "$DATA" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1] + "/meta.json"))
assert int(m["seq_len"]) == 32768, f"ABORT: data seq_len={m['seq_len']}, expected 32768"
print(f"data OK: seq_len={m['seq_len']} train_tokens={m['splits']['train']['tokens']:,}")
PY

echo "=== node ==="; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "from transformers.models.bamba import modeling_bamba as m; assert m.is_fast_path_available, 'ABORT: slow path -- wrong venv'; print('fast path OK')" || exit 1

torchrun --standalone --nproc_per_node=4 -m src.training.stage2_kd \
  --config configs/stage2b_kd_mamba2_dtfix.yaml "$@"
echo "kd_exit=$?"
