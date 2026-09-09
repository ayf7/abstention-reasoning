#!/usr/bin/env bash
# Build the `abstention` conda env on the GB200 (aarch64) cluster.
#
# Target verified against:
#   aarch64 (Grace) | NVIDIA GB200, compute capability 10.0 (sm_100), 189 GB
#   driver 580.126.16 | glibc 2.39 | 4 GPUs per node | slurm partition `beta`
#
# Run from the repo root:   ./setup-env.sh
set -euo pipefail

PREFIX="${CONDA_PREFIX_ROOT:-$HOME/miniforge3-aarch64}"
ENV_NAME=abstention
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PREFIX/envs/$ENV_NAME"

# ---------------------------------------------------------------------------
# 0. aarch64 Miniforge.
#    A pre-existing ~/miniforge3 on this account is an x86-64 install and cannot
#    execute here ("cannot execute binary file: Exec format error"). Home is
#    shared with x86_64 machines, so it is left alone and this lives beside it.
# ---------------------------------------------------------------------------
if [ ! -x "$PREFIX/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge-aarch64.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash /tmp/miniforge-aarch64.sh -b -p "$PREFIX"
fi

# ---------------------------------------------------------------------------
# 1. Env + all declared dependencies.
# ---------------------------------------------------------------------------
"$PREFIX/bin/conda" env create -f "$REPO/environment.yml" -n "$ENV_NAME"

# ---------------------------------------------------------------------------
# 2. POST-INSTALL FIXUP A -- expose the CUDA headers where nvcc looks.
#    conda's aarch64 CUDA packages install headers under
#    targets/sbsa-linux/include, but FlashInfer's JIT build passes only
#    -isystem $CUDA_HOME/include. Without this, the JIT fails with
#    "fatal error: nvrtc.h: No such file or directory".
#    (Checked: zero filename collisions with the env's existing headers.)
# ---------------------------------------------------------------------------
ln -sfn "$ENV_DIR"/targets/sbsa-linux/include/* "$ENV_DIR/include/"

# ---------------------------------------------------------------------------
# 3. POST-INSTALL FIXUP B -- CUDA_HOME + the libcuda link stub.
#    FlashInfer JIT needs nvcc (CUDA_HOME) and links -lcuda. libcuda is the
#    CUDA *driver* library, shipped with the driver rather than the toolkit, so
#    link against the stub; the real libcuda.so.1 is loaded at runtime.
#    LIBRARY_PATH is consulted at link time only and cannot shadow runtime libs.
# ---------------------------------------------------------------------------
mkdir -p "$ENV_DIR/etc/conda/activate.d" "$ENV_DIR/etc/conda/deactivate.d"
cat > "$ENV_DIR/etc/conda/activate.d/cuda_home.sh" <<'EOF'
export _ABSTENTION_OLD_CUDA_HOME="${CUDA_HOME:-}"
export _ABSTENTION_OLD_LIBRARY_PATH="${LIBRARY_PATH:-}"
export CUDA_HOME="$CONDA_PREFIX"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
EOF
cat > "$ENV_DIR/etc/conda/deactivate.d/cuda_home.sh" <<'EOF'
if [ -n "${_ABSTENTION_OLD_CUDA_HOME:-}" ]; then export CUDA_HOME="$_ABSTENTION_OLD_CUDA_HOME"; else unset CUDA_HOME; fi
if [ -n "${_ABSTENTION_OLD_LIBRARY_PATH:-}" ]; then export LIBRARY_PATH="$_ABSTENTION_OLD_LIBRARY_PATH"; else unset LIBRARY_PATH; fi
unset _ABSTENTION_OLD_CUDA_HOME _ABSTENTION_OLD_LIBRARY_PATH
EOF

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
"$ENV_DIR/bin/python" -c "import torch, vllm, flash_attn; print('torch', torch.__version__, '| vllm', vllm.__version__, '| flash_attn', flash_attn.__version__)"
