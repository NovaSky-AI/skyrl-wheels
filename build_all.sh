#!/bin/bash
# Build the torch-2.13-coupled CUDA extensions SkyRL pins.
# One arg: package key (fa | mamba | te)
PKG=$1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0;10.0"     # H100 + B200
B=/workspace/wheelbuild
PY=$B/venv213/bin/python
mkdir -p $B/out

case $PKG in
  fa)
    # NOTE: 2.8.3 exactly, NOT 2.8.3.post1 -- TE gates flash-attn on max_version 2.8.3
    export MAX_JOBS=112
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    unset FLASH_ATTENTION_SKIP_CUDA_BUILD
    SPEC="flash-attn==2.8.3"
    ;;
  mamba)
    export MAX_JOBS=48
    export MAMBA_FORCE_BUILD=TRUE
    export MAMBA_FORCE_CXX11_ABI=TRUE
    SPEC="mamba-ssm==2.3.2.post1"
    ;;
  te)
    export MAX_JOBS=48
    export NVTE_FRAMEWORK=pytorch
    SPEC="transformer-engine-torch==2.16.0"
    ;;
  *) echo "unknown pkg $PKG"; exit 2;;
esac

echo "=== building $SPEC ==="
nvcc --version | tail -2
$PY -c "import torch; print('build torch:', torch.__version__)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"
time $PY -m pip wheel --no-build-isolation --no-deps --wheel-dir $B/out "$SPEC"
echo "=== exit=$? ==="
ls -la $B/out/ | grep -iE "flash|mamba|transformer"
