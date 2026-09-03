#!/bin/bash
# Feasibility spike: build ONE small CUDA extension against torch 2.13 + CUDA 13.0.
set -x
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="10.0"          # B200 only, keeps the spike short
export MAX_JOBS=64
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE

BUILD=/workspace/wheelbuild
mkdir -p $BUILD/out && cd $BUILD

# torch 2.13 build env
if [ ! -d venv213 ]; then
  uv venv -q venv213 --python 3.12
  VIRTUAL_ENV=$BUILD/venv213 uv pip install -q --index https://download.pytorch.org/whl/cu130 "torch==2.13.0"
  VIRTUAL_ENV=$BUILD/venv213 uv pip install -q setuptools wheel ninja packaging psutil
fi
$BUILD/venv213/bin/python -c "import torch;print('build torch:',torch.__version__)"
nvcc --version | tail -2

time VIRTUAL_ENV=$BUILD/venv213 uv pip wheel --no-build-isolation \
  --wheel-dir $BUILD/out "causal-conv1d==1.6.2.post1"
ls -la $BUILD/out/
