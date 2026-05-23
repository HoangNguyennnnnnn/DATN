#!/usr/bin/env bash
# Cài causal-conv1d + mamba-ssm (compile CUDA). Chạy sau requirements.txt + torch OK.
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Use a CUDA *devel* image or install cuda-toolkit; set CUDA_HOME." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "CUDA_HOME=$CUDA_HOME  MAX_JOBS=$MAX_JOBS"
pip install causal-conv1d==1.6.1
pip install mamba-ssm==2.3.1

python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
