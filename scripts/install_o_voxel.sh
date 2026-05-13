#!/usr/bin/env bash
# Cài o_voxel (Microsoft TRELLIS.2) — kéo theo CuMesh + FlexGEMM (compile CUDA).
#
# Yêu cầu: torch đã cài; image CUDA *devel* có nvcc; build tools.
#   apt-get update && apt-get install -y git build-essential cmake ninja-build
#
# Nếu vẫn fail: xem README (train LMDB-only có thể không cần o_voxel trên GPU server).
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc không có. Dùng image nvidia/cuda:*-devel hoặc cài cuda-toolkit." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version | head -3

# Cài trước phụ thuộc git (một số môi trường pip resolve o_voxel dễ fail nếu build song song).
# --no-build-isolation: dùng torch đã cài trong env khi build extension (tránh lỗi isolate).
echo "[1/3] CuMesh..."
pip install --no-cache-dir --no-build-isolation \
  "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git"

echo "[2/3] FlexGEMM..."
pip install --no-cache-dir --no-build-isolation \
  "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git"

echo "[3/3] o_voxel (TRELLIS.2)..."
pip install --no-cache-dir --no-build-isolation \
  "o_voxel @ git+https://github.com/microsoft/TRELLIS.2.git#subdirectory=o-voxel"

python -c "import o_voxel; print('o_voxel OK:', getattr(o_voxel, '__file__', o_voxel))"
