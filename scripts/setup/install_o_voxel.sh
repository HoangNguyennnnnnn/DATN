#!/usr/bin/env bash
# Cài o_voxel (Microsoft TRELLIS.2) — kéo theo CuMesh + FlexGEMM (compile CUDA).
#
# Yêu cầu: torch đã cài; nvcc (CUDA compiler); build tools.
#   apt-get update && apt-get install -y git build-essential cmake ninja-build
#
# Nhiều image chỉ có CUDA runtime (PyTorch chạy GPU) nhưng KHÔNG có nvcc — phải cài toolkit:
#   apt-get install -y cuda-nvcc-12-4   # đổi 12-4 cho khớp driver/CUDA bạn đang dùng
# hoặc đổi sang image nvidia/cuda:12.4.1-devel-ubuntu22.04
#
# Nếu vẫn fail: README — train LMDB-only có thể không cần o_voxel trên GPU server.
set -euo pipefail

find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  local p
  for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda-12/bin/nvcc; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  shopt -s nullglob
  local candidates=(/usr/local/cuda-*/bin/nvcc)
  shopt -u nullglob
  if [[ ${#candidates[@]} -gt 0 ]]; then
    echo "${candidates[-1]}"
    return 0
  fi
  return 1
}

NVCC="$(find_nvcc || true)"
if [[ -z "${NVCC}" ]]; then
  echo "ERROR: Không tìm thấy nvcc (CUDA compiler)." >&2
  echo "" >&2
  echo "PyTorch vẫn chạy GPU với driver, nhưng build CuMesh/FlexGEMM BẮT CẦN nvcc." >&2
  echo "Cách 1 — cài compiler (Ubuntu + repo NVIDIA CUDA, ví dụ 12.x):" >&2
  echo "  apt-get update && apt-get install -y cuda-nvcc-12-4" >&2
  echo "  (đổi 12-4 cho đúng phiên bản: ls /usr/local/ | grep cuda)" >&2
  echo "Cách 2 — dùng container image *-devel* (có sẵn /usr/local/cuda/bin/nvcc)." >&2
  echo "Cách 3 — chỉ train từ LMDB đã pack: không cần o_voxel (xem README)." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$NVCC")")")}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "Using nvcc: $(command -v nvcc)"
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version | head -3

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
