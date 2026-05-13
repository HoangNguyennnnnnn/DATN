#!/usr/bin/env bash
# Cài o_voxel từ Microsoft TRELLIS.2 (mesh → O-Voxel). Cần torch + trimesh đã có trong env.
# Phụ thuộc build: git, g++, CUDA toolkit (image devel thường đủ).
set -euo pipefail
pip install "o_voxel @ git+https://github.com/microsoft/TRELLIS.2.git#subdirectory=o-voxel"
python -c "import o_voxel; print('o_voxel OK:', getattr(o_voxel, '__file__', o_voxel))"
