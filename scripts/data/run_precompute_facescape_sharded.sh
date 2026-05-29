#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/../.."

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p data/fs_slat_cache
mkdir -p logs/precompute_fs

NUM_SHARDS=8
echo "Launching $NUM_SHARDS parallel shards to precompute FaceScape cache..."

for i in $(seq 0 $((NUM_SHARDS-1))); do
    nohup python scripts/data/precompute_slat_cache.py \
        --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt \
        --dataset facescape \
        --fs-cache-dir data/fs_slat_cache \
        --context-lmdb data/hybrid_context.lmdb \
        --shard-id $i --num-shards $NUM_SHARDS \
        --num-workers 0 \
        --disable-id-filters \
        --skip-existing \
        --device cuda:0 > logs/precompute_fs/shard_${i}.log 2>&1 &
    sleep 2
done
echo "All shards launched. Check logs in logs/precompute_fs/shard_*.log"
wait
echo "All shards completed!"
