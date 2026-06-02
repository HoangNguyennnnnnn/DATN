#!/bin/bash
set -e

# Bước 1: Precompute lại cache .pt với CACHE_SCHEMA_VERSION = 4 (dense fix)
echo "=== 1. PRECOMPUTE DENSE SLAT CACHE ==="
PYTHONUNBUFFERED=1 miniconda3/envs/facediff/bin/python scripts/data/precompute_slat_cache.py \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt \
    --dataset faceverse \
    --num-workers 20 \
    --context-lmdb data/hybrid_context.lmdb \
    --ovoxel-lmdb data/ovoxel_cache_lmdb \
    --skip-existing

# Bước 2: Pack lại toàn bộ .pt file vào LMDB mới
echo "=== 2. PACKING TO LMDB ==="
PYTHONUNBUFFERED=1 miniconda3/envs/facediff/bin/python scripts/data/pack_slat_lmdb.py --output data/slat_context_v2.lmdb --fs-cache-dir none

# Bước 2.1: Tính toán lại stats
echo "=== 2.1. COMPUTING STATS ==="
PYTHONUNBUFFERED=1 miniconda3/envs/facediff/bin/python scripts/data/compute_slat_stats.py --lmdb data/slat_context_v2.lmdb --out data/slat_stats_v3.pt

# Bước 3: Retrain iMF U-Net từ đầu
echo "=== 3. TRAINING iMF U-NET ==="
export EMPTY_WEIGHT_FLOOR=0.1
PYTHONUNBUFFERED=1 miniconda3/envs/facediff/bin/python src/train_imf.py \
    --resume checkpoints/imf_unet_v3/best.pt \
    --offline-data \
    --dataset faceverse \
    --slat-lmdb data/slat_context_v2.lmdb \
    --slat-stats data/slat_stats_v3.pt \
    --backbone unet3d \
    --batch-size 64 \
    --lr 0.0002 \
    --checkpoint-dir checkpoints/imf_unet_v3 \
    --epochs 3000
