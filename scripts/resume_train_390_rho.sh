#!/bin/bash
# ==============================================================================
# SC-VAE Resume Training — Epoch 390 → 500
# Focus: Strengthen rho_head for better inference-mode topology
# ==============================================================================
set -e

PYTHON="/mnt/18TData/facediff/miniconda3/envs/facediff/bin/python"
WORKDIR="/mnt/18TData/facediff"
cd "$WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

echo "============================================"
echo " SC-VAE Resume: Epoch 390 → 500"
echo " rho_loss_weight: 0.5 (up from 0.2)"
echo " LR: 2e-5, Warmup: 1000 steps"
echo "============================================"

$PYTHON -u src/train_sc_vae.py \
    --resume checkpoints/sc_vae_shape/epoch_390.pt \
    --resume-model-only \
    \
    --feature-mode shape_mat \
    --in-channels 10 \
    --dataset both \
    \
    --lmdb-dir data/ovoxel_cache_lmdb \
    --lmdb-readahead \
    --lmdb-only \
    \
    --batch-size 4 \
    --max-voxels 350000 \
    --max-points-per-batch 10000000 \
    --epochs 500 \
    --lr 2e-5 \
    --rho-loss-weight 0.5 \
    --rho-warmup-epochs 5 \
    \
    --num-workers 16 \
    --prefetch-factor 8 \
    --dataloader-timeout 300 \
    \
    --gradient-accumulation-steps 33 \
    --use-activation-checkpointing \
    \
    --save-every-steps 500 \
    --val-every-epochs 5 \
    --val-split 0.05 \
    \
    --perf-log-every-steps 5 \
    --no-wandb \
    --checkpoint-dir checkpoints/sc_vae_shape
