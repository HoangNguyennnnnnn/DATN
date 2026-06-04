#!/bin/bash
set -e

# SC-VAE Both (FaceVerse + FaceScape) — Optimized for RTX 4090
# Tối ưu tốc độ (batch_size=4, giữ nguyên để tránh OOM):
#   - gradient_accumulation=16 (giảm từ 32): update weights gấp đôi → hội tụ nhanh hơn
#   - effective batch = 4 × 16 = 64 (vẫn đủ lớn cho SC-VAE)
#   - PYTHONUNBUFFERED=1: log real-time
#   - perf_log_every_steps=20: monitor throughput
#   - save_every_steps=500: checkpoint thường xuyên hơn
#   - PYTORCH_CUDA_ALLOC_CONF: giảm fragmentation

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

miniconda3/envs/facediff/bin/python src/train_sc_vae.py \
    --dataset both \
    --feature-mode shape_mat \
    --in-channels 10 \
    --lmdb-dir data/ovoxel_cache_lmdb \
    --lmdb-only \
    --resume checkpoints/sc_vae_shape/epoch_600.pt \
    --allow-unsafe-resume \
    --no-torch-compile \
    --enable-ema --ema-decay 0.9999 \
    --epochs 800 \
    --batch-size 4 \
    --gradient-accumulation-steps 16 \
    --lr 1e-5 \
    --checkpoint-dir checkpoints/sc_vae_both \
    --perf-log-every-steps 20 \
    --save-every-steps 500
