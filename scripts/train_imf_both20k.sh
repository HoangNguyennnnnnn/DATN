#!/usr/bin/env bash
# Train-only (data 20K đã sẵn): resume imf_v4/latest_step (ep5084) → ep8000.
# Tách khỏi orchestrator để KHÔNG chạy lại precompute/pack/stats/variance (đã xong).
# VRAM-safe: batch=16 accum=16 (~11GB) chạy chung biometrics 10GB. Occupancy ON (IoU).
set -eo pipefail
cd "$(dirname "$0")/.."
source miniconda3/etc/profile.d/conda.sh
conda activate facediff
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LMDB=data/slat_context_both20k.lmdb
STATS=data/slat_stats_both20k.pt
VAR=data/voxel_variance_both20k.pt
WHITEN=data/context_whiten_v4.pt
CKPT_DIR=checkpoints/imf_both20k
# Ưu tiên resume từ tiến độ mới nhất của chính run này; chưa có thì từ imf_v4 ep5084.
RESUME=checkpoints/imf_v4/latest_step.pt
[ -f "$CKPT_DIR/latest_step.pt" ] && RESUME="$CKPT_DIR/latest_step.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG=logs/imf_both20k_train_${TS}.log
mkdir -p "$CKPT_DIR" logs

export VOXEL_VARIANCE_PATH="$VAR" VOXEL_VARIANCE_MULT=4.0
export OCCUPANCY_LOSS_WEIGHT="${OCCUPANCY_LOSS_WEIGHT:-1.0}"
export PREDICTION_TYPE="${PREDICTION_TYPE:-velocity}"

echo "[$(date +%H:%M:%S)] Train iMF both20k: resume $RESUME → ep${TARGET:-8000}, occ=$OCCUPANCY_LOSS_WEIGHT, batch=${BATCH:-16}x${ACCUM:-16} → $LOG"

IMEFLOW_ADAPTIVE=paper IMEFLOW_ADAPTIVE_ON=1 \
nohup python -u src/train_imf.py \
    --offline-data --dataset both \
    --slat-lmdb "$LMDB" --slat-stats "$STATS" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_both/latest_step.pt \
    --manifest data/mesh_manifest.json \
    --resume "$RESUME" \
    --checkpoint-dir "$CKPT_DIR" \
    --backbone unet3d --unet-base 128 \
    --facescape-all-expressions \
    --context-use-all --context-whiten "$WHITEN" \
    --cfg-context-dropout 0.1 \
    --contrastive-loss-weight 0.0 --context-velocity-sep-weight 0.0 \
    --t-sampler logit_normal --ratio-r-neq-t 0.5 \
    --enable-cfg-conditioning --cfg-omega-max 8 --v-loss-weight 1.0 \
    --batch-size "${BATCH:-16}" --gradient-accumulation-steps "${ACCUM:-16}" \
    --num-workers 4 --prefetch-factor 4 --lr 1e-4 \
    --epochs "${TARGET:-8000}" \
    > "$LOG" 2>&1 &

echo "$!" > "$CKPT_DIR/train.pid"
echo "PID=$! LOG=$LOG"
