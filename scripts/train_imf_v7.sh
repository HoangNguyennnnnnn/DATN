#!/usr/bin/env bash
# Stage 2 v7: prefix tokens + per-layer ctx + paper-aligned hyperparams.
# Phase A: boundary-only, no aux losses. Phase B: enable JVP + contrastive.
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v7_${TS}.log"
CKPT_DIR="checkpoints/imf_v7"
mkdir -p logs "${CKPT_DIR}"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
if [ ! -f "${SLAT_LMDB}/data.mdb" ]; then
  echo "ERROR: ${SLAT_LMDB} missing"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

echo "=============================================="
echo "  iMF v7 TRAIN (scratch)"
echo "=============================================="
echo "  LMDB: ${SLAT_LMDB}"
echo "  ckpt: ${CKPT_DIR}"
echo "  arch: prefix=24, per_layer_ctx, Phase A (boundary-only)"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --lr 1e-4 \
    --epochs 400 \
    --num-workers 4 \
    --checkpoint-dir "${CKPT_DIR}" \
    --disable-cfg-conditioning \
    --disable-id-filters \
    --contrastive-loss-weight 0.0 \
    --contrastive-mode arcface \
    --context-velocity-sep-weight 0.0 \
    --context-velocity-sep-margin 0.0 \
    --ratio-r-neq-t 0 \
    --manifest data/mesh_manifest.json \
    > "${LOG_FILE}" 2>&1 &

echo "PID=$!"
echo "tail -f ${LOG_FILE}"
