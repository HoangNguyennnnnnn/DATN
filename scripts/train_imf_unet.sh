#!/usr/bin/env bash
# Stage 2 — 3D UNet backbone (29/05/2026). Thay VoxelMamba (không generate được).
# Standard flow-matching v-pred (ratio=0 boundary) + uniform t. FiLM conditioning (946 hybrid).
# Validate overfit: UNet sinh identity-specific (cos>0.9, 1-step OK); Mamba cos 0.03.
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_unet_${TS}.log"
CKPT_DIR="${CKPT_DIR:-checkpoints/imf_unet}"
mkdir -p "${CKPT_DIR}" logs

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_faceverse_balanced.lmdb}"
[ -f "${SLAT_LMDB}/data.mdb" ] || { echo "ERROR: ${SLAT_LMDB} missing"; exit 1; }
# Stats normalize phải KHỚP với LMDB: combined→slat_stats_both.pt, faceverse→slat_stats_faceverse.pt
SLAT_STATS="${SLAT_STATS:-data/slat_stats_both.pt}"
[ -f "${SLAT_STATS}" ] || { echo "ERROR: ${SLAT_STATS} missing"; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
UNET_BASE="${UNET_BASE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
LR="${LR:-2e-4}"

RESUME_ARGS=()
if [ "${FRESH_START:-1}" = "1" ]; then
  echo "  Fresh start (FRESH_START=1)"
elif [ -n "${RESUME}" ] && [ -f "${RESUME}" ]; then
  echo "  Resume: ${RESUME}"
  RESUME_ARGS=(--resume "${RESUME}")
  [ "${RESUME_MODEL_ONLY:-0}" = "1" ] && RESUME_ARGS+=(--resume-model-only)
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  iMF Stage 2 — 3D UNet backbone"
echo "=============================================="
echo "  ckpt: ${CKPT_DIR}"
echo "  batch=${BATCH_SIZE} accum=${GRAD_ACCUM} effective=$((BATCH_SIZE * GRAD_ACCUM))  unet_base=${UNET_BASE}"
echo "  epochs=${NUM_EPOCHS}  lr=${LR}  FM v-pred (ratio=0) + uniform t  CFG=OFF"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --slat-stats "${SLAT_STATS}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt \
    --manifest data/mesh_manifest.json \
    "${RESUME_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --backbone unet3d \
    --unet-base "${UNET_BASE}" \
    --facescape-all-expressions \
    --t-sampler uniform \
    --ratio-r-neq-t 0.0 \
    --context-use-all \
    --cfg-context-dropout 0.0 \
    --contrastive-loss-weight 0.0 \
    --context-velocity-sep-weight 0.0 \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor 4 \
    --lr "${LR}" \
    --epochs "${NUM_EPOCHS}" \
    --dataset both \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${CKPT_DIR}/train.pid"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}"
