#!/usr/bin/env bash
# Stage 2 v7 Phase B+C: JVP + v-head + contrastive enabled.
# Contrastive InfoNCE forces context dependency (Mamba's adaLN_ctx was suppressing context).
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v7_phaseB_${TS}.log"
CKPT_DIR="checkpoints/imf_v7_phaseB"
RESUME_CKPT="checkpoints/imf_v7_phaseB/latest_step.pt"
mkdir -p logs "${CKPT_DIR}"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
if [ ! -f "${SLAT_LMDB}/data.mdb" ]; then
  echo "ERROR: ${SLAT_LMDB} missing"
  exit 1
fi

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: ${RESUME_CKPT} missing — Phase A epoch 20 checkpoint required"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

echo "=============================================="
echo "  iMF v7 PHASE B (JVP + v-head)"
echo "=============================================="
echo "  Resume: ${RESUME_CKPT}"
echo "  LMDB: ${SLAT_LMDB}"
echo "  ckpt: ${CKPT_DIR}"
echo "  arch: JVP (r≠t 50%), v-head depth=8, boundary still 50%"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --resume "${RESUME_CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --lr 1e-4 \
    --epochs 400 \
    --num-workers 4 \
    --checkpoint-dir "${CKPT_DIR}" \
    --disable-cfg-conditioning \
    --disable-id-filters \
    --contrastive-loss-weight 0.2 \
    --contrastive-mode arcface \
    --context-velocity-sep-weight 0.0 \
    --context-velocity-sep-margin 0.0 \
    --ratio-r-neq-t 0.5 \
    --manifest data/mesh_manifest.json \
    > "${LOG_FILE}" 2>&1 &

echo "PID=$!"
echo "tail -f ${LOG_FILE}"
