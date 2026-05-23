#!/usr/bin/env bash
# iMF v8 LITE: 8 layers, FFN expand=2, cross-attn, v-head=8, JVP 0.5 (~70M params, ~25-30 min/epoch)
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v8_lite_${TS}.log"
CKPT_DIR="checkpoints/imf_v8_lite"
mkdir -p "${CKPT_DIR}"
RESUME="${RESUME:-${CKPT_DIR}/latest_step.pt}"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
[ -f "${SLAT_LMDB}/data.mdb" ] || { echo "ERROR: ${SLAT_LMDB} missing"; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-11}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-40}"

RESUME_ARGS=()
if [ "${FRESH_START:-0}" = "1" ]; then
  echo "  Fresh start (FRESH_START=1, no resume)"
elif [ -n "${RESUME}" ] && [ -f "${RESUME}" ]; then
  echo "  Resume: ${RESUME}"
  RESUME_ARGS=(--resume "${RESUME}")
fi

echo "=============================================="
echo "  iMF v8 LITE — 8L × FFN2 × cross-attn + JVP"
echo "=============================================="
echo "  ckpt: ${CKPT_DIR}"
echo "  batch=${BATCH_SIZE} accum=${GRAD_ACCUM} workers=${NUM_WORKERS}"
echo "  layers=8 ffn_expand=2 ratio-r-neq-t=0.5 ctx_dropout=0.1"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --manifest data/mesh_manifest.json \
    "${RESUME_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --mamba-num-layers 8 \
    --mamba-ffn-expand 2 \
    --lr 1e-4 \
    --epochs "${PHASE_A_EPOCHS}" \
    --disable-id-filters \
    --disable-cfg-conditioning \
    --cfg-context-dropout 0.1 \
    --contrastive-loss-weight 0.0 \
    --context-velocity-sep-weight 0.0 \
    --ratio-r-neq-t 0.5 \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${CKPT_DIR}/train.pid"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}  pidfile=${CKPT_DIR}/train.pid"
