#!/usr/bin/env bash
# iMF v8 Phase B: bật CFG sau khi Phase A đã học context (~epoch 40).
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v8_phaseB_cfg_${TS}.log"
CKPT_DIR="${CKPT_DIR:-checkpoints/imf_v8_lite}"
RESUME_CKPT="${RESUME_CKPT:-${CKPT_DIR}/latest_step.pt}"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
if [ ! -f "${SLAT_LMDB}/data.mdb" ]; then
  echo "ERROR: ${SLAT_LMDB} missing"
  exit 1
fi
if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: ${RESUME_CKPT} missing — chạy Phase A trước"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-11}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=============================================="
echo "  iMF v8 Phase B — CFG ON (resume Phase A)"
echo "=============================================="
echo "  Resume: ${RESUME_CKPT}"
echo "  ckpt: ${CKPT_DIR}"
echo "  epochs=${PHASE_B_EPOCHS} (CFG on, ~${PHASE_B_EPOCHS}×29min nếu lite batch=3)"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --manifest data/mesh_manifest.json \
    --resume "${RESUME_CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --mamba-num-layers 8 \
    --mamba-ffn-expand 2 \
    --lr 1e-4 \
    --epochs "${PHASE_B_EPOCHS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --disable-id-filters \
    --enable-cfg-conditioning \
    --cfg-context-dropout 0.1 \
    --contrastive-loss-weight 0.0 \
    --contrastive-mode arcface \
    --context-velocity-sep-weight 0.1 \
    --context-velocity-sep-margin 0.0 \
    --ratio-r-neq-t 0.5 \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
mkdir -p "${CKPT_DIR}"
echo "${TRAIN_PID}" > "${CKPT_DIR}/train.pid"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}  pidfile=${CKPT_DIR}/train.pid"
