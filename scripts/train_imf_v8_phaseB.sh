#!/usr/bin/env bash
# iMF v8 Phase B — Fine-tuning & Identity Locking script.
#
# Bật InfoNCE Contrastive Loss (0.2) và Context Separation Loss (0.1)
# giúp khóa chặt nhận dạng chân dung ArcFace vào cấu trúc Voxel 3D.
#
# Hướng dẫn chạy:
#   bash scripts/train_imf_v8_phaseB.sh
#
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v8_phaseB_${TS}.log"

# Cấu hình đường dẫn
PHASE_A_DIR="checkpoints/imf_v8_lite"
PHASE_B_DIR="checkpoints/imf_v8_lite_phaseB"
mkdir -p "${PHASE_B_DIR}" logs

# Tìm checkpoint Phase A tốt nhất làm điểm xuất phát
if [ -f "${PHASE_A_DIR}/best.pt" ]; then
  RESUME_CKPT="${PHASE_A_DIR}/best.pt"
elif [ -f "${PHASE_A_DIR}/latest_step.pt" ]; then
  RESUME_CKPT="${PHASE_A_DIR}/latest_step.pt"
else
  echo "ERROR: Không tìm thấy checkpoint Phase A tại ${PHASE_A_DIR}/best.pt hay latest_step.pt"
  echo "Vui lòng đợi Phase A hoàn thành ít nhất vài Epochs trước khi chạy Phase B."
  exit 1
fi

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
[ -f "${SLAT_LMDB}/data.mdb" ] || { echo "ERROR: ${SLAT_LMDB} missing"; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CROSS_ATTN_PROJ_LR_MULT="${CROSS_ATTN_PROJ_LR_MULT:-1.0}"
export CROSS_ATTN_PROJ_MAX_NORM="${CROSS_ATTN_PROJ_MAX_NORM:-1.0}"
export IMEFLOW_PAPER_STRICT="${IMEFLOW_PAPER_STRICT:-1}"
export IMEFLOW_ADAPTIVE="${IMEFLOW_ADAPTIVE:-paper}"

echo "=============================================="
echo "  iMF v8 PHASE B — IDENTITY LOCKING ACTIVATION"
echo "=============================================="
echo "  Resume from Phase A: ${RESUME_CKPT}"
echo "  Output ckpt dir:     ${PHASE_B_DIR}"
echo "  Batch/Accum/workers: batch=${BATCH_SIZE} accum=${GRAD_ACCUM} workers=${NUM_WORKERS}"
echo "  ============================================"
echo "  💥 InfoNCE Contrastive Weight:  0.2 (ENABLED)"
echo "  💥 Context Separation Weight:   0.1 (ENABLED)"
echo "  ============================================"
echo "  Log File: ${LOG_FILE}"
echo "=============================================="

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --manifest data/mesh_manifest.json \
    --resume "${RESUME_CKPT}" \
    --resume-model-only \
    --checkpoint-dir "${PHASE_B_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --mamba-num-layers 8 \
    --lr 5e-5 \
    --epochs "${NUM_EPOCHS}" \
    --disable-id-filters \
    --enable-cfg-conditioning \
    --cfg-omega-max 7 \
    --cfg-context-dropout 0.1 \
    --v-loss-weight 1.0 \
    --contrastive-loss-weight 0.2 \
    --contrastive-mode arcface \
    --context-velocity-sep-weight 0.1 \
    --context-velocity-sep-margin 0.2 \
    --ratio-r-neq-t 0.5 \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PHASE_B_DIR}/train.pid"
echo "Phase B has started in background!"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}"
echo "Dùng lệnh sau để theo dõi tiến trình:"
echo "  tail -f ${LOG_FILE}"
