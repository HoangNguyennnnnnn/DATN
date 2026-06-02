#!/usr/bin/env bash
# Stage 2 — 3D UNet (cross-attn conditioning) + flow-matching v-pred.
# Hướng (30/05): CFG guidance thay ctx_sep. Train MSE + CFG dropout (học null context),
# guided sampling lúc generate. ctx_sep mặc định TẮT (phá generation — đã chứng minh overfit).
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_unet_${TS}.log"
CKPT_DIR="${CKPT_DIR:-checkpoints/imf_unet}"
mkdir -p "${CKPT_DIR}" logs

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_both_balanced.lmdb}"
SLAT_STATS="${SLAT_STATS:-data/slat_stats_both.pt}"
[ -f "${SLAT_LMDB}/data.mdb" ] || { echo "ERROR: ${SLAT_LMDB} missing"; exit 1; }
[ -f "${SLAT_STATS}" ] || { echo "ERROR: ${SLAT_STATS} missing"; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
UNET_BASE="${UNET_BASE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
LR="${LR:-2e-4}"
CTX_SEP="${CTX_SEP:-0.0}"          # context-forcing loss (0 = tắt, dùng CFG thay thế)
CFG_DROPOUT="${CFG_DROPOUT:-0.1}"  # null-context dropout cho classifier-free guidance
DATASET="${DATASET:-both}"
CONTEXT_WHITEN="${CONTEXT_WHITEN:-}"            # Bước 2: PCA-whiten context (path .pt hoặc rỗng)
export VOXEL_VARIANCE_PATH="${VOXEL_VARIANCE_PATH:-}"   # Bước 3: variance-weighted loss
export VOXEL_VARIANCE_MULT="${VOXEL_VARIANCE_MULT:-4.0}"
export PREDICTION_TYPE="${PREDICTION_TYPE:-velocity}"   # velocity | x0 (fix noise lấn át identity)

# Idempotent: kill training cũ trên cùng CKPT_DIR trước khi chạy mới (tránh chồng tiến trình,
# tránh 2 run ghi đè cùng checkpoint). Bỏ qua bằng SKIP_KILL_OLD=1.
if [ "${SKIP_KILL_OLD:-0}" != "1" ] && [ -f "${CKPT_DIR}/train.pid" ]; then
  OLD_PID="$(cat "${CKPT_DIR}/train.pid" 2>/dev/null)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "  Killing old training PID=${OLD_PID} (cùng CKPT_DIR)"
    pkill -9 -P "${OLD_PID}" 2>/dev/null || true
    kill -9 "${OLD_PID}" 2>/dev/null || true
    sleep 4
  fi
fi

RESUME_ARGS=()
if [ "${FRESH_START:-1}" = "1" ]; then
  echo "  Fresh start (FRESH_START=1)"
elif [ -n "${RESUME}" ] && [ -f "${RESUME}" ]; then
  echo "  Resume: ${RESUME}"
  RESUME_ARGS=(--resume "${RESUME}")
  [ "${RESUME_MODEL_ONLY:-0}" = "1" ] && RESUME_ARGS+=(--resume-model-only)
fi

WHITEN_ARGS=()
[ -n "${CONTEXT_WHITEN}" ] && WHITEN_ARGS=(--context-whiten "${CONTEXT_WHITEN}")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  iMF Stage 2 — 3D UNet (cross-attn) + CFG"
echo "=============================================="
echo "  ckpt=${CKPT_DIR}  lmdb=${SLAT_LMDB}"
echo "  batch=${BATCH_SIZE} accum=${GRAD_ACCUM} base=${UNET_BASE}  epochs=${NUM_EPOCHS} lr=${LR}"
echo "  ctx_sep=${CTX_SEP}  cfg_dropout=${CFG_DROPOUT}  dataset=${DATASET}"
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
    --cfg-context-dropout "${CFG_DROPOUT}" \
    --contrastive-loss-weight 0.0 \
    --context-velocity-sep-weight "${CTX_SEP}" \
    --context-velocity-sep-margin 0.2 \
    --contrastive-mode arcface \
    "${WHITEN_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor 4 \
    --lr "${LR}" \
    --epochs "${NUM_EPOCHS}" \
    --dataset "${DATASET}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${CKPT_DIR}/train.pid"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}"
