#!/usr/bin/env bash
# iMF v8 — single-phase training aligned with official imeanflow (arXiv:2512.02012).
# https://github.com/Lyy-iiis/imeanflow
#
# Paper defaults: CFG on from step 0, class/context dropout 0.1, ratio_r_neq_t=0.5,
# logit-normal t, v-head aux loss, omega in [1,7], interval conditioning.
# No Phase A/B split.
#
# VRAM (4090 24GB): batch=2 × accum=16 (effective 32). OOM → BATCH_SIZE=1 GRAD_ACCUM=32
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_v8_${TS}.log"
CKPT_DIR="${CKPT_DIR:-checkpoints/imf_v8_lite}"
mkdir -p "${CKPT_DIR}" logs
RESUME="${RESUME:-${CKPT_DIR}/epoch_100.pt}"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
[ -f "${SLAT_LMDB}/data.mdb" ] || { echo "ERROR: ${SLAT_LMDB} missing"; exit 1; }

# 4090 24GB: batch=4 × accum=8 (effective 32), FFN expand=4 (full MLP)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAMBA_FFN_EXPAND="${MAMBA_FFN_EXPAND:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"

RESUME_ARGS=()
if [ "${FRESH_START:-1}" = "1" ]; then
  echo "  Fresh start (FRESH_START=1)"
elif [ -n "${RESUME}" ] && [ -f "${RESUME}" ]; then
  echo "  Resume: ${RESUME}"
  RESUME_ARGS=(--resume "${RESUME}")
  if [ "${RESUME_MODEL_ONLY:-0}" = "1" ]; then
    RESUME_ARGS+=(--resume-model-only)
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CROSS_ATTN_PROJ_LR_MULT="${CROSS_ATTN_PROJ_LR_MULT:-1.0}"
export CROSS_ATTN_PROJ_MAX_NORM="${CROSS_ATTN_PROJ_MAX_NORM:-1.0}"
export IMEFLOW_PAPER_STRICT="${IMEFLOW_PAPER_STRICT:-1}"
export IMEFLOW_ADAPTIVE="${IMEFLOW_ADAPTIVE:-paper}"

echo "=============================================="
echo "  iMF v8 — single phase (imeanflow-style)"
echo "=============================================="
echo "  ckpt: ${CKPT_DIR}"
echo "  batch=${BATCH_SIZE} accum=${GRAD_ACCUM} effective=$((BATCH_SIZE * GRAD_ACCUM))  ffn_expand=${MAMBA_FFN_EXPAND}"
echo "  epochs=${NUM_EPOCHS}  CFG=ON  ctx_dropout=0.1  ctx_sep=ON (ADALN Mode)"
echo "  imeanflow: strict_tr=${IMEFLOW_PAPER_STRICT}  adaptive=${IMEFLOW_ADAPTIVE}"
echo "  cross_attn proj: LR×${CROSS_ATTN_PROJ_LR_MULT} max_norm=${CROSS_ATTN_PROJ_MAX_NORM}"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

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
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --mamba-num-layers 8 \
    --mamba-ffn-expand "${MAMBA_FFN_EXPAND}" \
    --lr 5e-5 \
    --epochs "${NUM_EPOCHS}" \
    --facescape-unique-identities \
    --enable-cfg-conditioning \
    --cfg-omega-max 7 \
    --cfg-context-dropout 0.1 \
    --context-cond-mode adaln \
    --context-use-all \
    --context-segment-weights 1.5 1.0 0.5 \
    --v-loss-weight 1.0 \
    --contrastive-loss-weight 0.2 \
    --contrastive-mode arcface \
    --context-velocity-sep-weight 0.1 \
    --context-velocity-sep-margin 0.2 \
    --ratio-r-neq-t 0.5 \
    --dataset faceverse \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${CKPT_DIR}/train.pid"
echo "PID=${TRAIN_PID}  log=${LOG_FILE}"
