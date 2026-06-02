#!/usr/bin/env bash
# iMF v4 pipeline tự động: Phase A (boundary-only warmup) → Phase B (JVP mean-flow).
# JVP-from-scratch nổ loss (du/dt khổng lồ khi u random) → warmup boundary cho velocity field
# ổn định trước, rồi bật JVP. Data v4 ĐÚNG (dense). FaceVerse + whiten + variance-weight + CFG.
set -eo pipefail
cd "$(dirname "$0")/.."
source miniconda3/etc/profile.d/conda.sh
conda activate facediff
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR="checkpoints/imf_v4"
LMDB="data/slat_context_v4.lmdb"
STATS="data/slat_stats_v4.pt"
WHITEN="data/context_whiten_v4.pt"
VAR="data/voxel_variance_v4.pt"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-400}"   # boundary warmup
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-3000}"  # JVP mean-flow (tổng)
BATCH="${BATCH:-32}"; ACCUM="${ACCUM:-8}"; LR="${LR:-1e-4}"
mkdir -p "${CKPT_DIR}" logs
TS="$(date +%Y%m%d_%H%M%S)"

COMMON=(
  --offline-data --dataset faceverse
  --slat-lmdb "${LMDB}" --slat-stats "${STATS}"
  --context-lmdb data/hybrid_context.lmdb
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt
  --manifest data/mesh_manifest.json
  --checkpoint-dir "${CKPT_DIR}"
  --backbone unet3d --unet-base 128
  --facescape-all-expressions
  --context-use-all --context-whiten "${WHITEN}"
  --cfg-context-dropout 0.1
  --contrastive-loss-weight 0.0 --context-velocity-sep-weight 0.0
  --batch-size "${BATCH}" --gradient-accumulation-steps "${ACCUM}"
  --num-workers 4 --prefetch-factor 4 --lr "${LR}"
)
export VOXEL_VARIANCE_PATH="${VAR}" VOXEL_VARIANCE_MULT=4.0

# ---------- PHASE A: boundary-only (ratio=0, NO JVP, NO v-head) ----------
LOG_A="logs/imf_v4_phaseA_${TS}.log"
echo "=== PHASE A: boundary warmup ${PHASE_A_EPOCHS} ep (ratio=0, no JVP) → ${LOG_A} ==="
python -u src/train_imf.py "${COMMON[@]}" \
    --t-sampler logit_normal --ratio-r-neq-t 0.0 \
    --epochs "${PHASE_A_EPOCHS}" \
    > "${LOG_A}" 2>&1
echo "=== PHASE A done. latest_step.pt ready. ==="

# ---------- PHASE B: JVP mean-flow (ratio=0.5 + v-head + CFG), resume ----------
# BẮT BUỘC adaptive weighting "paper" (loss/(loss+eps)^p) để JVP KHÔNG nổ.
# Đây là cơ chế chống blowup của paper iMF — thiếu nó (như lần trước) → loss diverge 1e9.
LOG_B="logs/imf_v4_phaseB_${TS}.log"
echo "=== PHASE B: JVP mean-flow + adaptive=paper → ${PHASE_B_EPOCHS} ep (resume) → ${LOG_B} ==="
IMEFLOW_ADAPTIVE=paper IMEFLOW_ADAPTIVE_ON=1 \
python -u src/train_imf.py "${COMMON[@]}" \
    --resume "${CKPT_DIR}/latest_step.pt" \
    --t-sampler logit_normal --ratio-r-neq-t 0.5 \
    --enable-cfg-conditioning --cfg-omega-max 8 --v-loss-weight 1.0 \
    --epochs "${PHASE_B_EPOCHS}" \
    > "${LOG_B}" 2>&1
echo "=== PHASE B done. ==="
