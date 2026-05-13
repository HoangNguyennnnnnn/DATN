#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# resume_from_397.sh
#
# Resume FaceDiff SC-VAE training from the epoch-397 checkpoint with a fresh
# cosine warm-restart schedule, so we can keep refining without re-running the
# original 0-500 epoch budget. The model state is loaded strictly from the
# checkpoint; only the LR scheduler is rebuilt with `--resume-scheduler-mode
# cosine_restart`.
#
# Usage:
#   bash scripts/resume_from_397.sh [extra args forwarded to train_sc_vae.py]
#
# Tunables (override via env):
#   CKPT             default: checkpoints/sc_vae_shape/interrupt.pt (epoch 397, 0.0365)
#   EXTEND_EPOCHS    default: 100
#   TARGET_MIN_LR    default: 1e-6
#   LOG_FILE         default: logs/resume_from_397_<timestamp>.log
#   ENABLE_STAGE2_RENDER_LOSS  default: 0 (omit --enable-stage2-render-loss; ~24GB GPUs OOM if set to 1)
#
# Notes:
# - The codebase has been audited against TRELLIS.2 (microsoft/TRELLIS.2);
#   the SC-VAE now applies the canonical (1+2m)·sigmoid(h)−m dual-vertex
#   activation, a non-affine LayerNorm before to_mu/to_logvar, and divides
#   the KL by mu.numel() to match the report's normalisation. None of these
#   changes alter parameter shapes, so the epoch-397 checkpoint loads cleanly.
# - The render-loss now uses the activated dv for projection, matching the
#   inference path of the dual contouring extractor.
# - We keep gradient_accumulation_steps=33 and batch_size=4 to preserve the
#   resume_contract digest stored in the checkpoint.
# ----------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:-checkpoints/sc_vae_shape/interrupt.pt}"
EXTEND_EPOCHS="${EXTEND_EPOCHS:-100}"
TARGET_MIN_LR="${TARGET_MIN_LR:-1e-6}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/resume_from_397_${TS}.log}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[resume_from_397] Checkpoint not found: ${CKPT}" >&2
    echo "  Hint: pass CKPT=checkpoints/sc_vae_shape/<epoch_xxx.pt> if you prefer a clean epoch boundary." >&2
    exit 1
fi

mkdir -p logs

# Activate the project conda env if not already active.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "facediff" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate facediff
fi

echo "[resume_from_397] CKPT=${CKPT}"
echo "[resume_from_397] EXTEND_EPOCHS=${EXTEND_EPOCHS}  TARGET_MIN_LR=${TARGET_MIN_LR}"
echo "[resume_from_397] LOG_FILE=${LOG_FILE}"

STAGE2_ARGS=()
if [[ "${ENABLE_STAGE2_RENDER_LOSS:-0}" == "1" ]]; then
    STAGE2_ARGS+=(--enable-stage2-render-loss)
    echo "[resume_from_397] ENABLE_STAGE2_RENDER_LOSS=1 (LPIPS/render; needs >24GB VRAM)"
else
    STAGE2_ARGS+=(--disable-stage2-render-loss)
    echo "[resume_from_397] stage-2 render loss OFF (set ENABLE_STAGE2_RENDER_LOSS=1 to enable)"
fi

# NOTE: resume_contract sha1 will mismatch (model-only resume is fine).
# Batch/accum tunable via env for GPU memory constraints (RTX 4090 shared).
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-66}"
echo "[resume_from_397] BATCH_SIZE=${BATCH_SIZE}  GRAD_ACCUM=${GRAD_ACCUM}  (effective=$(( BATCH_SIZE * GRAD_ACCUM )))"

exec python -u src/train_sc_vae.py \
    --dataset both \
    --feature-mode shape_mat \
    --in-channels 10 \
    --lmdb-only \
    --checkpoint-dir checkpoints/sc_vae_shape \
    --resume "${CKPT}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --batch-size "${BATCH_SIZE}" \
    --resume-scheduler-mode cosine_restart \
    --resume-extend-epochs "${EXTEND_EPOCHS}" \
    --resume-target-min-lr "${TARGET_MIN_LR}" \
    "${STAGE2_ARGS[@]}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
