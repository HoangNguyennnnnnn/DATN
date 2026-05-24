#!/usr/bin/env bash
# Quick eval after Phase A or Phase B (identity + slat recon).
# Usage:
#   bash scripts/eval_imf_v8_checkpoint.sh checkpoints/imf_v8_lite/best.pt
#   PHASE=post_b bash scripts/eval_imf_v8_checkpoint.sh checkpoints/imf_v8_lite/best.pt
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-checkpoints/imf_v8_lite/best.pt}"
PHASE="${PHASE:-post_a}"
LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/eval_imf_v8_${PHASE}_${TS}.log"

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

if [ ! -f "${CKPT}" ]; then
  CKPT="checkpoints/imf_v8_lite/latest_step.pt"
fi
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint missing"; exit 1; }

mkdir -p logs
{
  echo "=== eval ${PHASE} $(date -Is) ckpt=${CKPT} ==="
  python scripts/test/test_imf_identity_t0.py \
    --checkpoint "${CKPT}" \
    --lmdb "${LMDB}" \
    --num-samples 16

  if [ "${PHASE}" = "post_a" ]; then
    python scripts/test/test_imf_slat_recon.py \
      --checkpoint "${CKPT}" \
      --lmdb "${LMDB}" \
      --num-steps 1 2 5 10 20
  else
    python scripts/test/test_imf_slat_recon.py \
      --checkpoint "${CKPT}" \
      --lmdb "${LMDB}" \
      --num-steps 1 2 5 10 20 \
      --omega 7.5 --cfg-tmin 0.4 --cfg-tmax 0.65
  fi
} 2>&1 | tee "${LOG}"

echo "Eval log: ${LOG}"
