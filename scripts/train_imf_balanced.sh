#!/usr/bin/env bash
# Train iMF với context đã cân bằng (balanced LMDB) + contrastive — tách khỏi run diagnostic.
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_balanced_${TS}.log"
mkdir -p logs checkpoints/imf_unet_balanced

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

SLAT_LMDB="${SLAT_LMDB:-data/slat_context_balanced.lmdb}"
if [ ! -f "${SLAT_LMDB}/data.mdb" ]; then
  echo "ERROR: ${SLAT_LMDB} not found. Run:"
  echo "  python scripts/rebalance_slat_context_lmdb.py"
  exit 1
fi

# batch 4 × accum 8 = sweet spot RTX 4090 (~7 batch/s, ~11 min/epoch; batch 8 không nhanh hơn)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=============================================="
echo "  iMF — BALANCED CONTEXT TRAIN (isolated)"
echo "=============================================="
echo "  slat LMDB: ${SLAT_LMDB}"
echo "  checkpoint: checkpoints/imf_unet_balanced/"
echo "  batch=${BATCH_SIZE} x accum=${GRAD_ACCUM}"
echo "  contrastive=0.2 mode=arcface, ctx weights 3/2/0.5, ratio_r_neq_t=0"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u src/train_imf.py \
    --offline-data \
    --slat-lmdb "${SLAT_LMDB}" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --checkpoint-dir checkpoints/imf_unet_balanced \
    --disable-cfg-conditioning \
    --disable-id-filters \
    --contrastive-loss-weight 0.2 \
    --contrastive-mode arcface \
    --ratio-r-neq-t 0 \
    --manifest data/mesh_manifest.json \
    > "${LOG_FILE}" 2>&1 &

echo "PID=$!"
echo "tail -f ${LOG_FILE}"
