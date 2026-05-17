#!/usr/bin/env bash
# train_imf.sh — iMF VoxelMamba training (offline .pt cache or LMDB)
set -eo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_imf_${TS}.log"

mkdir -p logs checkpoints/imf_unet

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # Parallel LMDB I/O — GPU không phải đợi (SlatDataset lazy re-open trong worker)
RESUME="${RESUME:-checkpoints/imf_unet/best.pt}"
SC_VAE_CKPT="${SC_VAE_CKPT:-checkpoints/sc_vae_shape/epoch_500.pt}"

# Use LMDB only when pack finished (avoid partial/corrupt DB during pack_slat_lmdb.py)
SLAT_LMDB_ARGS=()
DATA_MODE=".pt cache (slat_cache + slat_cache_facescape)"
if [ -f "data/slat_context.lmdb/data.mdb" ] && [ -s "data/slat_context.lmdb/data.mdb" ]; then
  PACKED=$(python3 -c "
import lmdb, json
try:
    e=lmdb.open('data/slat_context.lmdb',readonly=True,lock=False)
    with e.begin() as t:
        m=t.get(b'__meta__')
    e.close()
    print(json.loads(m).get('packed',0) if m else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  if [ "${PACKED}" -ge 20000 ] 2>/dev/null; then
    SLAT_LMDB_ARGS=(--slat-lmdb data/slat_context.lmdb)
    DATA_MODE="LMDB (${PACKED} samples)"
  fi
fi

EFFECTIVE=$((BATCH_SIZE * GRAD_ACCUM))

echo "=============================================="
echo "  FACEDIFF — Phase 2: iMF VoxelMamba Training"
echo "=============================================="
echo "  batch=${BATCH_SIZE} × grad_accum=${GRAD_ACCUM} = effective ${EFFECTIVE}"
echo "  CFG: ENABLED | Resume: ${RESUME}"
echo "  Data: ${DATA_MODE}"
echo "  SC-VAE (cache tag): ${SC_VAE_CKPT}"
echo "  Log: ${LOG_FILE}"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME_ARGS=()
if [ -n "${RESUME}" ] && [ -f "${RESUME}" ]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

# Default: backgrounded via nohup so the process survives shell close / SIGHUP.
# Set FOREGROUND=1 to run in foreground (e.g. inside a tmux pane).
if [ "${FOREGROUND:-0}" = "1" ]; then
  python -u src/train_imf.py \
      --offline-data \
      "${SLAT_LMDB_ARGS[@]}" \
      --context-lmdb data/hybrid_context.lmdb \
      --sc-vae-ckpt "${SC_VAE_CKPT}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --lr "${LR}" \
      --epochs "${EPOCHS}" \
      --num-workers "${NUM_WORKERS}" \
      --enable-cfg-conditioning \
      --disable-id-filters \
      --manifest data/mesh_manifest.json \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "${LOG_FILE}"
else
  nohup python -u src/train_imf.py \
      --offline-data \
      "${SLAT_LMDB_ARGS[@]}" \
      --context-lmdb data/hybrid_context.lmdb \
      --sc-vae-ckpt "${SC_VAE_CKPT}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --lr "${LR}" \
      --epochs "${EPOCHS}" \
      --num-workers "${NUM_WORKERS}" \
      --enable-cfg-conditioning \
      --disable-id-filters \
      --manifest data/mesh_manifest.json \
      "${RESUME_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1 &

  TRAIN_PID=$!
  echo "Training PID=${TRAIN_PID}"
  echo "Log:     ${LOG_FILE}"
  echo "Monitor: tail -f ${LOG_FILE}"
  echo "Stop:    kill ${TRAIN_PID}"
fi
