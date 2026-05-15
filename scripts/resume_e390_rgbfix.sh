#!/usr/bin/env bash
# resume_e390_rgbfix.sh — direct nohup-compatible version
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/sc_vae_shape/epoch_390.pt"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/resume_e390_rgbfix_${TS}.log"

if [[ ! -f "${CKPT}" ]]; then
    echo "[resume] Checkpoint not found: ${CKPT}" >&2
    exit 1
fi

mkdir -p logs

# Activate conda env
CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

echo "[resume] CKPT=${CKPT}"
echo "[resume] LOG_FILE=${LOG_FILE}"
echo "[resume] RGB render-loss fix applied"

# Use nohup + redirect instead of exec|tee for background compatibility
nohup python -u src/train_sc_vae.py \
    --resume "${CKPT}" \
    --lr 1e-5 \
    --no-torch-compile \
    --batch-size 4 \
    --gradient-accumulation-steps 33 \
    --save-every-steps 2000 \
    --val-every-epochs 10 \
    --perf-log-every-steps 50 \
    --num-workers 4 \
    --enable-ema \
    --ema-decay 0.9999 \
    --enable-stage2-render-loss \
    --epochs 700 \
    --allow-unsafe-resume \
    > "${LOG_FILE}" 2>&1 &

echo "Training PID=$!"
echo "Log: ${LOG_FILE}"
