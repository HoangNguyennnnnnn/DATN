#!/bin/bash
# ============================================================
# FaceDiff Pipeline: Steps 4 → 5 → 6 (auto-sequential)
# Anti-OOM: oom_score_adj, num_workers=4, auto-resume on crash
# ============================================================
set -eo pipefail
cd /mnt/18TData/facediff

# ---- OOM Protection ----
# Đặt oom_score_adj = -900 để kernel ưu tiên giữ process này sống
# (range: -1000 = never kill, +1000 = kill first)
echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

export PATH="/mnt/18TData/facediff/miniconda3/bin:$PATH"
set +u
eval "$(/mnt/18TData/facediff/miniconda3/bin/conda shell.bash hook)"
conda activate facediff
set -u

echo "=========================================="
echo "[AUTO] Resuming pipeline from Step 4"
echo "[AUTO] $(date)"
echo "[AUTO] OOM score adj: $(cat /proc/self/oom_score_adj 2>/dev/null || echo N/A)"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────
# STEP 4: Resume SC-VAE Training
# Tự động chọn checkpoint mới nhất (epoch_*.pt hoặc latest_step.pt)
# ─────────────────────────────────────────────
# Ưu tiên latest_step.pt nếu mới hơn epoch checkpoint
EPOCH_CKPT=$(ls -t checkpoints/sc_vae_shape/epoch_*.pt 2>/dev/null | head -1)
STEP_CKPT="checkpoints/sc_vae_shape/latest_step.pt"

if [ -f "$STEP_CKPT" ] && [ -f "$EPOCH_CKPT" ]; then
    # Dùng file nào mới hơn
    if [ "$STEP_CKPT" -nt "$EPOCH_CKPT" ]; then
        LATEST_CKPT="$STEP_CKPT"
    else
        LATEST_CKPT="$EPOCH_CKPT"
    fi
elif [ -f "$STEP_CKPT" ]; then
    LATEST_CKPT="$STEP_CKPT"
else
    LATEST_CKPT="$EPOCH_CKPT"
fi

echo "[STEP 4] Resuming SC-VAE training from $LATEST_CKPT..."
echo "  Target: 500 epochs"
echo "  LR warmup: 500 steps (~3.4 epochs)"
echo "  num_workers: 4 (reduced from 8 to prevent OOM)"
echo "  Starting at: $(date)"
echo ""

PYTHONUNBUFFERED=1 python src/train_sc_vae.py \
    --resume "$LATEST_CKPT" \
    --lr 1e-5 --no-torch-compile \
    --gradient-accumulation-steps 33 \
    --save-every-steps 2000 --val-every-epochs 10 \
    --perf-log-every-steps 50 --num-workers 4 \
    2>&1 | stdbuf -oL tee -a logs/train_sc_vae_gamma_fixed.log

echo ""
echo "=========================================="
echo "[AUTO] SC-VAE Training Completed!"
echo "[AUTO] $(date)"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────
# STEP 5: Precompute Slat Latent Cache for iMF
# ─────────────────────────────────────────────
BEST_CKPT=$(ls -t checkpoints/sc_vae_shape/epoch_*.pt 2>/dev/null | head -1)
echo "[STEP 5] Precomputing Slat latents for iMF Stage 2..."
echo "  SC-VAE checkpoint: $BEST_CKPT"
echo "  Starting at: $(date)"

PYTHONUNBUFFERED=1 python scripts/precompute_slat_cache.py \
    --sc-vae-ckpt "$BEST_CKPT" \
    --dataset both \
    --cache-dir data/slat_cache \
    2>&1 | stdbuf -oL tee logs/precompute_slat_cache.log

echo "[STEP 5] Slat cache completed."
echo ""

# ─────────────────────────────────────────────
# STEP 6: Train iMF (Stage 2) with cached Slats
# ─────────────────────────────────────────────
echo "[STEP 6] Starting iMF training (Stage 2)..."
echo "  Mode: --offline-data"
echo "  Epochs: 400, Batch: 48, LR: 2e-4"
echo "  Starting at: $(date)"

PYTHONUNBUFFERED=1 python src/train_imf.py \
    --offline-data \
    --batch-size 48 --lr 2e-4 --epochs 400 \
    2>&1 | stdbuf -oL tee logs/train_imf.log

echo ""
echo "=========================================="
echo "[AUTO] ALL STAGES COMPLETE!"
echo "[AUTO] $(date)"
echo "=========================================="
