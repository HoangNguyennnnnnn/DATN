#!/bin/bash
# =============================================================================
# FaceDiff Auto Pipeline: Fix Gamma → Re-pack LMDB → Resume SC-VAE Training
# =============================================================================
# Run in tmux: tmux new-session -d -s auto_pipeline 'bash scripts/auto_fix_and_resume.sh'
# =============================================================================

set -eo pipefail
cd /mnt/18TData/facediff

# Activate conda (set +u needed for conda activation scripts)
set +u
eval "$(conda shell.bash hook)"
conda activate facediff
set -u

LOG="logs/auto_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "[AUTO] FaceDiff Auto Pipeline Started"
echo "[AUTO] $(date)"
echo "=========================================="

# ─────────────────────────────────────────────
# STEP 1: Fix gamma in disk cache (22K .pt files)
# ─────────────────────────────────────────────
echo "[STEP 1] Fixing gamma in 22K disk cache files..."
echo "  Cache dir: data/ovoxel_cache_recached/ (275GB)"
echo "  Formula: gamma = (1 - var(dv_local)).clamp(0, 1)"
echo "  Also fixing aabb to [-0.5, 0.5]³"
echo ""
PYTHONUNBUFFERED=1 python scripts/fix_gamma_cache.py --cache-dir data/ovoxel_cache_recached
echo "[STEP 1] Gamma fix completed."

# Verify gamma fix
echo "[STEP 1.1] Verifying gamma fix..."
python -c "
import torch, glob, random
files = glob.glob('data/ovoxel_cache_recached/**/*.pt', recursive=True)
random.seed(42)
samples = random.sample(files, min(20, len(files)))
ok = 0
for f in samples:
    d = torch.load(f, map_location='cpu', weights_only=False)
    gamma = d['features'][:, 6].float()
    if gamma.min() < 0.999:
        ok += 1
print(f'Gamma verification: {ok}/{len(samples)} files have variable gamma')
if ok < len(samples) * 0.8:
    print('[ERROR] Gamma fix may have failed!')
    exit(1)
print('[OK] Gamma fix verified.')
"
echo ""

# ─────────────────────────────────────────────
# STEP 2: Stop SC-VAE training
# ─────────────────────────────────────────────
echo "[STEP 2] Stopping SC-VAE training..."
TRAIN_PID=$(pgrep -f "train_sc_vae.py" 2>/dev/null || true)
if [ -n "$TRAIN_PID" ]; then
    echo "  Sending SIGINT to PID=$TRAIN_PID..."
    kill -INT "$TRAIN_PID" 2>/dev/null || true
    # Wait for graceful shutdown (up to 60s)
    for i in $(seq 1 60); do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "  Training stopped gracefully after ${i}s."
            break
        fi
        sleep 1
    done
    # Force kill if still running
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "  Force killing..."
        kill -9 "$TRAIN_PID" 2>/dev/null || true
        sleep 2
    fi
else
    echo "  No training process found."
fi

# Find latest checkpoint
LATEST_CKPT=$(ls -t checkpoints/sc_vae_shape/epoch_*.pt 2>/dev/null | head -1)
echo "  Latest checkpoint: $LATEST_CKPT"
echo ""

# ─────────────────────────────────────────────
# STEP 3: Re-pack LMDB
# ─────────────────────────────────────────────
echo "[STEP 3] Re-packing LMDB from fixed cache..."
echo "  This will take ~1-3 hours on HDD..."
PYTHONUNBUFFERED=1 python scripts/pack_lmdb_fast.py
echo "[STEP 3] LMDB re-pack completed."
echo ""

# Verify LMDB
echo "[STEP 3.1] Verifying LMDB gamma..."
python -c "
import torch, lmdb, io
env = lmdb.open('data/ovoxel_cache_lmdb', readonly=True, lock=False, readahead=False)
txn = env.begin()
cursor = txn.cursor()
cursor.first()
data = torch.load(io.BytesIO(cursor.value()), weights_only=False)
gamma = data['features'][:, 6].float()
aabb = data.get('aabb', None)
print(f'LMDB gamma: min={gamma.min():.6f}, max={gamma.max():.6f}, mean={gamma.mean():.6f}')
if aabb is not None:
    print(f'LMDB aabb: {aabb.tolist()}')
if gamma.min() < 0.999:
    print('[OK] LMDB gamma is variable (fix applied).')
else:
    print('[ERROR] LMDB gamma still constant 1.0!')
    exit(1)
env.close()
"
echo ""

# ─────────────────────────────────────────────
# STEP 4: Resume SC-VAE Training
# ─────────────────────────────────────────────
echo "[STEP 4] Resuming SC-VAE training from $LATEST_CKPT..."
echo "  Target: 500 epochs"
echo "  Starting at: $(date)"
echo ""

PYTHONUNBUFFERED=1 python src/train_sc_vae.py \
    --resume "$LATEST_CKPT" \
    --lr 1e-5 --no-torch-compile \
    --gradient-accumulation-steps 33 \
    --save-every-steps 2000 --val-every-epochs 10 \
    --perf-log-every-steps 50 --num-workers 4 \
    2>&1 | stdbuf -oL tee logs/train_sc_vae_gamma_fixed.log

echo ""
echo "=========================================="
echo "[AUTO] SC-VAE Training Completed!"
echo "[AUTO] $(date)"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────
# STEP 5: Precompute Slat Latent Cache for iMF
# ─────────────────────────────────────────────
# NOTE: precompute_sc_vae_cache.py chỉ cache O-Voxel features (Stage 1 data).
# Slat latents cho Stage 2 cần SC-VAE encoder → dùng script riêng.
BEST_CKPT=$(ls -t checkpoints/sc_vae_shape/epoch_*.pt 2>/dev/null | head -1)
echo "[STEP 5] Precomputing Slat latents for iMF Stage 2..."
echo "  SC-VAE checkpoint: $BEST_CKPT"
echo "  This encodes all meshes with SC-VAE and caches Slat+Context to disk."
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
echo "  Mode: --offline-data (reads from precomputed Slat cache)"
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
