#!/usr/bin/env bash
# Auto pipeline: Fix FLAME context bug → recompute LMDB → restart training
#
# Sequence:
#   1. Backup old LMDBs + checkpoints
#   2. Smoke-test new FLAME adapter (MediaPipe)
#   3. Recompute hybrid_context.lmdb (~25-30 min, 20K samples)
#   4. Repack slat_context.lmdb với new context (~10 min)
#   5. Restart training from epoch 0
#
# Run: nohup bash scripts/auto_fix_flame_and_retrain.sh > logs/auto_fix.log 2>&1 &
# Or via tmux: tmux new -d -s autofix "bash scripts/auto_fix_flame_and_retrain.sh"

set -e
cd "$(dirname "$0")/.."

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/auto_fix_${TS}.log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  AUTO FLAME FIX + RETRAIN PIPELINE"
echo "=============================================="
echo "  Started: $(date)"
echo "  Log: ${LOG}"
echo "=============================================="

# === Step 1: Backup ===
echo ""
echo "[1/5] Backing up old broken-FLAME LMDBs + checkpoints..."
BACKUP_DIR="data/backup_broken_flame_${TS}"
mkdir -p "${BACKUP_DIR}"
mv data/hybrid_context.lmdb "${BACKUP_DIR}/hybrid_context.lmdb" 2>/dev/null || true
mv data/slat_context.lmdb "${BACKUP_DIR}/slat_context.lmdb" 2>/dev/null || true
mv checkpoints/imf_unet "${BACKUP_DIR}/checkpoints_imf_unet" 2>/dev/null || true
mkdir -p checkpoints/imf_unet
echo "  Backed up to: ${BACKUP_DIR}"
du -sh "${BACKUP_DIR}"/* 2>/dev/null | head -5

# Nếu BACKUP_DIR/slat_context.lmdb không tồn tại (re-run sau khi đã backup từ trước),
# tìm backup folder cũ nhất có slat_context.lmdb → dùng cho step 4 remix
if [ ! -d "${BACKUP_DIR}/slat_context.lmdb" ]; then
    PREV_BACKUP=$(ls -td data/backup_broken_flame_*/slat_context.lmdb 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "${PREV_BACKUP}" ] && [ -d "${PREV_BACKUP}/slat_context.lmdb" ]; then
        echo "  [INFO] No fresh slat backup in ${BACKUP_DIR}; using previous: ${PREV_BACKUP}"
        BACKUP_DIR="${PREV_BACKUP}"
    fi
fi

# === Step 2: Smoke test new FLAME ===
echo ""
echo "[2/5] Smoke testing new MediaPipe FLAME adapter..."
python -c "
from src.data.flame_adapter import FLAMEExpressionAdapter
flame = FLAMEExpressionAdapter(expression_dim=50, device='cpu')
import torch
out = flame.extract_from_image(torch.zeros(1, 3, 256, 256))
print(f'  Output shape: {out.shape}')
assert out.shape == (1, 50), f'Expected [1, 50] got {out.shape}'
print('  ✓ Smoke test passed')
" || { echo "  ✗ Smoke test FAILED. Aborting."; exit 1; }

# === Step 3: Recompute hybrid_context.lmdb ===
echo ""
echo "[3/5] Recomputing hybrid_context.lmdb (~25-30 min, 20K samples)..."
echo "       Output: data/hybrid_context.lmdb"
START_T=$(date +%s)
python scripts/build_context_lmdb.py \
    --out-lmdb data/hybrid_context.lmdb \
    --dirs /mnt/16TData/Datasets/FaceVerse_3D/FaceVerse /mnt/16TData/Datasets/FaceScape \
    --device cuda:0
ELAPSED=$(( $(date +%s) - START_T ))
echo "  Done in ${ELAPSED}s. Verifying..."

python -c "
import lmdb, io, torch, json
env = lmdb.open('data/hybrid_context.lmdb', readonly=True, lock=False)
with env.begin() as txn:
    cnt = 0
    sample_keys = []
    for k, _ in txn.cursor():
        if k == b'__meta__': continue
        cnt += 1
        if len(sample_keys) < 3:
            sample_keys.append(k)
    # Verify FLAME varies between samples
    flames = []
    for k in sample_keys:
        raw = txn.get(k)
        ctx = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False).float()
        if ctx.ndim == 1: ctx = ctx.unsqueeze(0)
        flames.append(ctx[0, 512:562].numpy())
    import numpy as np
    if len(flames) >= 2:
        diff = np.abs(flames[0] - flames[1]).max()
        print(f'  Total entries: {cnt}')
        print(f'  FLAME max_diff first 2 samples: {diff:.4f}')
        if diff < 0.001:
            print('  ✗ FLAME STILL CONSTANT! Bug not fixed.')
            exit(1)
        else:
            print('  ✓ FLAME context varies correctly')
"

# === Step 4: Remix slat_context.lmdb (giữ slat cũ, thay context mới) ===
echo ""
echo "[4/5] Remixing slat_context.lmdb (keep slat, swap context)..."
START_T=$(date +%s)
OLD_SLAT_LMDB="${BACKUP_DIR}/slat_context.lmdb"
if [ ! -d "${OLD_SLAT_LMDB}" ]; then
    echo "  ✗ Backup slat LMDB not found at ${OLD_SLAT_LMDB}. Aborting."
    exit 1
fi
python scripts/remix_slat_lmdb_with_new_context.py \
    --old-slat-lmdb "${OLD_SLAT_LMDB}" \
    --new-context-lmdb data/hybrid_context.lmdb \
    --out-lmdb data/slat_context.lmdb
ELAPSED=$(( $(date +%s) - START_T ))
echo "  Done in ${ELAPSED}s"

python -c "
import lmdb, json
env = lmdb.open('data/slat_context.lmdb', readonly=True, lock=False)
with env.begin() as t:
    meta = t.get(b'__meta__')
    print(f'  Meta: {json.loads(meta).get(\"packed\", \"?\") if meta else \"?\"} entries')
"

# === Step 5: Restart training from epoch 0 ===
echo ""
echo "[5/5] Restarting training from epoch 0..."
echo "  Config: NUM_WORKERS=4, batch=2, grad_accum=32, lr=2e-4, epochs=400"
echo "  Curriculum: switch_ratio=0.3 (ep120)"
RESUME="" bash scripts/train_imf.sh

echo ""
echo "=============================================="
echo "  AUTO PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "=============================================="
