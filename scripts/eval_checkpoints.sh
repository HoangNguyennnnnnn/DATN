#!/bin/bash
# =================================================================
# Evaluate SC-VAE checkpoints: 390, 400, 450, 470
# Generates point clouds + DC mesh + Poisson mesh for 3 samples each
# Output: /mnt/18TData/facediff/outputs/eval_checkpoints/
# =================================================================
set -e

export PYTHONPATH=/mnt/18TData/facediff:$PYTHONPATH
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate facediff

EPOCHS="390 400 450 470"
NUM_SAMPLES=3
OUT_ROOT="/mnt/18TData/facediff/outputs/eval_checkpoints"

echo "============================================================"
echo "  SC-VAE Checkpoint Evaluation"
echo "  Epochs: $EPOCHS"
echo "  Samples per epoch: $NUM_SAMPLES"
echo "  Output: $OUT_ROOT"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for EP in $EPOCHS; do
    CKPT="/mnt/18TData/facediff/checkpoints/sc_vae_shape/epoch_${EP}.pt"
    EP_OUT="$OUT_ROOT/epoch_${EP}"
    
    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] epoch_${EP}.pt not found"
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "  Evaluating Epoch $EP"
    echo "  $(date '+%H:%M:%S')"
    echo "============================================================"
    
    python -u /mnt/18TData/facediff/.gemini/antigravity/brain/b7d64c3b-9b7e-4c2e-8c45-84a04e2e94e7/scratch/eval_checkpoint.py \
        --ckpt "$CKPT" \
        --out-dir "$EP_OUT" \
        --num-samples $NUM_SAMPLES \
        --epoch $EP
    
    echo "  [DONE] Epoch $EP → $EP_OUT"
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUT_ROOT"
echo "============================================================"
