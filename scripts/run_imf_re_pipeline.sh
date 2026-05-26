#!/usr/bin/env bash
# iMF full re-pipeline script.
#
# Tasks performed:
# 1. Terminate any active train_sc_vae.py process to free up VRAM.
# 2. Extract latent shapes (slat) & context from FaceVerse dataset using epoch_600.pt checkpoint.
# 3. Pack the precomputed slats into a single LMDB (data/slat_context.lmdb).
# 4. Rebalance context vectors (data/slat_context_balanced.lmdb) using balance_hybrid_context_segments.
# 5. Recompute the slat channel mean/std statistics (data/slat_stats_faceverse.pt)
# Ensure native libraries (OpenMP, MKL) use all CPU cores
# (PyTorch intra‑ and inter‑op threads are set inside the Python script)
# 6. Kick off iMF training using the newly generated balanced LMDB and statistics.

set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="/mnt/18TData/facediff/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate facediff

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# We will manage thread counts per-step below, since parallel sharding requires OMP_NUM_THREADS=1
# export OMP_NUM_THREADS=$(nproc)
# export MKL_NUM_THREADS=$(nproc)

echo "=============================================================="
# Step 1: Kill active training processes
# echo "[Step 1/6] Terminating active train_sc_vae.py training processes..."
# SC_VAE_PIDS=$(ps aux | grep train_sc_vae.py | grep -v grep | awk '{print $2}' || true)
# if [ -n "${SC_VAE_PIDS}" ]; then
#   echo "Killing PIDs: ${SC_VAE_PIDS}"
#   kill -9 ${SC_VAE_PIDS} || true
# else
#   echo "No active train_sc_vae.py processes found."
# fi

# Step 2: Precompute SLAT cache
# echo "[Step 2/6] Precomputing SLAT + Context cache (ALL meshes, no ID filters)..."
# mkdir -p data/fv_slat_cache
# NUM_SHARDS=12
# echo "Launching $NUM_SHARDS parallel shards to cache remaining files..."
# export OMP_NUM_THREADS=2
# export MKL_NUM_THREADS=2
# for i in $(seq 0 $((NUM_SHARDS-1))); do
#     python scripts/data/precompute_slat_cache.py \
#         --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt \
#         --dataset faceverse \
#         --fv-cache-dir data/fv_slat_cache \
#         --context-lmdb data/hybrid_context.lmdb \
#         --shard-id $i --num-shards $NUM_SHARDS \
#         --num-workers 0 \
#         --disable-id-filters \
#         --skip-existing \
#         --device cuda:0 > logs/precompute_shard_${i}.log 2>&1 &
#     sleep 2
# done
# wait
# echo "All shards completed."

# Step 3: Pack into LMDB
echo "[Step 3/6] Packing precomputed .pt files into LMDB (FaceVerse only)..."
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
rm -rf data/slat_context_faceverse.lmdb
python scripts/data/pack_slat_lmdb.py \
    --fv-cache-dir data/fv_slat_cache \
    --fs-cache-dir "" \
    --output data/slat_context_faceverse.lmdb

# Step 4: Rebalance Context LMDB
echo "[Step 4/6] Rebalancing context vectors..."
rm -rf data/slat_context_faceverse_balanced.lmdb
python scripts/data/rebalance_slat_context_lmdb.py \
    --in-lmdb data/slat_context_faceverse.lmdb \
    --out-lmdb data/slat_context_faceverse_balanced.lmdb

# Step 5: Compute Slat statistics and check distribution
echo "[Step 5/6] Computing slat channel mean & std statistics..."
python scripts/data/compute_slat_stats.py \
    --lmdb data/slat_context_faceverse.lmdb \
    --out data/slat_stats_faceverse.pt \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_600.pt

# Step 6: Start iMF training
echo "[Step 6/6] Launching iMF v8 training using the new slat cache..."
# Force resuming from the latest epoch_240.pt checkpoint
export FRESH_START=0
export RESUME="checkpoints/imf_v8_lite/epoch_240.pt"
export RESUME_MODEL_ONLY=0
export SLAT_LMDB="data/slat_context_faceverse_balanced.lmdb"
bash scripts/train_imf_v8.sh

echo "=============================================================="
echo "✔ All steps initiated successfully! Pipeline is running."
echo "=============================================================="
