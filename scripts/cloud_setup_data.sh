#!/bin/bash

# Script: cloud_setup_data.sh
# Purpose: Extract data and prepare checkpoints on the new Cloud GPU machine.

echo "=========================================="
echo "FaceDiff Cloud Setup - Data & Checkpoints"
echo "=========================================="

# Define paths (adjusting to ~/facediff/DATN as per your GitHub repo)
BASE_DIR="$HOME/facediff"
REPO_DIR="$BASE_DIR/DATN"
DATA_DIR="$REPO_DIR/data"
TAR_FILE="$BASE_DIR/data/ovoxel_cache.tar"

# 1. Create necessary directories
echo "[1/3] Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$REPO_DIR/checkpoints/sc_vae_shape"

# 2. Extract Data
if [ -f "$TAR_FILE" ]; then
    echo "[2/3] Extracting 272GB LMDB data (this may take 20-30 mins)..."
    tar -xvf "$TAR_FILE" -C "$DATA_DIR"
    echo "Done! Data extracted to $DATA_DIR/ovoxel_cache_lmdb"
else
    echo "ERROR: File $TAR_FILE not found! Please make sure you downloaded it from Drive to $BASE_DIR/data/"
    exit 1
fi

# 3. Move Checkpoints
echo "[3/3] Organizing checkpoints..."
# Assuming you download checkpoints from Drive to $BASE_DIR/checkpoints/
if [ -d "$BASE_DIR/checkpoints" ]; then
    cp -r "$BASE_DIR/checkpoints"/* "$REPO_DIR/checkpoints/"
    echo "Checkpoints moved to $REPO_DIR/checkpoints/"
else
    echo "Note: $BASE_DIR/checkpoints not found. If you downloaded them elsewhere, please move them to $REPO_DIR/checkpoints/ manually."
fi

echo "=========================================="
echo "SETUP COMPLETE! You can now run:"
echo "cd $REPO_DIR && bash scripts/resume_from_397.sh"
echo "=========================================="
