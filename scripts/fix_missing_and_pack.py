#!/usr/bin/env python3
"""
One-shot script: fill missing hybrid contexts → precompute missing slats → pack all to LMDB.
"""
from __future__ import annotations
import os, sys, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lmdb

# ── Step 1: Find missing context keys ──────────────────────────────────
print("=" * 60)
print("Step 1: Scanning for missing hybrid contexts")
print("=" * 60)

CONTEXT_LMDB = "data/hybrid_context.lmdb"
SLAT_CACHE_DIRS = {
    "faceverse": "data/slat_cache",
    "facescape": "data/slat_cache_facescape",
}

# Collect all obj_paths from precompute error log
ERRORS_LOG = "logs/precompute_slat_e500.log"
missing_obj_paths = []
if os.path.isfile(ERRORS_LOG):
    with open(ERRORS_LOG) as f:
        for line in f:
            if "Context not found in LMDB" in line:
                # Extract path: "... for /mnt/16TData/..."
                idx = line.find(" for /")
                if idx > 0:
                    path = line[idx+5:].split(".obj")[0] + ".obj"
                    path = path.strip()
                    if os.path.isfile(path):
                        missing_obj_paths.append(path)

missing_obj_paths = sorted(set(missing_obj_paths))
print(f"Found {len(missing_obj_paths)} meshes with missing context")

if missing_obj_paths:
    print("Filling missing contexts...")
    from src.data.mesh_renderer import MeshRenderer
    from src.data.arcface_extractor import ArcFaceExtractor
    from src.data.flame_adapter import FLAMEExpressionAdapter
    from src.data.feature_extractor import DinoV3Extractor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    renderer = MeshRenderer(device=device, image_size=512)
    arcface = ArcFaceExtractor(device=device)
    flame = FLAMEExpressionAdapter(expression_dim=50, device=device)
    dino = DinoV3Extractor(model_name="facebook/dinov2-small", device=device)

    DATA_DIRS = [
        ("/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse", "faceverse"),
        ("/mnt/16TData/Datasets/FaceScape", "facescape"),
    ]

    env = lmdb.open(CONTEXT_LMDB, map_size=50 * 1024**3)
    txn = env.begin(write=True)
    filled = 0
    failed = 0

    for obj_path in missing_obj_paths:
        # Build key
        dataset_name = "unknown"
        rel_path = obj_path
        for d, name in DATA_DIRS:
            if obj_path.startswith(d):
                dataset_name = name
                rel_path = os.path.relpath(obj_path, d)
                break
        key = f"{dataset_name}/{rel_path}".encode("utf-8")

        if txn.get(key) is not None:
            filled += 1
            continue

        try:
            front, back = renderer.render_front_and_back(obj_path)
            id_vec = arcface.extract_identity(front)
            exp_vec = flame.extract_from_image(front)
            shape_vec = dino.extract_features(back)
            context = torch.cat([id_vec, exp_vec, shape_vec], dim=-1).squeeze(0).cpu()
            buf = io.BytesIO()
            torch.save(context.half(), buf)
            txn.put(key, buf.getvalue())
            filled += 1
        except Exception as e:
            print(f"  [FAIL] {obj_path}: {e}")
            failed += 1

    txn.commit()
    env.close()
    print(f"Step 1 done: filled={filled}, failed={failed}")

    # Clean up GPU
    del renderer, arcface, flame, dino
    torch.cuda.empty_cache()
else:
    print("No missing contexts to fill.")

# ── Step 2: Re-precompute missing slat caches ─────────────────────────
print("\n" + "=" * 60)
print("Step 2: Re-precomputing missing slat caches")
print("=" * 60)

if missing_obj_paths and filled > 0:
    # Re-run precompute with --skip-existing to only fill gaps
    import subprocess
    cmd = [
        sys.executable, "scripts/precompute_slat_cache.py",
        "--sc-vae-ckpt", "checkpoints/sc_vae_shape/epoch_500.pt",
        "--dataset", "both",
        "--context-lmdb", CONTEXT_LMDB,
        "--ovoxel-lmdb", "data/ovoxel_cache_lmdb",
        "--skip-existing",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    print(f"Precompute exited with code {result.returncode}")
else:
    print("Skipping — no new contexts were added.")

# ── Step 3: Pack all .pt → slat_context.lmdb ──────────────────────────
print("\n" + "=" * 60)
print("Step 3: Packing slat cache → LMDB")
print("=" * 60)

import subprocess
cmd = [sys.executable, "scripts/pack_slat_lmdb.py"]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=False)
print(f"Pack LMDB exited with code {result.returncode}")

print("\n" + "=" * 60)
print("ALL DONE. Ready for: python src/train_imf.py --offline-data --slat-lmdb data/slat_context.lmdb")
print("=" * 60)
