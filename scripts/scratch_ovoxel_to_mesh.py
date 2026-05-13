import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import extraction logic from test_sc_vae_recon_v2
from scripts.test_sc_vae_recon_v2 import load_cache_sample, extract_ovoxel_mesh, export_colored_mesh_ply

def verify_direct_conversion(cache_path, output_path):
    print(f"Loading cache: {cache_path}")
    feats, coords, resolution, aabb = load_cache_sample(cache_path)
    print(f"Points: {feats.shape[0]}, Resolution: {resolution}")

    print("Extracting mesh...")
    # feats has 10 channels: [v(3), delta(3), gamma(1), r(1), g(1), b(1)]
    # extract_ovoxel_mesh expects these channels
    verts, faces, colors = extract_ovoxel_mesh(
        coords, feats, aabb, resolution, 
        is_logits=False,  # Cache is already activated/processed
        threshold=0.5,
        target_faces=0,
        smooth_iters=6,
        color_knn=8
    )

    if verts is not None and faces is not None:
        print(f"Exporting mesh to: {output_path}")
        export_colored_mesh_ply(verts, faces, colors, output_path)
        print("Success!")
    else:
        print("Failed to extract mesh.")

if __name__ == "__main__":
    sample_pt = "data/ovoxel_cache_recached/faceverse/068_09_068_09.c10.shape_mat.mx350000.pt"
    out_ply = "outputs/verify_direct_068_09.ply"
    os.makedirs("outputs", exist_ok=True)
    
    if os.path.exists(sample_pt):
        verify_direct_conversion(sample_pt, out_ply)
    else:
        print(f"File not found: {sample_pt}")
