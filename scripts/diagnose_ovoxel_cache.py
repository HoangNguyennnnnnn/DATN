#!/usr/bin/env python3
"""Diagnose O-Voxel cache quality: delta, gamma, dv, rgb statistics.

CPU-only script. Checks if cached data has quality issues (non-binary delta,
wrong gamma range, NaN/Inf, etc).

Usage:
    python scripts/diagnose_ovoxel_cache.py
    python scripts/diagnose_ovoxel_cache.py --samples path1.pt path2.pt
"""
import argparse
import os
import sys
import torch
import numpy as np

# Default test samples (both datasets)
DEFAULT_SAMPLES = [
    ("FaceVerse", "data/ovoxel_cache_recached/faceverse/005_19_005_19.c10.shape_mat.mx350000.pt"),
    ("FaceVerse", "data/ovoxel_cache_recached/faceverse/022_02_022_02.c10.shape_mat.mx350000.pt"),
    ("FaceScape", "data/ovoxel_cache_recached/facescape/100_models_reg_10_dimpler.c10.shape_mat.mx350000.pt"),
    ("FaceScape", "data/ovoxel_cache_recached/facescape/100_models_reg_14_sadness.c10.shape_mat.mx350000.pt"),
]


def find_original_mesh(cache_path: str) -> str | None:
    """Heuristic to find original .obj mesh from cache filename."""
    basename = os.path.basename(cache_path)
    # Remove suffixes: .c10.shape_mat.mx350000.pt
    name = basename.split(".c10.")[0]

    if "faceverse" in cache_path.lower():
        obj_path = f"/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/{name}/{name}.obj"
        if os.path.exists(obj_path):
            return obj_path
    elif "facescape" in cache_path.lower():
        # Format: {subject}_models_reg_{expression}
        parts = name.split("_models_reg_")
        if len(parts) == 2:
            subject, expr = parts
            obj_path = f"/mnt/16TData/Datasets/FaceScape/{subject}/models_reg/{expr}.obj"
            if os.path.exists(obj_path):
                return obj_path
    return None


def diagnose_sample(cache_path: str, label: str):
    """Load and diagnose a single cache file."""
    print(f"\n{'='*70}")
    print(f"  {label}: {os.path.basename(cache_path)}")
    print(f"  Path: {cache_path}")
    print(f"{'='*70}")

    if not os.path.exists(cache_path):
        print(f"  [ERROR] File not found!")
        return

    # Load cache
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    features = data["features"].to(torch.float32)
    coords = data["coords"].to(torch.int32)
    resolution = int(data.get("resolution", 256))
    aabb = data.get("aabb", None)

    n_voxels = features.shape[0]
    n_channels = features.shape[1]
    print(f"\n  Voxels: {n_voxels:,}")
    print(f"  Channels: {n_channels}")
    print(f"  Resolution: {resolution}")
    if aabb is not None:
        aabb_np = aabb.numpy() if isinstance(aabb, torch.Tensor) else np.array(aabb)
        print(f"  AABB: {aabb_np.tolist()}")

    # Coord range
    coord_min = coords.min(dim=0).values.tolist()
    coord_max = coords.max(dim=0).values.tolist()
    print(f"\n  Coord range: min={coord_min}, max={coord_max}")
    out_of_bounds = ((coords < 0) | (coords >= resolution)).any(dim=1).sum().item()
    if out_of_bounds > 0:
        print(f"  [WARNING] {out_of_bounds} coords out of [0, {resolution-1}] range!")
    else:
        print(f"  Coords OK: all within [0, {resolution-1}]")

    # NaN/Inf check
    nan_count = torch.isnan(features).sum().item()
    inf_count = torch.isinf(features).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"\n  [CRITICAL] NaN: {nan_count}, Inf: {inf_count}")
    else:
        print(f"\n  NaN/Inf: NONE (clean)")

    # === DV channels [0:3] ===
    dv = features[:, 0:3]
    print(f"\n  --- DV (dual vertex offsets) [0:3] ---")
    print(f"  Range: [{dv.min().item():.6f}, {dv.max().item():.6f}]")
    print(f"  Mean: {dv.mean().item():.6f}, Std: {dv.std().item():.6f}")
    below_0 = (dv < 0).sum().item()
    above_1 = (dv > 1).sum().item()
    if below_0 > 0 or above_1 > 0:
        print(f"  [WARNING] {below_0} values < 0, {above_1} values > 1")
    else:
        print(f"  All values in [0, 1]: OK")

    # === DELTA channels [3:6] ===
    delta = features[:, 3:6]
    print(f"\n  --- DELTA (intersection flags) [3:6] ---")
    print(f"  Range: [{delta.min().item():.6f}, {delta.max().item():.6f}]")

    # Check binary
    unique_vals = torch.unique(delta)
    is_binary = (unique_vals.numel() <= 2) and all(v in [0.0, 1.0] for v in unique_vals.tolist())
    non_binary_mask = ~((delta == 0.0) | (delta == 1.0))
    n_non_binary = non_binary_mask.sum().item()

    if is_binary:
        print(f"  Unique values: {unique_vals.tolist()} — BINARY OK")
    else:
        print(f"  Unique values ({unique_vals.numel()}): {unique_vals[:20].tolist()}")
        print(f"  [WARNING] {n_non_binary} non-binary values detected!")
        if n_non_binary > 0:
            non_bin_vals = delta[non_binary_mask]
            print(f"  Non-binary sample: {non_bin_vals[:10].tolist()}")

    # Delta statistics
    n_ones = (delta == 1.0).sum().item()
    n_zeros = (delta == 0.0).sum().item()
    total_flags = delta.numel()
    print(f"  Flags=1: {n_ones:,} ({100*n_ones/total_flags:.1f}%)")
    print(f"  Flags=0: {n_zeros:,} ({100*n_zeros/total_flags:.1f}%)")

    # Per-voxel delta sum (how many edges intersected per voxel)
    delta_sum = delta.sum(dim=1)
    for k in [0, 1, 2, 3]:
        count = (delta_sum == k).sum().item()
        print(f"  Voxels with {k} intersected edges: {count:,} ({100*count/n_voxels:.1f}%)")

    # === GAMMA channel [6] ===
    gamma = features[:, 6:7]
    print(f"\n  --- GAMMA (split weight) [6] ---")
    print(f"  Range: [{gamma.min().item():.6f}, {gamma.max().item():.6f}]")
    print(f"  Mean: {gamma.mean().item():.6f}, Std: {gamma.std().item():.6f}")

    if abs(gamma.min().item() - 1.0) < 1e-4 and abs(gamma.max().item() - 1.0) < 1e-4:
        print(f"  [BUG] Gamma is CONSTANT 1.0 — old unfixed cache!")
    elif gamma.min().item() >= 0.5 and gamma.max().item() <= 1.0:
        print(f"  Gamma range looks correct (post-fix)")
    else:
        print(f"  [WARNING] Unexpected gamma range")

    # === RGB channels [7:10] ===
    rgb = features[:, 7:10]
    print(f"\n  --- RGB (color) [7:10] ---")
    print(f"  Range: [{rgb.min().item():.6f}, {rgb.max().item():.6f}]")
    print(f"  Mean: {rgb.mean().item():.6f}, Std: {rgb.std().item():.6f}")
    below_0_rgb = (rgb < 0).sum().item()
    above_1_rgb = (rgb > 1).sum().item()
    if below_0_rgb > 0 or above_1_rgb > 0:
        print(f"  [WARNING] {below_0_rgb} values < 0, {above_1_rgb} values > 1")
    else:
        print(f"  All values in [0, 1]: OK")

    # === Compare with original mesh ===
    obj_path = find_original_mesh(cache_path)
    if obj_path:
        try:
            import trimesh
            mesh = trimesh.load(obj_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            print(f"\n  --- Original Mesh ---")
            print(f"  Path: {obj_path}")
            print(f"  Vertices: {len(mesh.vertices):,}")
            print(f"  Faces: {len(mesh.faces):,}")
            bbox_min = mesh.vertices.min(axis=0)
            bbox_max = mesh.vertices.max(axis=0)
            bbox_size = bbox_max - bbox_min
            print(f"  BBox size: {bbox_size}")
            print(f"  Voxel/Face ratio: {n_voxels / len(mesh.faces):.2f}x")
        except Exception as e:
            print(f"\n  [WARN] Could not load original mesh: {e}")
    else:
        print(f"\n  Original mesh: not found")

    print()
    return {
        "n_voxels": n_voxels,
        "delta_binary": is_binary,
        "delta_non_binary_count": n_non_binary,
        "gamma_min": gamma.min().item(),
        "gamma_max": gamma.max().item(),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose O-Voxel cache quality")
    parser.add_argument("--samples", nargs="+", help="Specific .pt files to check")
    args = parser.parse_args()

    print("=" * 70)
    print("  O-VOXEL CACHE DIAGNOSTIC")
    print("=" * 70)

    results = []
    if args.samples:
        for path in args.samples:
            r = diagnose_sample(path, os.path.basename(path))
            if r:
                results.append(r)
    else:
        for label, path in DEFAULT_SAMPLES:
            full_path = os.path.join("/mnt/18TData/facediff", path) if not os.path.isabs(path) else path
            r = diagnose_sample(full_path, label)
            if r:
                results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_binary = all(r["delta_binary"] for r in results)
    all_gamma_ok = all(r["gamma_min"] > 0.5 and r["gamma_max"] <= 1.001 for r in results)
    all_clean = all(r["nan_count"] == 0 and r["inf_count"] == 0 for r in results)

    print(f"  Delta binary: {'OK' if all_binary else 'ISSUES FOUND'}")
    print(f"  Gamma range:  {'OK' if all_gamma_ok else 'ISSUES FOUND'}")
    print(f"  NaN/Inf:      {'CLEAN' if all_clean else 'ISSUES FOUND'}")

    if all_binary and all_gamma_ok and all_clean:
        print(f"\n  >>> Cache data looks HEALTHY. Holes in GT are likely visualization artifact.")
    else:
        print(f"\n  >>> ISSUES DETECTED — may need to regenerate cache.")


if __name__ == "__main__":
    main()
