#!/usr/bin/env python3
"""Visualize FULL Ground Truth O-Voxel data (no intersection filtering).

Loads cached O-Voxel, extracts mesh via Dual Contouring, compares with
original mesh. Exports PLY files and renders comparison figures.

Usage:
    python scripts/visualize_full_gt.py
    python scripts/visualize_full_gt.py --samples faceverse/005_19_005_19.c10.shape_mat.mx350000.pt
"""
import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Reuse functions from test script
from scripts.test_sc_vae_recon_v2 import (
    extract_ovoxel_mesh,
    export_colored_ply,
    export_colored_mesh_ply,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DEFAULT_SAMPLES = [
    ("FaceVerse-005_19", "data/ovoxel_cache_recached/faceverse/005_19_005_19.c10.shape_mat.mx350000.pt"),
    ("FaceVerse-022_02", "data/ovoxel_cache_recached/faceverse/022_02_022_02.c10.shape_mat.mx350000.pt"),
    ("FaceScape-100_dimpler", "data/ovoxel_cache_recached/facescape/100_models_reg_10_dimpler.c10.shape_mat.mx350000.pt"),
    ("FaceScape-100_sadness", "data/ovoxel_cache_recached/facescape/100_models_reg_14_sadness.c10.shape_mat.mx350000.pt"),
]

BASE_DIR = "/mnt/18TData/facediff"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/full_gt_diagnostic")


def find_original_mesh(cache_path: str) -> str | None:
    """Find original .obj from cache filename."""
    basename = os.path.basename(cache_path)
    name = basename.split(".c10.")[0]

    if "faceverse" in cache_path.lower():
        obj_path = f"/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/{name}/{name}.obj"
        if os.path.exists(obj_path):
            return obj_path
    elif "facescape" in cache_path.lower():
        parts = name.split("_models_reg_")
        if len(parts) == 2:
            subject, expr = parts
            obj_path = f"/mnt/16TData/Datasets/FaceScape/{subject}/models_reg/{expr}.obj"
            if os.path.exists(obj_path):
                return obj_path
    return None


def render_trimesh_to_pointcloud(mesh, n_points=50000):
    """Sample points from trimesh for point cloud visualization."""
    try:
        points, face_idx = mesh.sample(n_points, return_index=True)
        if mesh.visual and hasattr(mesh.visual, 'face_colors'):
            colors = mesh.visual.face_colors[face_idx, :3] / 255.0
        else:
            colors = np.ones((len(points), 3)) * 0.7  # gray
        return points, colors
    except Exception:
        # Fallback: use vertices directly
        if len(mesh.vertices) > n_points:
            idx = np.random.choice(len(mesh.vertices), n_points, replace=False)
        else:
            idx = np.arange(len(mesh.vertices))
        points = mesh.vertices[idx]
        colors = np.ones((len(idx), 3)) * 0.7
        return points, colors


def save_comparison_figure(panels, out_png, title):
    """Save multi-panel 3D scatter comparison.

    panels: list of (xyz, rgb, subtitle) tuples
    """
    if not HAS_MPL:
        print("  [WARN] matplotlib not available, skipping figure")
        return

    n = len(panels)
    fig = plt.figure(figsize=(6 * n, 6))

    def subsample(xyz, rgb, n_max=50000):
        if len(xyz) <= n_max:
            return xyz, rgb
        idx = np.random.choice(len(xyz), size=n_max, replace=False)
        return xyz[idx], rgb[idx]

    for i, (xyz, rgb, subtitle) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        xs, cs = subsample(xyz, rgb)
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=cs, s=0.5)
        ax.set_title(subtitle, fontsize=11)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved figure: {out_png}")


def process_sample(cache_path: str, label: str, out_dir: str):
    """Process one sample: load cache, extract full GT mesh, compare with original."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if not os.path.exists(cache_path):
        print(f"  [ERROR] Not found: {cache_path}")
        return

    # Load cache
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    features = data["features"].to(torch.float32)
    coords = data["coords"].to(torch.int32)
    resolution = int(data.get("resolution", 256))
    aabb = data.get("aabb", torch.tensor([[-0.5]*3, [0.5]*3], dtype=torch.float32))
    if isinstance(aabb, np.ndarray):
        aabb = torch.from_numpy(aabb).float()

    n_voxels = features.shape[0]
    print(f"  Loaded: {n_voxels:,} voxels, resolution={resolution}")

    sample_dir = os.path.join(out_dir, label)
    os.makedirs(sample_dir, exist_ok=True)

    # Compute FULL GT world positions
    voxel_size = (aabb[1] - aabb[0]) / float(resolution)
    dv = features[:, 0:3]
    rgb = features[:, 7:10].clamp(0, 1)

    full_world = ((coords.float() + dv) * voxel_size + aabb[0]).numpy()
    full_rgb = rgb.numpy()

    # Export full GT point cloud
    export_colored_ply(full_world, full_rgb, os.path.join(sample_dir, "full_gt_pointcloud.ply"))
    print(f"  Exported: full_gt_pointcloud.ply ({n_voxels:,} points)")

    # Count voxels with 0 intersected edges (potential "floating" voxels)
    delta = features[:, 3:6]
    delta_sum = delta.sum(dim=1)
    n_no_edges = (delta_sum == 0).sum().item()
    print(f"  Voxels with 0 intersected edges: {n_no_edges:,} ({100*n_no_edges/n_voxels:.1f}%)")
    print(f"  Voxels with >=1 intersected edge: {n_voxels - n_no_edges:,}")

    # Extract mesh from FULL GT via Dual Contouring
    # Use heavier smoothing (20 iters) to remove voxel grid artifacts
    print(f"  Extracting mesh via Dual Contouring (smooth=20, knn=16)...")
    gt_verts, gt_faces, gt_colors = extract_ovoxel_mesh(
        coords, features, aabb.numpy(), resolution, is_logits=False,
        smooth_iters=20, color_knn=16
    )
    if gt_verts is not None:
        mesh_path = os.path.join(sample_dir, "full_gt_mesh.ply")
        export_colored_mesh_ply(gt_verts, gt_faces, gt_colors, mesh_path)
        print(f"  Exported: full_gt_mesh.ply ({len(gt_verts):,} verts, {len(gt_faces):,} faces)")
    else:
        print(f"  [WARN] Dual contouring failed")
        gt_verts, gt_faces = None, None

    # Load original mesh if available
    panels = []
    obj_path = find_original_mesh(cache_path)
    if obj_path:
        try:
            import trimesh
            orig_mesh = trimesh.load(obj_path, process=False)
            if isinstance(orig_mesh, trimesh.Scene):
                orig_mesh = orig_mesh.dump(concatenate=True)

            # Normalize original mesh to same coordinate space
            v = orig_mesh.vertices
            v_min, v_max = v.min(axis=0), v.max(axis=0)
            center = (v_min + v_max) / 2
            scale = 0.95 / max(v_max - v_min)
            v_norm = (v - center) * scale

            orig_pts, orig_colors = render_trimesh_to_pointcloud(orig_mesh)
            # Normalize sampled points too
            orig_pts_norm = (orig_pts - center) * scale

            export_colored_ply(orig_pts_norm, orig_colors, os.path.join(sample_dir, "original_mesh_sampled.ply"))
            print(f"  Exported: original_mesh_sampled.ply ({len(orig_pts):,} points)")
            print(f"  Original mesh: {len(orig_mesh.vertices):,} verts, {len(orig_mesh.faces):,} faces")

            panels.append((orig_pts_norm, orig_colors, f"Original Mesh\n({len(orig_mesh.faces):,} faces)"))
        except Exception as e:
            print(f"  [WARN] Could not load original: {e}")

    panels.append((full_world, full_rgb, f"Full GT O-Voxel\n({n_voxels:,} voxels)"))

    if gt_verts is not None:
        # Sample points from GT mesh for scatter plot
        n_sample = min(50000, len(gt_verts))
        if len(gt_verts) > n_sample:
            idx = np.random.choice(len(gt_verts), n_sample, replace=False)
        else:
            idx = np.arange(len(gt_verts))
        panels.append((gt_verts[idx], gt_colors[idx], f"GT Mesh (DC)\n({len(gt_faces):,} faces)"))

    # Save comparison figure
    save_comparison_figure(
        panels,
        os.path.join(sample_dir, "full_gt_comparison.png"),
        f"Full GT Diagnostic: {label}"
    )

    # Chamfer distance: GT mesh vs original mesh (if both available)
    if gt_verts is not None and obj_path:
        try:
            from scipy.spatial import cKDTree
            # Sample from GT mesh evenly
            gt_sample = gt_verts[np.random.choice(len(gt_verts), min(10000, len(gt_verts)), replace=False)]
            orig_sample = orig_pts_norm[np.random.choice(len(orig_pts_norm), min(10000, len(orig_pts_norm)), replace=False)]

            tree_gt = cKDTree(gt_sample)
            tree_orig = cKDTree(orig_sample)

            d_gt2orig, _ = tree_gt.query(orig_sample)
            d_orig2gt, _ = tree_orig.query(gt_sample)

            cd = (d_gt2orig.mean() + d_orig2gt.mean()) / 2
            hausdorff = max(d_gt2orig.max(), d_orig2gt.max())
            print(f"  Chamfer Distance (orig↔gt_mesh): {cd:.6f}")
            print(f"  Hausdorff Distance: {hausdorff:.6f}")
        except Exception as e:
            print(f"  [WARN] Could not compute Chamfer: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize full GT O-Voxel data")
    parser.add_argument("--samples", nargs="+", help="Specific .pt cache files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}\n")

    if args.samples:
        for path in args.samples:
            full_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
            process_sample(full_path, os.path.basename(path).split(".c10.")[0], args.output_dir)
    else:
        for label, path in DEFAULT_SAMPLES:
            full_path = os.path.join(BASE_DIR, path)
            process_sample(full_path, label, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Done! Check PLY files in: {args.output_dir}")
    print(f"  Open in MeshLab to verify mesh quality.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
