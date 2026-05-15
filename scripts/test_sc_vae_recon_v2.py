#!/usr/bin/env python3
"""
SC-VAE Reconstruction Test v2 — Proper colored mesh output.

Fixes over v1:
1. Loads directly from O-Voxel cache (.pt) — same data pipeline as training
2. Exports colored point clouds (PLY with vertex RGB)
3. Uses 256³ grid for mesh extraction (matches training resolution)
4. Reduced smoothing to preserve face details
5. Exports colored mesh via marching cubes + barycentric color interpolation

Usage:
  python scripts/test_sc_vae_recon_v2.py
  python scripts/test_sc_vae_recon_v2.py --ckpt checkpoints/sc_vae_shape/epoch_200.pt --num-samples 5
"""

import argparse
import os
import sys
import random
import glob
import time
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from skimage.measure import marching_cubes
    HAS_MC = True
except Exception:
    HAS_MC = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TrainConfig
from src.models.sc_vae import SC_VAE, SPCONV_AVAILABLE

if SPCONV_AVAILABLE:
    import spconv.pytorch as spconv


def load_cache_sample(cache_path: str, max_voxels: int = 0):
    """Load a precomputed O-Voxel cache file."""
    pt = torch.load(cache_path, map_location="cpu", weights_only=False)
    feats = pt["features"].float()  # [N, 10]
    coords = pt["coords"].int()     # [N, 3]
    resolution = int(pt.get("resolution", 256))
    aabb = pt.get("aabb", None)

    if max_voxels > 0 and feats.shape[0] > max_voxels:
        idx = torch.randperm(feats.shape[0])[:max_voxels]
        feats = feats[idx]
        coords = coords[idx]

    return feats, coords, resolution, aabb


def build_sparse_input(feats: torch.Tensor, coords: torch.Tensor,
                       spatial_size: int, device: torch.device):
    """Build spconv.SparseConvTensor for SC-VAE forward pass."""
    batch_col = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    indices = torch.cat([batch_col, coords], dim=1).contiguous().to(device)
    sparse = spconv.SparseConvTensor(
        features=feats.contiguous().to(device),
        indices=indices,
        spatial_shape=[spatial_size] * 3,
        batch_size=1,
    )
    return sparse


def export_colored_ply(xyz: np.ndarray, rgb: np.ndarray, out_path: str):
    """Export colored point cloud as PLY file.

    Args:
        xyz: [N, 3] vertex positions
        rgb: [N, 3] vertex colors in [0, 1] range
    """
    n = xyz.shape[0]
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n")


def export_colored_mesh_ply(verts: np.ndarray, faces: np.ndarray,
                            colors: np.ndarray, out_path: str):
    """Export mesh with vertex colors as PLY.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] triangle indices
        colors: [V, 3] vertex colors in [0, 1]
    """
    nv = verts.shape[0]
    nf = faces.shape[0]
    rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(nv):
            f.write(f"{verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f} "
                    f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n")
        for i in range(nf):
            f.write(f"3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n")


def extract_poisson_mesh(world_xyz: np.ndarray, world_rgb: np.ndarray,
                          poisson_depth: int = 9, density_quantile: float = 0.01,
                          smooth_iters: int = 3, target_faces: int = 0):
    """
    Tạo mesh kín (watertight) từ Point Cloud bằng Poisson Surface Reconstruction.
    Đây là phương pháp thay thế cho Dual Contouring, phù hợp khi point cloud đẹp
    nhưng DC bị lỗ do thiếu voxel hàng xóm.
    
    Args:
        world_xyz: [N, 3] tọa độ world-space
        world_rgb: [N, 3] màu sRGB [0,1]
        poisson_depth: Độ sâu octree (cao hơn = chi tiết hơn, chậm hơn). 9-10 cho face mesh.
        density_quantile: Loại bỏ vùng mật độ thấp (nhiễu). 0.01 = bỏ 1% thấp nhất.
        smooth_iters: Số vòng Taubin smoothing sau khi tạo mesh.
        target_faces: Nếu > 0, decimation xuống số mặt này.
    Returns:
        verts, faces, colors (hoặc None, None, None nếu thất bại)
    """
    try:
        import open3d as o3d
        import trimesh as _tm
        
        # 1. Tạo point cloud với normals ước lượng
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(world_rgb, 0, 1).astype(np.float64))
        
        # Ước lượng normals (quan trọng cho Poisson)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        # Orient normals nhất quán (hướng ra ngoài)
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # 2. Poisson Surface Reconstruction
        o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        
        # 3. Loại bỏ vùng mật độ thấp (nhiễu bên ngoài bề mặt)
        densities_np = np.asarray(densities)
        density_threshold = np.quantile(densities_np, density_quantile)
        vertices_to_remove = densities_np < density_threshold
        o3d_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 4. Chuyển màu từ point cloud sang mesh bằng kNN
        from scipy.spatial import cKDTree
        mesh_verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
        tree = cKDTree(world_xyz)
        dist, idx = tree.query(mesh_verts, k=8)
        weights = 1.0 / np.maximum(dist, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        mesh_colors = (world_rgb[idx] * weights[..., None]).sum(axis=1)
        mesh_colors = np.clip(mesh_colors, 0, 1)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        
        # 5. Smooth
        if smooth_iters > 0:
            o3d_mesh = o3d_mesh.filter_smooth_taubin(
                number_of_iterations=smooth_iters,
                lambda_filter=0.5, mu=-0.53
            )
        
        # 6. Clean up
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()
        o3d_mesh.remove_non_manifold_edges()
        
        verts_np = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(o3d_mesh.triangles, dtype=np.int64)
        colors_np = np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)
        
        # 7. Decimation (nếu cần)
        if target_faces > 0 and len(faces_np) > target_faces:
            mesh_tm = _tm.Trimesh(vertices=verts_np, faces=faces_np,
                                   vertex_colors=(colors_np * 255).astype(np.uint8),
                                   process=False)
            mesh_tm = mesh_tm.simplify_quadric_decimation(target_faces)
            verts_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
            faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
            if mesh_tm.visual and hasattr(mesh_tm.visual, 'vertex_colors'):
                colors_np = np.asarray(mesh_tm.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
            else:
                # Fallback: re-color via kNN
                tree2 = cKDTree(world_xyz)
                d2, i2 = tree2.query(verts_np, k=8)
                w2 = 1.0 / np.maximum(d2, 1e-8)
                w2 /= w2.sum(axis=1, keepdims=True)
                colors_np = np.clip((world_rgb[i2] * w2[..., None]).sum(axis=1), 0, 1).astype(np.float32)
        
        print(f"    Poisson: {len(verts_np):,} verts, {len(faces_np):,} faces (depth={poisson_depth})")
        return verts_np, faces_np, colors_np
        
    except Exception as e:
        print(f"[WARN] Poisson mesh extraction failed: {e}")
        import traceback; traceback.print_exc()
        return None, None, None


def _prefill_boundary_voxels(coords, dv, flag, split_weight, grid_size):
    """
    Pre-fill missing boundary voxels so Dual Contouring can form complete quads.

    DC requires all 4 neighbor voxels for each edge quad. Boundary voxels in sparse
    grids often miss neighbors, causing dropped quads → holes. This adds synthetic
    voxels at missing neighbor positions with default values (cell center, no edges).
    """
    device = coords.device
    N = coords.shape[0]

    # Edge neighbor offsets — same as flexible_dual_grid.py
    edge_offsets = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # x-axis
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # y-axis
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # z-axis
    ], dtype=coords.dtype, device=device)  # [3, 4, 3]

    # Vectorized: compute all neighbor coords for intersected edges
    # edge_neighbors[i, axis, corner] = coords[i] + offset[axis, corner]
    edge_neighbors = coords.reshape(N, 1, 1, 3) + edge_offsets.unsqueeze(0)  # [N, 3, 4, 3]

    # Select only edges where flag is True: flag is [N, 3] bool
    # Expand flag to [N, 3, 4] and gather neighbor coords
    needed_all = edge_neighbors[flag]  # [M, 4, 3] where M = number of active edges
    if needed_all.numel() == 0:
        return coords, dv, flag, split_weight

    needed_flat = needed_all.reshape(-1, 3).long()  # [M*4, 3]

    # Filter out-of-bounds
    gs = int(grid_size)
    in_bounds = (
        (needed_flat[:, 0] >= 0) & (needed_flat[:, 0] < gs) &
        (needed_flat[:, 1] >= 0) & (needed_flat[:, 1] < gs) &
        (needed_flat[:, 2] >= 0) & (needed_flat[:, 2] < gs)
    )
    needed_flat = needed_flat[in_bounds]

    # Deduplicate
    needed_keys = needed_flat[:, 0] * (gs * gs) + needed_flat[:, 1] * gs + needed_flat[:, 2]
    needed_keys_unique = torch.unique(needed_keys)

    # Build existing coord keys
    cx, cy, cz = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    existing_keys = cx * (gs * gs) + cy * gs + cz

    # Find missing: needed but not existing
    # Use broadcasting comparison for GPU-friendly set difference
    # For large tensors, use a hash set approach via sorting
    all_keys = torch.cat([existing_keys, needed_keys_unique])
    all_keys_sorted, sort_idx = torch.sort(all_keys)
    # Mark duplicates (keys that appear in both existing and needed)
    is_dup = torch.zeros(all_keys.shape[0], dtype=torch.bool, device=device)
    is_dup[1:] = all_keys_sorted[1:] == all_keys_sorted[:-1]
    # Also mark the first occurrence of duplicates
    is_dup[:-1] |= all_keys_sorted[:-1] == all_keys_sorted[1:]
    # Unsort
    is_dup_unsorted = torch.zeros_like(is_dup)
    is_dup_unsorted[sort_idx] = is_dup
    # Missing = needed keys that are NOT duplicates (not in existing)
    n_existing = existing_keys.shape[0]
    missing_mask = ~is_dup_unsorted[n_existing:]
    missing_keys = needed_keys_unique[missing_mask]

    if missing_keys.numel() == 0:
        return coords, dv, flag, split_weight

    # Decode keys back to coords
    new_x = missing_keys // (gs * gs)
    new_y = (missing_keys % (gs * gs)) // gs
    new_z = missing_keys % gs
    new_coords = torch.stack([new_x, new_y, new_z], dim=1).to(coords.dtype)

    n_new = new_coords.shape[0]
    new_dv = torch.full((n_new, 3), 0.5, dtype=dv.dtype, device=device)
    new_flag = torch.zeros(n_new, 3, dtype=flag.dtype, device=device)

    coords_aug = torch.cat([coords, new_coords], dim=0)
    dv_aug = torch.cat([dv, new_dv], dim=0)
    flag_aug = torch.cat([flag, new_flag], dim=0)

    if split_weight is not None:
        new_sw = torch.ones(n_new, 1, dtype=split_weight.dtype, device=device)
        split_weight_aug = torch.cat([split_weight, new_sw], dim=0)
    else:
        split_weight_aug = None

    print(f"    Boundary pre-fill: added {n_new} synthetic voxels ({N} → {N + n_new})")
    return coords_aug, dv_aug, flag_aug, split_weight_aug


def extract_ovoxel_mesh(coords: torch.Tensor, feats: torch.Tensor, aabb, res: int,
                         is_logits: bool = False, threshold: float = 0.5, 
                         target_faces: int = 0, smooth_iters: int = 6, color_knn: int = 8):
    """
    Extract high-fidelity mesh using Dual Contouring with Open3D post-processing.
    """
    try:
        from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
    except ImportError:
        print("[WARN] o_voxel not installed, skipping dual grid extraction.")
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = coords.to(device)
    feats = feats.to(device)
    if isinstance(aabb, (list, tuple)):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    elif isinstance(aabb, np.ndarray):
        aabb = torch.from_numpy(aabb).to(dtype=torch.float32, device=device)
    else:
        aabb = aabb.clone().detach().to(dtype=torch.float32, device=device)

    # O-Voxel structure: [0:3]=dv, [3:6]=delta, [6:7]=gamma, [7:10]=rgb
    # Following TRELLIS.2 fdg_vae.py protocol EXACTLY:
    #   dv = (1 + 2*margin) * sigmoid(logits) - margin   (margin=0.5 → range [-0.5, 1.5])
    #   flag = logits > 0                                 (threshold on raw logits, NOT sigmoid)
    #   split_weight = softplus(gamma)                    (raw, no normalization)
    VOXEL_MARGIN = 0.5
    if is_logits:
        dv = (1 + 2 * VOXEL_MARGIN) * torch.sigmoid(feats[:, 0:3]) - VOXEL_MARGIN
        # Delta flag: threshold on RAW LOGITS (>0 ↔ sigmoid > 0.5), matching fdg_vae.py line 101
        flag = (feats[:, 3:6] > 0).bool()
        gamma = torch.nn.functional.softplus(feats[:, 6:7])
        rgb_linear_vox = torch.clamp(feats[:, 7:10], 0.0, 1.0)
    else:
        dv = feats[:, 0:3]
        flag = feats[:, 3:6].bool()
        gamma = feats[:, 6:7]
        rgb_linear_vox = feats[:, 7:10]

    # split_weight: pass softplus(gamma) directly, NO min-max normalization
    # (TRELLIS.2 fdg_vae.py line 89,102 passes raw softplus output)
    split_weight = gamma if (gamma.numel() > 0 and torch.isfinite(gamma).all()) else None

    # Pre-fill missing boundary voxels to prevent DC holes
    n_orig = coords.shape[0]
    coords, dv, flag, split_weight = _prefill_boundary_voxels(
        coords, dv, flag, split_weight, grid_size=int(res)
    )
    if coords.shape[0] > n_orig:
        n_synthetic = coords.shape[0] - n_orig
        rgb_pad = torch.full((n_synthetic, 3), 0.5, dtype=rgb_linear_vox.dtype, device=device)
        rgb_linear_vox = torch.cat([rgb_linear_vox, rgb_pad], dim=0)

    try:
        # 1. Dual Contouring Extraction (TRELLIS.2 protocol)
        verts, faces = flexible_dual_grid_to_mesh(
            coords, dv, flag, split_weight=split_weight, aabb=aabb, grid_size=int(res), train=False
        )
        
        verts_np = verts.cpu().numpy()
        faces_np = faces.cpu().numpy()
        
        # 2. Geometry Post-processing (trimesh repair + Open3D smoothing)
        try:
            import trimesh as _tm

            mesh = _tm.Trimesh(vertices=verts_np, faces=faces_np, process=False)
            n_before = len(mesh.faces)

            # Step 1: Fix normals & face orientations (KEY for correct shading)
            _tm.repair.fix_normals(mesh)
            _tm.repair.fix_winding(mesh)

            # Step 2: Fill small holes
            _tm.repair.fill_holes(mesh)

            # Step 3: Remove degenerate/duplicate faces
            mask_nondeg = mesh.nondegenerate_faces()
            if not mask_nondeg.all():
                mesh.update_faces(mask_nondeg)
            _, unique_idx = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)
            if len(unique_idx) < len(mesh.faces):
                mask_uniq = np.zeros(len(mesh.faces), dtype=bool)
                mask_uniq[unique_idx] = True
                mesh.update_faces(mask_uniq)
            mesh.remove_unreferenced_vertices()

            # Step 4: Remove only TINY disconnected components (noise)
            # BUG FIX: Trước đây giữ CHỈ mảnh lớn nhất → xoá mất nhiều bề mặt hợp lệ.
            # Giờ chỉ xoá mảnh < 1% tổng số faces (nhiễu thực sự).
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    total_faces = sum(len(c.faces) for c in components)
                    min_faces = max(100, int(total_faces * 0.01))  # 1% threshold
                    kept = [c for c in components if len(c.faces) >= min_faces]
                    if kept:
                        mesh = _tm.util.concatenate(kept)
                        n_removed = len(components) - len(kept)
                        if n_removed > 0:
                            print(f"    Removed {n_removed} tiny components (< {min_faces} faces)")
            except Exception:
                pass  # body_count can be slow/fail on large meshes

            # Step 5: Decimation (if target specified)
            if target_faces > 0 and len(mesh.faces) > target_faces:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                # Re-fix normals after decimation
                _tm.repair.fix_normals(mesh)

            # Step 6: Taubin smoothing via Open3D
            if smooth_iters > 0:
                import open3d as o3d
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(
                    np.asarray(mesh.vertices, dtype=np.float64))
                o3d_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(mesh.faces, dtype=np.int32))
                o3d_mesh = o3d_mesh.filter_smooth_taubin(
                    number_of_iterations=smooth_iters,
                    lambda_filter=0.5, mu=-0.53
                )
                o3d_mesh.remove_degenerate_triangles()
                o3d_mesh.remove_unreferenced_vertices()
                verts_np = np.asarray(o3d_mesh.vertices, dtype=np.float32)
                faces_np = np.asarray(o3d_mesh.triangles, dtype=np.int64)
            else:
                verts_np = np.asarray(mesh.vertices, dtype=np.float32)
                faces_np = np.asarray(mesh.faces, dtype=np.int64)

            n_after = faces_np.shape[0]
            if n_before != n_after:
                print(f"    Post-process: {n_before:,} → {n_after:,} faces")
        except Exception as e:
            print(f"[WARN] Mesh post-processing failed: {e}")

        # 3. IDW k-NN Color Mapping
        from scipy.spatial import cKDTree
        voxel_size = (aabb[1] - aabb[0]) / float(res)
        world_dv_np = ((coords.float() + dv) * voxel_size + aabb[0]).cpu().numpy()
        
        tree = cKDTree(world_dv_np)
        dist, idx = tree.query(verts_np, k=int(color_knn))
        
        if int(color_knn) == 1:
            idx = idx.reshape(-1, 1)
            dist = dist.reshape(-1, 1)
            
        weights = 1.0 / np.maximum(dist, 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        
        rgb_vox_np = rgb_linear_vox.cpu().numpy()
        colors_linear = (rgb_vox_np[idx] * weights[..., None]).sum(axis=1)
        
        # LƯU Ý: O-Voxel converter đã trả về sRGB, KHÔNG cần gamma correction lần nữa.
        # Áp dụng 1/2.2 ở đây sẽ gây hiệu ứng rửa trôi màu (washed-out).
        colors_srgb = np.clip(colors_linear, 0.0, 1.0)
        
        return verts_np, faces_np, colors_srgb
    except Exception as e:
        print(f"[WARN] extract_ovoxel_mesh failed: {e}")
        return None, None, None


def save_comparison_figure(inp_xyz, inp_rgb, rec_xyz, rec_rgb, mse_dict, out_png, title):
    """Save 4-panel comparison: input colored, recon colored, error histogram, metrics."""
    if not HAS_MPL:
        return

    fig = plt.figure(figsize=(20, 5))

    def subsample(xyz, rgb, n=50000):
        if len(xyz) <= n:
            return xyz, rgb
        idx = np.random.choice(len(xyz), size=n, replace=False)
        return xyz[idx], rgb[idx]

    def depth_sort(xyz, rgb):
        """Sort points back-to-front by z (painter's algorithm) for correct overlap."""
        order = np.argsort(xyz[:, 2])
        return xyz[order], rgb[order]

    inp_s, inp_c = subsample(inp_xyz, inp_rgb)
    rec_s, rec_c = subsample(rec_xyz, rec_rgb)
    inp_s, inp_c = depth_sort(inp_s, inp_c)
    rec_s, rec_c = depth_sort(rec_s, rec_c)

    # Input with color
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.scatter(inp_s[:, 0], inp_s[:, 1], inp_s[:, 2], c=inp_c, s=0.5, depthshade=False)
    ax1.set_title("Input (GT)", fontsize=11)
    ax1.set_axis_off()

    # Recon with color
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.scatter(rec_s[:, 0], rec_s[:, 1], rec_s[:, 2], c=rec_c, s=0.5, depthshade=False)
    ax2.set_title("Reconstruction", fontsize=11)
    ax2.set_axis_off()

    # Error histogram (nearest-neighbor distance from recon → GT)
    ax3 = fig.add_subplot(1, 4, 3)
    try:
        from scipy.spatial import cKDTree
        tree_gt = cKDTree(inp_xyz[np.random.choice(len(inp_xyz), min(50000, len(inp_xyz)), replace=False)])
        rec_sample = rec_xyz[np.random.choice(len(rec_xyz), min(50000, len(rec_xyz)), replace=False)]
        err, _ = tree_gt.query(rec_sample)
        ax3.hist(err, bins=50, color="coral", alpha=0.8)
        ax3.set_title("NN Distance (Recon→GT)", fontsize=11)
        ax3.set_xlabel("Distance")
    except Exception:
        # Fallback: paired MSE if arrays happen to match
        n_common = min(len(inp_xyz), len(rec_xyz))
        err = np.mean((inp_xyz[:n_common] - rec_xyz[:n_common]) ** 2, axis=1)
        ax3.hist(err, bins=50, color="coral", alpha=0.8)
        ax3.set_title("Per-point XYZ MSE", fontsize=11)
        ax3.set_xlabel("MSE")

    # Metrics text
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis("off")
    lines = [f"{k}: {v:.6f}" for k, v in mse_dict.items()]
    ax4.text(0.1, 0.5, "\n".join(lines), fontsize=12, family="monospace",
             verticalalignment="center", transform=ax4.transAxes)
    ax4.set_title("Metrics", fontsize=11)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved figure: {out_png}")


def test_one_sample(
    model, cache_path: str, device: torch.device,
    spatial_size: int, out_dir: str, sample_name: str,
    max_voxels: int, mesh_grid: int, mesh_sigma: float, mesh_level: float,
    hotfix: bool = False, reference_obj: str = None, smooth_iters: int = 6, color_knn: int = 8
):
    """Run full SC-VAE reconstruction test on one O-Voxel cache sample."""
    print(f"\n{'='*60}")
    print(f"  Testing: {sample_name}")
    print(f"{'='*60}")

    feats, coords, resolution, aabb = load_cache_sample(cache_path, max_voxels=max_voxels)
    print(f"  Points: {feats.shape[0]:,} | Channels: {feats.shape[1]} | Grid: {resolution}³")

    # Build sparse input and run forward
    sparse_input = build_sparse_input(feats, coords, spatial_size, device)

    t0 = time.time()
    with torch.no_grad():
        x1 = model.enc1(sparse_input)
        x2 = model.enc2(x1)
        x3 = model.enc3(x2)
        x4 = model.enc4(x3)
        mu = model.to_mu(x4.features)
        logvar = model.to_logvar(x4.features)
        z = model.reparameterize(mu, logvar)
        
        # We MUST get the out_indices because spconv does not preserve the input order!
        out = model.decode(
            z,
            original_indices=sparse_input.indices,
            sparse_template=x4,
            sparse_pyramid=None, # DISABLED teacher-forcing to test autonomous generation!
            return_indices=True
        )
        recon = out[0]
        out_indices = out[-1]

    elapsed = time.time() - t0
    print(f"  Forward pass: {elapsed:.2f}s | VRAM: {torch.cuda.max_memory_allocated(device)/1024**2:.0f}MB")

    target = feats.to(device)

    # ALIGN RECON TO TARGET (Crucial fix! spconv output is in Hash order, target is in Z-curve order)
    from src.models.sc_vae import _hash_indices, _SPARSE_HASH_BASE
    tgt_keys = _hash_indices(sparse_input.indices, _SPARSE_HASH_BASE)
    out_keys = _hash_indices(out_indices, _SPARSE_HASH_BASE)

    sorted_tgt_keys, tgt_order = torch.sort(tgt_keys)
    
    pos = torch.searchsorted(sorted_tgt_keys, out_keys)
    safe_pos = torch.clamp(pos, 0, max(sorted_tgt_keys.shape[0] - 1, 0))
    valid = (pos < sorted_tgt_keys.shape[0]) & (sorted_tgt_keys[safe_pos] == out_keys)

    n_recon = recon.shape[0]
    n_target = target.shape[0]
    n_common = int(valid.sum().item())
    print(f"  Recon points: {n_recon:,} | Target points: {n_target:,} | Common: {n_common:,}")

    # Reorder so both are exactly matched by their spatial keys
    recon_aligned = recon[valid]
    
    matching_tgt_idx = tgt_order[safe_pos[valid]]
    target_aligned = target[matching_tgt_idx]
    aligned_coords = sparse_input.indices[matching_tgt_idx, 1:4]

    # Apply activations to reconstruction for proper comparison
    recon_dv = torch.clamp(recon_aligned[:, 0:3], 0.0, 1.0)
    recon_delta = torch.sigmoid(recon_aligned[:, 3:6])
    recon_gamma = torch.nn.functional.softplus(recon_aligned[:, 6:7])
    recon_rgb_act = torch.clamp(recon_aligned[:, 7:10], 0.0, 1.0)
    recon_activated = torch.cat([recon_dv, recon_delta, recon_gamma, recon_rgb_act], dim=1)

    # Compute per-channel MSE (on activated outputs vs targets)
    mse_all = torch.mean((recon_activated - target_aligned) ** 2).item()
    mse_xyz = torch.mean((recon_dv - target_aligned[:, :3]) ** 2).item()
    mse_delta = torch.mean((recon_delta - target_aligned[:, 3:6]) ** 2).item()
    mse_gamma = torch.mean((recon_gamma - target_aligned[:, 6:7]) ** 2).item()
    mse_rgb = torch.mean((recon_rgb_act - target_aligned[:, 7:10]) ** 2).item()

    mse_dict = {
        "mse_all": mse_all,
        "mse_xyz (v)": mse_xyz,
        "mse_delta (flags)": mse_delta,
        "mse_gamma (split)": mse_gamma,
        "mse_rgb (color)": mse_rgb,
        "mu_mean": mu.mean().item(),
        "logvar_mean": logvar.mean().item(),
    }

    for k, v in mse_dict.items():
        print(f"  {k}: {v:.6f}")

    # Compute world XYZ using O-Voxel formula: (coords + dv) * voxel_size + aabb_min
    aligned_coords_f = aligned_coords.float().to(device)
    if aabb is None:
        aabb_t = torch.tensor([[-0.5]*3, [0.5]*3], dtype=torch.float32, device=device)
    elif isinstance(aabb, (list, tuple)):
        aabb_t = torch.tensor(aabb, dtype=torch.float32, device=device)
    elif isinstance(aabb, np.ndarray):
        aabb_t = torch.from_numpy(aabb).float().to(device)
    else:
        aabb_t = aabb.clone().detach().float().to(device)
    voxel_size = (aabb_t[1] - aabb_t[0]) / float(spatial_size)

    recon_world = ((aligned_coords_f + recon_dv) * voxel_size + aabb_t[0]).cpu().numpy()
    recon_rgb_np = recon_rgb_act.cpu().numpy()

    target_dv = target_aligned[:, 0:3].to(device)
    target_world = ((aligned_coords_f + target_dv) * voxel_size + aabb_t[0]).cpu().numpy()
    target_rgb_np = np.clip(target_aligned[:, 7:10].cpu().numpy(), 0.0, 1.0)

    # Export colored point clouds
    sample_dir = os.path.join(out_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    export_colored_ply(target_world, target_rgb_np, os.path.join(sample_dir, "input_colored.ply"))
    print(f"  Exported: input_colored.ply ({n_common:,} intersecting points)")

    export_colored_ply(recon_world, recon_rgb_np, os.path.join(sample_dir, "recon_aligned.ply"))
    print(f"  Exported: recon_aligned.ply ({n_common:,} intersecting points)")

    # Export RAW autonomous geometry (including False Positives/Negatives)
    raw_coords_f = out_indices[:, 1:4].float().to(device)
    raw_recon_all = recon.detach()
    VOXEL_MARGIN = 0.5
    raw_dv = (1 + 2 * VOXEL_MARGIN) * torch.sigmoid(raw_recon_all[:, 0:3]) - VOXEL_MARGIN
    raw_world = ((raw_coords_f + raw_dv) * voxel_size + aabb_t[0]).cpu().numpy()
    raw_rgb = np.clip(raw_recon_all[:, 7:10].cpu().numpy(), 0.0, 1.0)
    export_colored_ply(raw_world, raw_rgb, os.path.join(sample_dir, "recon_raw_autonomous.ply"))
    print(f"  Exported: recon_raw_autonomous.ply ({len(raw_world):,} generated points)")

    # Export FULL ground truth (all target points, no intersection filtering)
    all_tgt_coords = sparse_input.indices[:, 1:4].float().to(device)
    all_tgt_dv = target[:, 0:3].to(device)
    full_gt_world = ((all_tgt_coords + all_tgt_dv) * voxel_size + aabb_t[0]).cpu().numpy()
    full_gt_rgb = np.clip(target[:, 7:10].cpu().numpy(), 0.0, 1.0)
    export_colored_ply(full_gt_world, full_gt_rgb, os.path.join(sample_dir, "full_groundtruth.ply"))
    print(f"  Exported: full_groundtruth.ply ({len(full_gt_world):,} points — complete GT)")

    # Topology accuracy metrics
    n_false_negative = n_target - n_common  # GT points missing from reconstruction
    n_false_positive = n_recon - n_common   # Recon points not in GT
    topo_recall = n_common / max(n_target, 1) * 100
    topo_precision = n_common / max(n_recon, 1) * 100
    print(f"  Topology: recall={topo_recall:.1f}% precision={topo_precision:.1f}% "
          f"(FN={n_false_negative:,} FP={n_false_positive:,})")
    mse_dict["topo_recall"] = topo_recall
    mse_dict["topo_precision"] = topo_precision

    # Export colored mesh via proper O-Voxel Dual Contouring
    # We automatically try to find a reference OBJ in the original dataset directory
    ref_obj = None
    if reference_obj:
        ref_obj = reference_obj
    else:
        # Heuristic: find original obj in FaceVerse/FaceScape dataset paths
        identity = sample_name.split('.')[0] # e.g. 011_01
        potential_paths = [
            f"/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/{identity}/{identity}.obj",
            f"/mnt/16TData/Datasets/FaceScape/{identity.split('_')[0]}/models_reg/{identity.split('_')[-1]}.obj"
        ]
        for p in potential_paths:
            if os.path.exists(p):
                ref_obj = p
                print(f"  Found reference mesh: {ref_obj}")
                break
    
    target_faces = 0
    if ref_obj:
        try:
            import trimesh
            ref_mesh = trimesh.load(ref_obj, process=False)
            if isinstance(ref_mesh, trimesh.Scene):
                ref_mesh = ref_mesh.dump(concatenate=True)
            target_faces = len(ref_mesh.faces)
        except Exception:
            pass

    verts, faces, colors = extract_ovoxel_mesh(
        out_indices[:, 1:4], recon, aabb, resolution, 
        is_logits=True, threshold=mesh_level,
        target_faces=target_faces, 
        smooth_iters=smooth_iters, 
        color_knn=color_knn
    )
    if verts is not None and faces is not None:
        mesh_path = os.path.join(sample_dir, "recon_mesh_colored.ply")
        export_colored_mesh_ply(verts, faces, colors, mesh_path)
        print(f"  Exported: recon_mesh_colored.ply ({len(verts):,} verts, {len(faces):,} faces)")
    else:
        print("  [WARN] DC Mesh extraction failed")
    
    # Poisson Surface Reconstruction — watertight mesh từ point cloud
    poisson_verts, poisson_faces, poisson_colors = extract_poisson_mesh(
        raw_world, raw_rgb, poisson_depth=9, density_quantile=0.01,
        smooth_iters=3, target_faces=target_faces
    )
    if poisson_verts is not None:
        poisson_path = os.path.join(sample_dir, "recon_poisson_colored.ply")
        export_colored_mesh_ply(poisson_verts, poisson_faces, poisson_colors, poisson_path)
        print(f"  Exported: recon_poisson_colored.ply ({len(poisson_verts):,} verts, {len(poisson_faces):,} faces)")

    # Export GT mesh (intersection-aligned, for paired comparison)
    gt_verts, gt_faces, gt_colors = extract_ovoxel_mesh(
        aligned_coords, target_aligned, aabb, resolution, is_logits=False
    )
    if gt_verts is not None:
        gt_mesh_path = os.path.join(sample_dir, "input_mesh_colored.ply")
        export_colored_mesh_ply(gt_verts, gt_faces, gt_colors, gt_mesh_path)
        print(f"  Exported: input_mesh_colored.ply ({len(gt_verts):,} verts, {len(gt_faces):,} faces)")

    # Export FULL GT mesh (all voxels, no intersection filtering)
    full_gt_verts, full_gt_faces, full_gt_colors = extract_ovoxel_mesh(
        sparse_input.indices[:, 1:4], target, aabb, resolution, is_logits=False
    )
    if full_gt_verts is not None:
        full_gt_mesh_path = os.path.join(sample_dir, "full_gt_mesh_colored.ply")
        export_colored_mesh_ply(full_gt_verts, full_gt_faces, full_gt_colors, full_gt_mesh_path)
        print(f"  Exported: full_gt_mesh_colored.ply ({len(full_gt_verts):,} verts, {len(full_gt_faces):,} faces)")

    # ---- TRELLIS.2 Metrics (Section D.1.1) ----
    from src.scvae_train.metrics import (
        compute_f_score,
        compute_mesh_distance,
        compute_normal_psnr_lpips,
    )

    # F-score (point cloud level, τ=1e-6 per TRELLIS.2)
    recon_xyz_t = torch.from_numpy(recon_world).float()
    target_xyz_t = torch.from_numpy(full_gt_world).float()
    fscore_dict = compute_f_score(recon_xyz_t, target_xyz_t, tau=1e-6)
    mse_dict["f_score"] = fscore_dict["f_score"]
    mse_dict["f_precision"] = fscore_dict["precision"]
    mse_dict["f_recall"] = fscore_dict["recall"]
    print(f"  F-score: {fscore_dict['f_score']:.4f} (prec={fscore_dict['precision']:.4f}, rec={fscore_dict['recall']:.4f})")

    # Mesh Distance + Normal PSNR/LPIPS (requires extracted meshes)
    if verts is not None and full_gt_verts is not None:
        # Mesh Distance (bidirectional point-to-mesh)
        md = compute_mesh_distance(verts, faces, full_gt_verts, full_gt_faces, n_samples=100_000)
        mse_dict["mesh_distance"] = md
        print(f"  Mesh Distance: {md:.8f}")

        # Normal PSNR + LPIPS (4 views, TRELLIS.2 protocol)
        try:
            npl = compute_normal_psnr_lpips(
                verts, faces, full_gt_verts, full_gt_faces,
                image_size=512, device=str(device),
            )
            mse_dict["normal_psnr"] = npl["normal_psnr"]
            mse_dict["normal_lpips"] = npl["normal_lpips"]
            print(f"  Normal PSNR: {npl['normal_psnr']:.2f} dB | Normal LPIPS: {npl['normal_lpips']:.4f}")
        except Exception as e:
            print(f"  [WARN] Normal PSNR/LPIPS computation failed: {e}")

    # Save comparison figure
    # Use FULL GT (all voxels) for the comparison figure instead of intersection-only
    save_comparison_figure(
        full_gt_world, full_gt_rgb, recon_world, recon_rgb_np,
        mse_dict,
        os.path.join(sample_dir, "comparison.png"),
        f"SC-VAE Recon: {sample_name}",
    )

    # Save metrics
    with open(os.path.join(sample_dir, "metrics.txt"), "w") as f:
        f.write(f"cache_file: {cache_path}\n")
        f.write(f"n_target: {n_target}\n")
        f.write(f"n_recon: {n_recon}\n")
        for k, v in mse_dict.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

    return mse_dict


def main():
    parser = argparse.ArgumentParser("SC-VAE Reconstruction Test v2")
    parser.add_argument("--ckpt", default="checkpoints/sc_vae_shape/epoch_200.pt")
    parser.add_argument("--cache-dir", default="data/voxel_cache/ovoxel_cache_recached/faceverse")
    parser.add_argument("--output-dir", default="outputs/sc_vae_recon_v2")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of random samples to test")
    parser.add_argument("--max-voxels", type=int, default=0, help="Max voxels per sample (0=all)")
    parser.add_argument("--dc-grid", type=int, default=256, help="Dual Contouring grid resolution")
    parser.add_argument("--dc-level", type=float, default=0.5, help="Dual Contouring delta threshold (sigmoid output in [0,1])")
    parser.add_argument("--rho-threshold", type=float, default=0.5, help="Pruning threshold for SC-VAE (0.0=no pruning)")
    parser.add_argument("--hotfix", action="store_true", help="Apply quality hotfixes (dilation + gamma smoothing)")
    parser.add_argument("--reference-obj", default=None, help="Path to original .obj for face count matching")
    parser.add_argument("--smooth-iters", type=int, default=6, help="Number of Taubin smoothing iterations")
    parser.add_argument("--color-knn", type=int, default=8, help="k-NN neighbors for color interpolation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = TrainConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    spatial_size = int(cfg.data.voxel_resolution)

    # Find cache files
    cache_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    if not cache_files:
        raise FileNotFoundError(f"No .pt cache files found in {args.cache_dir}")
    print(f"Found {len(cache_files)} cached samples in {args.cache_dir}")

    # Select random samples
    n = min(args.num_samples, len(cache_files))
    selected = random.sample(cache_files, n)

    # Load model
    print(f"\nLoading SC-VAE from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SC_VAE(
        in_channels=cfg.sc_vae.in_channels,
        latent_dim=cfg.sc_vae.latent_dim,
        device=str(device),
        rho_prune_threshold=args.rho_threshold,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"  Model loaded: in_channels={cfg.sc_vae.in_channels}, latent_dim={cfg.sc_vae.latent_dim}")
    print(f"  VRAM after load: {torch.cuda.memory_allocated(device)/1024**2:.0f}MB")

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    for cache_path in selected:
        sample_name = Path(cache_path).stem.split('.mx')[0]
        mse = test_one_sample(
            model, cache_path, device, spatial_size, args.output_dir, sample_name,
            max_voxels=args.max_voxels,
            mesh_grid=args.dc_grid,
            mesh_sigma=0.5,
            mesh_level=args.dc_level,
            hotfix=args.hotfix,
            reference_obj=args.reference_obj,
            smooth_iters=args.smooth_iters,
            color_knn=args.color_knn
        )
        all_metrics.append(mse)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({n} samples)")
    print(f"{'='*60}")
    summary_keys = [
        "mse_all", "mse_xyz (v)", "mse_rgb (color)",
        "f_score", "f_precision", "f_recall",
        "mesh_distance", "normal_psnr", "normal_lpips",
        "topo_recall", "topo_precision",
    ]
    for key in summary_keys:
        vals = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
        if vals:
            print(f"  {key}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
