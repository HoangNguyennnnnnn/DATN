"""Helper decode cho E2E inference: slat_to_mesh (slat → SC-VAE → DC mesh) + save_ply.

Dùng bởi `test_e2e_inference_unet.py` (backbone hiện tại = VoxelUNet3D).
(Phần test/sample VoxelMamba cũ đã bỏ khi deprecate Mamba — xem git history nếu cần.)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.sc_vae import SC_VAE


def slat_to_mesh(
    slat_raw: torch.Tensor,  # [1, L, 32] in SC-VAE latent space (post-unnormalize)
    sc_vae: SC_VAE,
    decode_device: torch.device,
    slat_grid_size: int = 16,
    ovoxel_resolution: int = 256,
    mask: torch.Tensor | None = None,
):
    """Decode slat → SC-VAE → DC mesh (verts, faces, rgb)."""
    b, L, D = slat_raw.shape
    assert b == 1, "batch=1 only"
    slat_raw = slat_raw.to(decode_device)

    # Build grid_indices for 16³ slat grid (mapping latent token → spatial position)
    coords_1d = torch.arange(slat_grid_size, device=decode_device)
    gz, gy, gx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    grid_indices = torch.stack(
        [torch.zeros_like(gx.flatten()), gz.flatten(), gy.flatten(), gx.flatten()],
        dim=1,
    ).int()  # [L, 4] với cột 0 = batch_id

    z_flat = slat_raw[0].contiguous()  # [L, D]

    # [BUG FIX]: Lọc bỏ các token rỗng (do UNet3D dense padding) trước khi đưa vào decode
    # Nếu đưa toàn bộ grid_indices vào, SC_VAE sẽ bị OOM vì cố gắng decode vùng không gian rỗng (sponge).
    if mask is None:
        mask = (z_flat.norm(dim=-1) > 0.1)
    else:
        mask = mask.to(decode_device)

    valid_z = z_flat[mask]
    valid_indices = grid_indices[mask]

    print(f"      -> Valid tokens (num_voxels): {mask.sum().item()} / 4096")

    if valid_z.shape[0] == 0:
        print("[WARN] UNet3D sinh ra rỗng hoàn toàn! Bỏ qua mask.")
        valid_z = z_flat
        valid_indices = grid_indices

    voxel_feats, _, _, out_indices = sc_vae.decode(
        valid_z,
        original_indices=valid_indices,
        batch_size=1,
        return_indices=True,
    )

    # Extract mesh via DC (sử dụng helper từ test_sc_vae_recon_v2)
    from scripts.test.test_sc_vae_recon_v2 import extract_ovoxel_mesh

    coords_dec = out_indices[:, 1:].int()  # drop batch column, must be int32 for o_voxel DC
    aabb = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    verts, faces, colors = extract_ovoxel_mesh(
        coords=coords_dec,
        feats=voxel_feats.float(),
        aabb=aabb,
        res=ovoxel_resolution,
        is_logits=True,  # SC-VAE outputs raw logits
        threshold=0.5,
        target_faces=0,  # no remesh
        smooth_iters=2,
        color_knn=8,
    )
    return verts, faces, colors, voxel_feats.shape[0]


def save_ply(verts, faces, colors, path):
    """Save .ply file with vertex colors."""
    if verts is None or len(verts) == 0:
        print(f"  [WARN] Empty mesh, skipping save: {path}")
        return False

    n_v = len(verts)
    n_f = len(faces) if faces is not None else 0
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_v}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {n_f}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n_v):
            v = verts[i]
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if colors is not None:
                c = colors[i]
                line += f" {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}"
            f.write(line + "\n")
        if n_f > 0:
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return True
