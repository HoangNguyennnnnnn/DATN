"""
End-to-End Pipeline Verification (v14 — Forensic Correct)
==========================================================
This script follows the EXACT algorithm from:
  https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel

Flow: Mesh → O-Voxel → SC-VAE encode → SC-VAE decode → O-Voxel → Dual Contouring → Mesh

CRITICAL FINDINGS from LMDB audit:
  - Ch[0:3] = dv_local = (dv_raw * RES - coords).clamp(0,1)  ← PACKED fractional offset
  - Ch[3:5] = intersected_flag as 3 separate bools {0, 1}
  - Ch[6]   = gamma (always 1.0 in cache)
  - Ch[7:9] = RGB in [0, 1]
  - AABB    = real mesh AABB (NOT normalized to [-0.5, 0.5])

Microsoft decode formula (flexible_dual_grid.py line 262):
  mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]

Where:
  dual_vertices = dv_local (the [0,1] fractional offset)
  voxel_size = (aabb[1] - aabb[0]) / grid_size
"""
import os
import sys
import torch
import numpy as np
import trimesh
import lmdb
import io

sys.path.append(os.getcwd())

# Add o_voxel path
ovoxel_path = "/mnt/18TData/facediff/third_party/TRELLIS.2/o-voxel"
if ovoxel_path not in sys.path:
    sys.path.append(ovoxel_path)

from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
import spconv.pytorch as spconv
from src.models.sc_vae import SC_VAE


def decode_ovoxel_to_mesh(coords, features, aabb, resolution, tag=""):
    """
    Decode 10-channel O-Voxel features to mesh using Microsoft's exact algorithm.
    
    Features layout: [dv_local(3), flag(3), gamma(1), rgb(3)]
    """
    device = coords.device
    
    # 1. Depack features
    dv_local = features[:, 0:3].clamp(0.0, 1.0)    # fractional offset [0, 1]
    flag_bool = (features[:, 3:6] > 0.5)             # 3-channel bool
    # gamma (Ch6) — not used by flexible_dual_grid_to_mesh when split_weight=None
    rgb = features[:, 7:10].clamp(0.0, 1.0)
    
    # 2. Run Microsoft's Dual Contouring
    # flexible_dual_grid_to_mesh internally computes:
    #   mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]
    # where dual_vertices = dv_local (our Ch0-2)
    verts, faces = flexible_dual_grid_to_mesh(
        coords.to(torch.int32),
        dv_local,
        flag_bool,
        split_weight=None,  # Auto-split by min angle (most robust)
        grid_size=resolution,
        aabb=aabb,
    )
    
    print(f"  [{tag}] DC output: {verts.shape[0]} verts, {faces.shape[0]} faces")
    
    # 3. Vertex coloring via nearest-neighbor lookup
    from scipy.spatial import cKDTree
    voxel_size = (aabb[1] - aabb[0]) / resolution
    dual_world = (coords.float() + dv_local) * voxel_size + aabb[0]
    
    tree = cKDTree(dual_world.cpu().numpy())
    _, idx = tree.query(verts.cpu().numpy(), k=1)
    vert_colors = (rgb.cpu().numpy()[idx] * 255).astype(np.uint8)
    
    return verts.cpu().numpy(), faces.cpu().numpy(), vert_colors


def run_e2e_from_lmdb(ckpt_path, epoch_label, num_samples=1):
    """Load from LMDB, pass through SC-VAE, decode back."""
    device = torch.device("cuda")
    
    # 1. Load SC-VAE
    print(f"\n=== Loading SC-VAE from {ckpt_path} ===")
    model = SC_VAE(in_channels=10, latent_dim=32, device="cuda").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 2. Open LMDB
    lmdb_path = "/mnt/18TData/facediff/data/ovoxel_cache_lmdb"
    env = lmdb.open(lmdb_path, readonly=True, lock=False, map_size=400 * 1024 * 1024 * 1024)
    
    out_dir = "outputs/verification/e2e_v14"
    os.makedirs(out_dir, exist_ok=True)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        
        for i in range(num_samples):
            k, v = cursor.item()
            name = k.decode()
            print(f"\n--- Sample {i+1}: {name} ---")
            
            data = torch.load(io.BytesIO(v), map_location=device, weights_only=False)
            features = data['features'].to(device)
            coords = data['coords'].to(device)
            aabb = data['aabb'].to(device)
            res = data.get('resolution', 256)
            
            # === STEP A: Direct decode (bypass SC-VAE) for ground truth ===
            verts_gt, faces_gt, colors_gt = decode_ovoxel_to_mesh(
                coords, features, aabb, res, tag="GT"
            )
            gt_path = os.path.join(out_dir, f"{name.split('.')[0]}_GT.obj")
            trimesh.Trimesh(vertices=verts_gt, faces=faces_gt, vertex_colors=colors_gt, process=False).export(gt_path)
            print(f"  Saved GT: {gt_path}")
            
            # === STEP B: Through SC-VAE ===
            # Build sparse tensor for spconv
            spatial_shape = [res, res, res]
            indices = torch.cat([
                torch.zeros((coords.shape[0], 1), device=device, dtype=torch.int32),
                coords.to(torch.int32)
            ], dim=1)
            x_sparse = spconv.SparseConvTensor(features, indices, spatial_shape, 1)
            
            with torch.no_grad():
                # Encode
                x1 = model.enc1(x_sparse)
                x2 = model.enc2(x1)
                x3 = model.enc3(x2)
                x4 = model.enc4(x3)
                
                mu = model.to_mu(x4.features)
                mu = torch.clamp(mu, -5.0, 5.0)
                
                # Decode (inference mode — auto pruning)
                recon_feat, _, _, out_indices = model.decode(
                    mu,
                    original_indices=x4.indices,
                    batch_size=1,
                    return_indices=True,
                )
            
            # out_indices is [N, 4] (batch, x, y, z) — strip batch dim
            recon_coords = out_indices[:, 1:].to(device)
            recon_features = recon_feat.to(device)
            
            print(f"  SC-VAE: {coords.shape[0]} input voxels → {recon_coords.shape[0]} output voxels")
            
            # === STEP C: Decode SC-VAE output back to mesh ===
            verts_sc, faces_sc, colors_sc = decode_ovoxel_to_mesh(
                recon_coords, recon_features, aabb, res, tag=f"SCVAE-{epoch_label}"
            )
            sc_path = os.path.join(out_dir, f"{name.split('.')[0]}_SCVAE_{epoch_label}.obj")
            trimesh.Trimesh(vertices=verts_sc, faces=faces_sc, vertex_colors=colors_sc, process=False).export(sc_path)
            print(f"  Saved SC-VAE: {sc_path}")
            
            if not cursor.next():
                break
    
    env.close()
    del model
    torch.cuda.empty_cache()


def run_e2e_from_mesh(obj_path, ckpt_path, epoch_label, dataset_name):
    """Convert raw mesh → O-Voxel → SC-VAE → mesh."""
    device = torch.device("cuda")
    
    print(f"\n=== Fresh mesh: {obj_path} ===")
    
    # 1. Mesh → O-Voxel using Microsoft's library directly (no custom converter wrapper)
    import trimesh as tm
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
    from o_voxel.convert.volumetic_attr import textured_mesh_to_volumetric_attr
    from o_voxel.serialize import encode_seq
    from trimesh.visual.material import PBRMaterial
    
    asset = tm.load(obj_path, process=False)
    mesh = asset.to_mesh() if isinstance(asset, tm.Scene) else asset
    
    # PBR-ify for texture sampling
    if not isinstance(mesh.visual.material, PBRMaterial):
        image = getattr(mesh.visual.material, 'image', getattr(mesh.visual.material, 'diffuse', None))
        mesh.visual.material = PBRMaterial(
            baseColorFactor=[255, 255, 255, 255],
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    
    RES = 256
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    
    # Normalize mesh to [-0.5, 0.5]
    verts_t = torch.from_numpy(mesh.vertices).float()
    vmin, vmax = verts_t.min(0)[0], verts_t.max(0)[0]
    center = (vmin + vmax) / 2
    scale = 0.95 / (vmax - vmin).max().clamp_min(1e-8)
    mesh.vertices = ((verts_t - center) * scale).numpy()
    
    # Geometry voxelization — EXACTLY as Microsoft README
    res_geo = mesh_to_flexible_dual_grid(
        torch.from_numpy(mesh.vertices).float(),
        torch.from_numpy(mesh.faces).long(),
        grid_size=RES, aabb=aabb, regularization_weight=1e-2,
    )
    coords_geo = res_geo['coords']
    dv_geo = res_geo['dual_vertices']
    flag_geo = res_geo['intersected_flag']
    
    # Sort by Morton code — as per README
    vid = encode_seq(coords_geo.cuda()).cpu()
    order = torch.argsort(vid)
    coords_geo = coords_geo[order]
    dv_geo = dv_geo[order]
    flag_geo = flag_geo[order]
    
    # Material voxelization
    coords_mat, attrs = textured_mesh_to_volumetric_attr(mesh, grid_size=RES, aabb=aabb)
    vid_mat = encode_seq(coords_mat.cuda()).cpu()
    order_mat = torch.argsort(vid_mat)
    coords_mat = coords_mat[order_mat]
    base_color = attrs['base_color'][order_mat]
    
    # Align geo & mat voxels
    common = np.intersect1d(
        encode_seq(coords_geo.cuda()).cpu().numpy(),
        encode_seq(coords_mat.cuda()).cpu().numpy(),
    )
    vid_geo_np = encode_seq(coords_geo.cuda()).cpu().numpy()
    vid_mat_np = encode_seq(coords_mat.cuda()).cpu().numpy()
    mask_g = np.isin(vid_geo_np, common)
    mask_m = np.isin(vid_mat_np, common)
    
    coords = coords_geo[mask_g]
    dv = dv_geo[mask_g]
    flag = flag_geo[mask_g]
    color = base_color[mask_m]
    
    # Pack — EXACTLY as Microsoft README line 69-73:
    #   dual_vertices = dual_vertices * RES - voxel_indices   ← [0, 1]
    #   intersected stored as 3-channel bool
    #   rgb as [0, 1] float
    dv_local = (dv * RES - coords.float()).clamp(0.0, 1.0)
    flag_float = flag.float()
    gamma = torch.ones(coords.shape[0], 1)  # constant 1.0 as in LMDB
    color_float = color.float() / 255.0
    
    features = torch.cat([dv_local, flag_float, gamma, color_float], dim=-1)  # [N, 10]
    
    print(f"  Fresh O-Voxel: {features.shape[0]} voxels, {features.shape[1]} channels")
    
    # Save GT mesh from fresh O-Voxel
    out_dir = "outputs/verification/e2e_v14"
    os.makedirs(out_dir, exist_ok=True)
    
    verts_gt, faces_gt, colors_gt = decode_ovoxel_to_mesh(
        coords.cuda(), features.cuda(), aabb.cuda(), RES, tag="FreshGT"
    )
    gt_path = os.path.join(out_dir, f"{dataset_name}_fresh_GT.obj")
    tm.Trimesh(vertices=verts_gt, faces=faces_gt, vertex_colors=colors_gt, process=False).export(gt_path)
    print(f"  Saved: {gt_path}")
    
    # 2. Through SC-VAE
    model = SC_VAE(in_channels=10, latent_dim=32, device="cuda").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    spatial_shape = [RES, RES, RES]
    indices = torch.cat([
        torch.zeros((coords.shape[0], 1), dtype=torch.int32),
        coords.to(torch.int32),
    ], dim=1).cuda()
    x_sparse = spconv.SparseConvTensor(features.cuda(), indices, spatial_shape, 1)
    
    with torch.no_grad():
        x1 = model.enc1(x_sparse)
        x2 = model.enc2(x1)
        x3 = model.enc3(x2)
        x4 = model.enc4(x3)
        mu = torch.clamp(model.to_mu(x4.features), -5.0, 5.0)
        recon_feat, _, _, out_indices = model.decode(
            mu, original_indices=x4.indices, batch_size=1, return_indices=True
        )
    
    recon_coords = out_indices[:, 1:]
    print(f"  SC-VAE: {coords.shape[0]} → {recon_coords.shape[0]} voxels")
    
    verts_sc, faces_sc, colors_sc = decode_ovoxel_to_mesh(
        recon_coords, recon_feat, aabb.cuda(), RES, tag=f"SCVAE-{epoch_label}"
    )
    sc_path = os.path.join(out_dir, f"{dataset_name}_SCVAE_{epoch_label}.obj")
    tm.Trimesh(vertices=verts_sc, faces=faces_sc, vertex_colors=colors_sc, process=False).export(sc_path)
    print(f"  Saved: {sc_path}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Part 1: From LMDB (as-is cached data)
    for epoch, path in [("390", "checkpoints/sc_vae_shape/epoch_390.pt"),
                         ("400", "checkpoints/sc_vae_shape/epoch_400.pt")]:
        if os.path.exists(path):
            run_e2e_from_lmdb(path, epoch, num_samples=1)
    
    # Part 2: From fresh mesh
    fs_sample = "/mnt/16TData/Datasets/FaceScape/436/models_reg/1_neutral.obj"
    fv_sample = "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/064_03/064_03.obj"
    
    for epoch, ckpt in [("390", "checkpoints/sc_vae_shape/epoch_390.pt"),
                         ("400", "checkpoints/sc_vae_shape/epoch_400.pt")]:
        if not os.path.exists(ckpt):
            continue
        run_e2e_from_mesh(fs_sample, ckpt, epoch, "facescape")
        run_e2e_from_mesh(fv_sample, ckpt, epoch, "faceverse")
