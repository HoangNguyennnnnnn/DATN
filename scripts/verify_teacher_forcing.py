"""
Verify GT mesh quality by checking face/vert ratio and topology.
Also test: what if we DON'T prune in the decoder (use teacher forcing)?
"""
import os
import sys
import torch
import numpy as np
import trimesh
import lmdb
import io

sys.path.append(os.getcwd())
ovoxel_path = "/mnt/18TData/facediff/third_party/TRELLIS.2/o-voxel"
if ovoxel_path not in sys.path:
    sys.path.append(ovoxel_path)

from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
import spconv.pytorch as spconv
from src.models.sc_vae import SC_VAE


def decode_to_mesh(coords, features, aabb, res, tag=""):
    dv_local = features[:, 0:3].clamp(0.0, 1.0)
    flag_bool = (features[:, 3:6] > 0.5)
    
    verts, faces = flexible_dual_grid_to_mesh(
        coords.to(torch.int32), dv_local, flag_bool,
        split_weight=None, grid_size=res, aabb=aabb,
    )
    
    from scipy.spatial import cKDTree
    rgb = features[:, 7:10].clamp(0.0, 1.0)
    voxel_size = (aabb[1] - aabb[0]) / res
    dual_world = (coords.float() + dv_local) * voxel_size + aabb[0]
    tree = cKDTree(dual_world.cpu().numpy())
    _, idx = tree.query(verts.cpu().numpy(), k=1)
    colors = (rgb.cpu().numpy()[idx] * 255).astype(np.uint8)
    
    print(f"  [{tag}] {verts.shape[0]} verts, {faces.shape[0]} faces, ratio={faces.shape[0]/max(verts.shape[0],1):.2f}")
    return verts.cpu().numpy(), faces.cpu().numpy(), colors


def test_scvae_with_teacher_forcing(ckpt_path, epoch_label):
    """
    Run SC-VAE with TEACHER FORCING (use the encoder pyramid to guide decoder).
    This should produce much better topology because the decoder knows exactly
    which voxels to keep at each level.
    """
    device = torch.device("cuda")
    
    model = SC_VAE(in_channels=10, latent_dim=32, device="cuda").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    lmdb_path = "/mnt/18TData/facediff/data/ovoxel_cache_lmdb"
    env = lmdb.open(lmdb_path, readonly=True, lock=False, map_size=400 * 1024 * 1024 * 1024)
    out_dir = "outputs/verification/e2e_v14"
    os.makedirs(out_dir, exist_ok=True)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        k, v = cursor.item()
        name = k.decode()
        
        data = torch.load(io.BytesIO(v), map_location=device, weights_only=False)
        features = data['features'].to(device)
        coords = data['coords'].to(device)
        aabb = data['aabb'].to(device)
        res = data.get('resolution', 256)
        
        print(f"\n=== Teacher-Forcing test: {name} with {epoch_label} ===")
        
        spatial_shape = [res, res, res]
        indices = torch.cat([
            torch.zeros((coords.shape[0], 1), device=device, dtype=torch.int32),
            coords.to(torch.int32)
        ], dim=1)
        x_sparse = spconv.SparseConvTensor(features, indices, spatial_shape, 1)
        
        with torch.no_grad():
            # Full forward pass WITH teacher forcing (training mode topology)
            recon, mu, logvar, rho_logits_list, rho_targets_list, out_indices = model(x_sparse)
        
        print(f"  Input: {coords.shape[0]} voxels → Output: {out_indices.shape[0]} voxels")
        print(f"  Recon shape: {recon.shape}")
        
        recon_coords = out_indices[:, 1:]
        
        # Decode with teacher-forced topology
        verts, faces, colors = decode_to_mesh(
            recon_coords, recon, aabb, res, tag=f"TF-{epoch_label}"
        )
        
        out_path = os.path.join(out_dir, f"{name.split('.')[0]}_TF_{epoch_label}.obj")
        trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False).export(out_path)
        print(f"  Saved: {out_path}")
    
    env.close()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test with teacher forcing for both checkpoints
    for epoch, path in [("390", "checkpoints/sc_vae_shape/epoch_390.pt"),
                         ("400", "checkpoints/sc_vae_shape/epoch_400.pt")]:
        if os.path.exists(path):
            test_scvae_with_teacher_forcing(path, epoch)
