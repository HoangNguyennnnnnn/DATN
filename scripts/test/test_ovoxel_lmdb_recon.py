import torch
import lmdb
import io
import sys
import os
import spconv.pytorch as spconv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.sc_vae import SC_VAE
from src.config import TrainConfig
from scripts.test.test_sc_vae_recon_v2 import extract_ovoxel_mesh
from scripts.test.test_e2e_inference import save_ply

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SC-VAE
    cfg = TrainConfig()
    sc_vae = SC_VAE(
        in_channels=int(cfg.sc_vae.in_channels),
        latent_dim=int(cfg.sc_vae.latent_dim),
        num_res_blocks=int(cfg.sc_vae.num_res_blocks),
        encoder_dims=list(cfg.sc_vae.encoder_dims),
    ).to(device)
    
    ckpt_path = "checkpoints/sc_vae_shape/epoch_500.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    sc_vae.load_state_dict(state, strict=False)
    sc_vae.eval()
    print("SC-VAE loaded.")

    # Open ovoxel_cache_lmdb
    env = lmdb.open("data/ovoxel_cache_lmdb", readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            if k.decode() != "__meta__":
                key_str = k.decode()
                payload = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
                break
    
    print(f"Loaded {key_str} from data/ovoxel_cache_lmdb")
    coords = payload["coords"].int()
    feats = payload["features"].float()
    
    if feats.shape[1] > 10:
        feats = feats[:, :10]

    bcol = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    sparse_indices = torch.cat([bcol, coords], dim=1).to(device)
    sparse_input = spconv.SparseConvTensor(
        features=feats.to(device),
        indices=sparse_indices,
        spatial_shape=[256, 256, 256],
        batch_size=1,
    )

    print("\n--- 1. ENCODING (OVoxel -> Slat) ---")
    with torch.no_grad():
        x = sc_vae.enc1(sparse_input)
        x = sc_vae.enc2(x)
        x = sc_vae.enc3(x)
        x4 = sc_vae.enc4(x)
        mu = sc_vae.to_mu(x4.features)
        print(f"Encoded Slat shape: {mu.shape}")

    print("\n--- 2. DECODING (Slat -> OVoxel) ---")
    with torch.no_grad():
        voxel_feats, _, _, out_indices = sc_vae.decode(
            mu,
            original_indices=sparse_input.indices,
            sparse_template=x4,
            batch_size=1,
            return_indices=True,
        )
        print(f"Decoded voxel feats shape: {voxel_feats.shape}")

    print("\n--- 3. EXTRACTING MESH (OVoxel -> Mesh) ---")
    coords_dec = out_indices[:, 1:].int()
    aabb = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    verts, faces, colors = extract_ovoxel_mesh(
        coords=coords_dec,
        feats=voxel_feats.float(),
        aabb=aabb,
        res=256,
        is_logits=True,
        threshold=0.5,
        target_faces=0,
        smooth_iters=2,
        color_knn=8,
    )
    
    if verts is not None:
        os.makedirs("outputs_pipeline_test", exist_ok=True)
        out_path = "outputs_pipeline_test/ovoxel_lmdb_recon.ply"
        save_ply(verts, faces, colors, out_path)
        print(f"Mesh saved successfully to {out_path} with {len(verts)} vertices!")
    else:
        print("Failed to extract mesh.")

if __name__ == "__main__":
    main()
