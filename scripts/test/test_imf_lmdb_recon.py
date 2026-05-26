import torch
import lmdb
import io
import sys
import os
import spconv.pytorch as spconv
import numpy as np
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.sc_vae import SC_VAE
from src.models.voxel_mamba import VoxelMamba
from src.models.imf_diffusion import ImprovedMeanFlow
from src.config import TrainConfig
from scripts.test.test_sc_vae_recon_v2 import extract_ovoxel_mesh, extract_poisson_mesh
from scripts.test.test_e2e_inference import save_ply

def load_models(device):
    cfg = TrainConfig()
    
    # 1. Load SC-VAE
    print("[Load] Initializing SC-VAE...")
    sc_vae = SC_VAE(
        in_channels=int(cfg.sc_vae.in_channels),
        latent_dim=int(cfg.sc_vae.latent_dim),
        num_res_blocks=int(cfg.sc_vae.num_res_blocks),
        encoder_dims=list(cfg.sc_vae.encoder_dims),
    ).to(device)
    
    ckpt_vae = torch.load("checkpoints/sc_vae_shape/epoch_500.pt", map_location="cpu", weights_only=False)
    state_vae = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt_vae.get("model_state_dict", ckpt_vae).items()}
    sc_vae.load_state_dict(state_vae, strict=False)
    sc_vae.eval()
    print("[Load] SC-VAE loaded successfully.")
    
    # 2. Load iMF Mamba
    print("[Load] Initializing VoxelMamba (iMF)...")
    model_cfg = {
        "arch": "voxel_mamba",
        "input_dim": 32,
        "context_dim": 946,
        "slat_length": 4096,
        "slat_stats_path": None,
        "hidden_dim": 512,
        "num_layers": 8,
        "backend": "auto",
        "strict": False,
        "num_context_tokens": 8,
        "num_time_tokens": 4,
        "num_r_tokens": 4,
        "num_interval_tokens": 4,
        "num_guidance_tokens": 4,
        "use_per_layer_context": False,
        "context_cond_mode": "adaln",
        "context_use_arcface_only": False,
        "num_context_kv_tokens": 8,
        "context_cross_attn_heads": 8,
    }
    imf = VoxelMamba(
        input_dim=model_cfg["input_dim"],
        context_dim=model_cfg["context_dim"],
        slat_length=model_cfg["slat_length"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        backend=model_cfg["backend"],
        strict=model_cfg["strict"],
        num_context_tokens=model_cfg["num_context_tokens"],
        num_time_tokens=model_cfg["num_time_tokens"],
        num_r_tokens=model_cfg["num_r_tokens"],
        num_interval_tokens=model_cfg["num_interval_tokens"],
        num_guidance_tokens=model_cfg["num_guidance_tokens"],
        use_per_layer_context=model_cfg["use_per_layer_context"],
        context_cond_mode=model_cfg["context_cond_mode"],
        context_use_arcface_only=model_cfg["context_use_arcface_only"],
        num_context_kv_tokens=model_cfg["num_context_kv_tokens"],
        context_cross_attn_heads=model_cfg["context_cross_attn_heads"],
    ).to(device)
    
    ckpt_path = "checkpoints/imf_v8_lite/latest_step.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/imf_v8_lite/epoch_240.pt"
        
    print(f"[Load] Loading iMF checkpoint from {ckpt_path}...")
    ckpt_imf = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_imf = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt_imf.get("model_state_dict", ckpt_imf).items()}
    imf.load_state_dict(state_imf, strict=False)
    imf.eval()
    print(f"[Load] iMF loaded successfully.")
    
    return sc_vae, imf

def get_aligned_sample(sc_vae, device):
    # Try different slat dbs
    slat_db_path = "data/slat_context_balanced.lmdb"
    if not os.path.exists(slat_db_path):
        slat_db_path = "data/slat_context.lmdb"
        
    print(f"[LMDB] Connecting to Slat DB: {slat_db_path}")
    env_slat = lmdb.open(slat_db_path, readonly=True, lock=False)
    env_ov = lmdb.open("data/ovoxel_cache_lmdb", readonly=True, lock=False)
    
    chosen_key_slat = None
    chosen_key_ovoxel = None
    
    with env_slat.begin(write=False) as txn_slat:
        cursor = txn_slat.cursor()
        for k, v in cursor:
            if k != b"__meta__":
                key_str = k.decode()
                parts = key_str.split("/")
                dataset_name = parts[0]
                if dataset_name != "faceverse":
                    continue
                rel_path = "/".join(parts[1:])
                # Map relative path to ovoxel key (replacing slashes with underscores)
                ovoxel_key = rel_path.replace("/", "_").replace(".obj", ".c10.shape_mat.mx350000.pt")
                
                with env_ov.begin(write=False) as txn_ov:
                    # Check if mapped key exists
                    if txn_ov.get(ovoxel_key.encode("utf-8")) is not None:
                        chosen_key_slat = k
                        chosen_key_ovoxel = ovoxel_key.encode("utf-8")
                        break
                        
    if chosen_key_slat is None:
        raise RuntimeError("No matching key found between Slat and OVoxel LMDB cache databases.")
        
    print(f"[LMDB] Aligned Slat Key: {chosen_key_slat.decode()} -> OVoxel Key: {chosen_key_ovoxel.decode()}")
    
    with env_slat.begin(write=False) as txn_slat:
        v_slat = txn_slat.get(chosen_key_slat)
        payload_slat = torch.load(io.BytesIO(v_slat), map_location="cpu", weights_only=False)
        
    with env_ov.begin(write=False) as txn_ov:
        v_ov = txn_ov.get(chosen_key_ovoxel)
        payload_ov = torch.load(io.BytesIO(v_ov), map_location="cpu", weights_only=False)
        
    context = payload_slat["context"]
    
    coords = payload_ov["coords"].int()
    feats = payload_ov["features"].float()
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

    with torch.no_grad():
        x = sc_vae.enc1(sparse_input)
        x = sc_vae.enc2(x)
        x = sc_vae.enc3(x)
        x4 = sc_vae.enc4(x)
        
    return sparse_input.indices, x4, context, chosen_key_slat.decode()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sc_vae, imf = load_models(device)
    print("Models loaded successfully.")

    # 1. Load Slat Stats for Denormalization
    slat_norm_mean = None
    slat_norm_std = None
    stats_path = "data/slat_stats.pt"
    if os.path.exists(stats_path):
        print(f"[Stats] Loading slat normalization statistics from {stats_path}...")
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1).contiguous()  # [1, 1, 32]
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1).contiguous()    # [1, 1, 32]
        print(f"  Loaded stats: mean range [{slat_norm_mean.min().item():.4f}, {slat_norm_mean.max().item():.4f}]")
    else:
        print("[Stats] Warning: slat_stats.pt not found. Inference will proceed without denormalization.")

    # 2. Get matched Ground Truth Topology and Context from LMDB
    original_indices, x4_template, context_t, key_str = get_aligned_sample(sc_vae, device)
    n_active = x4_template.features.shape[0]
    print(f"GT Topology active tokens: {n_active}")
    
    context = context_t.unsqueeze(0).to(device) # [1, 946]
    print(f"Context shape: {context.shape}")

    # 3. Generate Slat using iMF (Flow Matching 1-Step)
    print("\n--- Generating Slat via iMF (1-Step) ---")
    imf_sampler = ImprovedMeanFlow()
    pred_slat_4096 = imf_sampler.sample_1_step(
        imf,
        context,
        shape=(1, 4096, 32),
        omega=1.0,
        cfg_tmin=0.0,
        cfg_tmax=1.0,
    )
    print(f"Generated raw Slat shape: {pred_slat_4096.shape}")
    
    # 4. Denormalize generated slats
    if slat_norm_mean is not None and slat_norm_std is not None:
        print("[Denorm] Applying slat denormalization (un-normalize)...")
        pred_slat_4096 = pred_slat_4096 * slat_norm_std + slat_norm_mean
    
    # Extract only the active tokens (since padding zeros were at the end of the Hilbert sequence)
    pred_mu = pred_slat_4096[0, :n_active, :].contiguous()
    print(f"Extracted predicted active mu shape: {pred_mu.shape}")
    
    # Measure distance between Predicted Slat and GT Slat
    with torch.no_grad():
        gt_mu = sc_vae.to_mu(x4_template.features)
        mse_loss = torch.nn.functional.mse_loss(pred_mu, gt_mu).item()
        print(f"\n--- METRICS ---")
        print(f"MSE Distance (Predicted Slat vs GT Slat): {mse_loss:.6f}")

    # 5. Decode Slat -> OVoxel features
    print("\n--- DECODING (Slat -> OVoxel) ---")
    with torch.no_grad():
        voxel_feats, _, _, out_indices = sc_vae.decode(
            pred_mu,
            original_indices=original_indices,
            sparse_template=x4_template,
            batch_size=1,
            return_indices=True,
        )
        print(f"Decoded voxel features shape: {voxel_feats.shape}")

    # 6. Extract Raw Point Cloud
    print("\n--- EXTRACTING POINT CLOUD ---")
    feats_np = voxel_feats.detach().cpu().float().numpy()
    out_indices_np = out_indices.detach().cpu().numpy()
    
    # Extract coords and intra-voxel offsets
    coords_np = out_indices_np[:, 1:4].astype(np.float32)
    v_np = np.clip(feats_np[:, :3], 0.0, 1.0)
    
    # Compute spatial coordinate based on AABB [-1.0, 1.0] and grid size 256
    grid_size = 256.0
    voxel_size = 2.0 / grid_size
    points_np = (coords_np + v_np) * voxel_size - 1.0
    
    colors_np = None
    colors_uint8 = None
    if feats_np.shape[1] >= 10:
        colors_np = np.nan_to_num(feats_np[:, 7:10], nan=0.5, posinf=1.0, neginf=0.0)
        colors_np = np.clip(colors_np, 0.0, 1.0)
        colors_uint8 = (colors_np * 255.0).astype(np.uint8)
        
    os.makedirs("outputs_pipeline_test", exist_ok=True)
    pcd = trimesh.points.PointCloud(vertices=points_np, colors=colors_uint8)
    pcd_path = "outputs_pipeline_test/imf_mamba_raw_pointcloud.ply"
    pcd.export(pcd_path)
    print(f"Raw Point Cloud saved to {pcd_path} ({len(points_np)} points)")

    # 7. Extract Poisson Mesh
    print("\n--- EXTRACTING POISSON MESH ---")
    poisson_verts, poisson_faces, poisson_colors = extract_poisson_mesh(
        points_np,
        colors_np,
        poisson_depth=9,
        density_quantile=0.01,
        smooth_iters=3,
        target_faces=0,
    )
    
    if poisson_verts is not None:
        poisson_path = "outputs_pipeline_test/imf_mamba_poisson_recon.ply"
        save_ply(poisson_verts, poisson_faces, poisson_colors, poisson_path)
        print(f"Poisson Mesh saved successfully to {poisson_path} with {len(poisson_verts)} vertices!")
    else:
        print("Failed to extract Poisson mesh.")

    # 8. Extract Standard Voxel Mesh (Fallback/Dual Contouring comparison)
    print("\n--- EXTRACTING STANDARD VOXEL MESH ---")
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
        voxel_mesh_path = "outputs_pipeline_test/imf_mamba_voxel_recon.ply"
        save_ply(verts, faces, colors, voxel_mesh_path)
        print(f"Standard Voxel Mesh saved to {voxel_mesh_path} with {len(verts)} vertices!")
    else:
        print("Failed to extract standard voxel mesh.")

if __name__ == "__main__":
    main()
