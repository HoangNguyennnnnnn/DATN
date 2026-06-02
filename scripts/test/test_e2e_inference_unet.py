#!/usr/bin/env python3
"""
E2E inference test với UNet3D cho tập TEST SET context (mesh KHÔNG có trong train).
Sinh mesh từ random noise + context sử dụng UNet3D, xuất file .ply.
Đã tối ưu: Full GPU + Sửa lỗi Autocast + Clamp chống bọt biển (Sponge Cube).
"""
import argparse
import io
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TrainConfig
from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.sc_vae import SC_VAE
from src.models.unet3d import voxel_unet3d_from_stage2_config
from scripts.test.test_e2e_inference import slat_to_mesh, save_ply

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf-ckpt", default="checkpoints/imf_unet/latest_step.pt")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_600.pt")
    ap.add_argument("--context-lmdb", default="data/slat_context_faceverse_balanced.lmdb")
    ap.add_argument("--slat-lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--slat-stats", default="data/slat_stats_faceverse.pt")
    ap.add_argument("--n-samples", type=int, default=2, help="Số test samples")
    ap.add_argument("--steps", type=int, nargs="+", default=[50], help="Sampling steps")
    ap.add_argument("--omega", type=float, default=1.5)
    ap.add_argument("--out-dir", default="outputs_e2e")
    ap.add_argument("--device", default="cuda", help="Device for UNet3D sampling (cuda/cpu)")
    ap.add_argument("--decode-device", default="cuda", help="Device cho SC-VAE decode + DC.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    decode_device = torch.device(args.decode_device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load checkpoint UNet3D ---
    print(f"[1/5] Loading iMF UNet3D checkpoint: {args.imf_ckpt} (device={device})")
    ckpt = torch.load(args.imf_ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    print(f"   epoch={ckpt.get('epoch', 'N/A')} loss={ckpt.get('loss', 0.0):.4f}")

    model = voxel_unet3d_from_stage2_config(mcfg).to(device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # --- Load SC-VAE ---
    print(f"[2/5] Loading SC-VAE checkpoint: {args.sc_vae_ckpt} (decode_device={decode_device})")
    cfg = TrainConfig()
    sc_vae = SC_VAE(
        in_channels=int(cfg.sc_vae.in_channels),
        latent_dim=int(cfg.sc_vae.latent_dim),
        num_res_blocks=int(cfg.sc_vae.num_res_blocks),
        encoder_dims=list(cfg.sc_vae.encoder_dims),
    ).to(decode_device)
    sc_ckpt = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    sc_state = sc_ckpt.get("model_state_dict", sc_ckpt)
    sc_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sc_state.items()}
    sc_vae.load_state_dict(sc_state, strict=False)
    sc_vae.eval()

    # --- Load stats ---
    stats_path = args.slat_stats
    print(f"[3/5] Loading slat stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    slat_mean = stats["mean"].to(device).view(1, 1, -1)
    slat_std = stats["std"].to(device).view(1, 1, -1)

    # --- Load Test Samples ---
    print(f"[4/5] Loading test contexts from LMDB...")
    ctx_env = lmdb.open(args.context_lmdb, readonly=True, lock=False)
    
    test_keys = []
    with ctx_env.begin() as txn:
        cur = txn.cursor()
        cur.first()
        for k, _ in cur:
            if k == b"__meta__":
                continue
            test_keys.append(k)
            if len(test_keys) >= args.n_samples + 20:
                break
                
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(test_keys), size=min(args.n_samples, len(test_keys)), replace=False)
    test_keys = [test_keys[i] for i in picks]

    contexts, names = [], []
    with ctx_env.begin() as txn:
        for k in test_keys:
            raw = txn.get(k)
            blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            ctx = blob["context"].float()
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            contexts.append(ctx)
            names.append(k.decode().replace("/", "_").replace(".obj", ""))
    
    contexts = torch.cat(contexts, dim=0).to(device)
    ctx_env.close()

    # --- Generation and Export ---
    print(f"\n[5/5] Sampling UNet3D + Decoding to PLY...")
    slat_shape = (1, mcfg["slat_length"], mcfg["input_dim"])

    for i, name in enumerate(names):
        ctx = contexts[i : i + 1]
        print(f"\n--- Sample {i+1}/{len(names)}: {name} ---")
        for n_steps in args.steps:
            print(f"  [{n_steps}-step] Sampling...")
            torch.manual_seed(args.seed + i * 100)
            
            B = 1
            z = torch.randn(slat_shape, device=device)
            om = torch.full((B,), args.omega, device=device)
            zc = torch.zeros(B, device=device)
            oc = torch.ones(B, device=device)
            null = torch.zeros_like(ctx)
            
            for k in range(n_steps):
                tv = 1.0 - k / n_steps
                tt = torch.full((B,), tv, device=device)
                
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                    out_c = model(z, tt, ctx, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
                    if args.omega != 1.0:
                        out_u = model(z, tt, null, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
                        x0h = out_u + args.omega * (out_c - out_u)
                    else:
                        x0h = out_c
                
                v = (z - x0h) / max(tv, 1e-3)
                z = z - (1.0 / n_steps) * v
            
            slat_norm = z.float()
            
            # --- BẢN VÁ: Gọt bọt biển (Clamp) ---
            slat_norm = torch.clamp(slat_norm, min=-3.0, max=3.0)
            
            slat_raw = slat_norm * slat_std + slat_mean
            
            print(f"    Decoding via SC-VAE + DC...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                slat_norm = slat_norm.to(decode_device)
                
                # Check valid tokens based on normalized predictions
                # Since we trained the model to predict exactly 0.0 for empty space in the normalized domain,
                # slat_norm will be close to 0.0 for empty space.
                # Real surface tokens will have variance 1.0 per channel, so norm over 32 channels is ~sqrt(32)=5.65
                mask = (slat_norm.norm(dim=-1) > 1.0).squeeze(0) # Threshold safely above noise but below 5.65
                
                # Un-normalize for VAE decoding
                slat_raw = slat_norm * slat_std.to(slat_norm) + slat_mean.to(slat_norm)
                
                # Apply mask explicitly to empty space before passing to VAE
                slat_raw = slat_raw * mask.unsqueeze(-1).to(slat_raw.dtype)
                
                verts, faces, colors, n_voxels = slat_to_mesh(slat_raw, sc_vae, decode_device, mask=mask)
                out_path = os.path.join(args.out_dir, f"{name}_unet_{n_steps}step.ply")
                ok = save_ply(verts, faces, colors, out_path)
                if ok:
                    print(f"    ✓ Mesh exported successfully to: {out_path}")
            except Exception as e:
                print(f"    [ERROR] {e}")

    print("\n✓ Done!")

if __name__ == "__main__":
    main()