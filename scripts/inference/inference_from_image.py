"""Real image → 3D face mesh (VoxelUNet3D + iMF).

Pipeline:
  ảnh frontal (+ back optional) → ImagePreprocessor → context [946]
    → VoxelUNet3D sample slat (velocity CFG, multi-step) → un-normalize
    → SC-VAE decode → O-Voxel Dual Contouring → mesh .ply

Context whitening (946→632) do CHÍNH model áp nội bộ (buffer) ⇒ truyền context RAW 946-d.

Ví dụ:
  python scripts/inference/inference_from_image.py --input photo.jpg \
    --imf-ckpt checkpoints/imf_both20k/latest_step.pt \
    --sc-vae-ckpt checkpoints/sc_vae_both/latest_step.pt \
    --slat-stats data/slat_stats_both20k.pt \
    --steps 8 --omega 2 --output out.ply
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TrainConfig
from src.models.sc_vae import SC_VAE
from src.models.unet3d import voxel_unet3d_from_stage2_config
from scripts.test.test_e2e_inference import slat_to_mesh, save_ply


def _load_imf(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    model = voxel_unet3d_from_stage2_config(mcfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[imf] {ckpt_path} epoch={ckpt.get('epoch')} arch={mcfg.get('arch')}")
    return model, mcfg


def _load_scvae(ckpt_path: str, cfg: TrainConfig, device: torch.device):
    sc_vae = SC_VAE(
        in_channels=int(cfg.sc_vae.in_channels),
        latent_dim=int(cfg.sc_vae.latent_dim),
        num_res_blocks=int(cfg.sc_vae.num_res_blocks),
        encoder_dims=list(cfg.sc_vae.encoder_dims),
    ).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sc_vae.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    sc_vae.eval()
    print(f"[sc-vae] {ckpt_path} epoch={ck.get('epoch')}")
    return sc_vae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ảnh frontal (jpg/png).")
    ap.add_argument("--back", default=None, help="Ảnh back-of-head (optional; thiếu → DINO=mean/0).")
    ap.add_argument("--imf-ckpt", default="checkpoints/imf_both20k/latest_step.pt")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_both/latest_step.pt")
    ap.add_argument("--slat-stats", default="data/slat_stats_both20k.pt")
    ap.add_argument("--steps", type=int, default=8, help="Sampling steps (1=iMF 1-step; 8=mượt hơn).")
    ap.add_argument("--omega", type=float, default=2.0, help="CFG guidance (1=off).")
    ap.add_argument("--mask-threshold", type=float, default=2.8,
                    help="Raw slat norm threshold lọc voxel occupied (GT occupied min~2.8).")
    ap.add_argument("--output", default="outputs_inference/from_image.ply")
    ap.add_argument("--device", default="cuda:0", help="Device cho preprocess + iMF sample.")
    ap.add_argument("--decode-device", default="cuda:0", help="Device cho SC-VAE decode + DC.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug-dir", default=None, help="Lưu ảnh preprocessing debug.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    decode_device = torch.device(args.decode_device if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- 1. Ảnh → context [946] ----
    print("\n[1/4] Preprocess ảnh → context [946]...")
    from src.data.image_preprocessor import ImagePreprocessor
    t0 = time.time()
    pp = ImagePreprocessor(device=str(device))
    context = pp.process(args.input, back_image_path=args.back, save_debug_dir=args.debug_dir)
    print(f"       context shape={tuple(context.shape)} ({(time.time()-t0)*1000:.0f} ms)")
    context = context.reshape(1, -1).to(device)  # [1, 946] RAW (model tự whiten)
    del pp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 2. Load model + SC-VAE + stats ----
    print("\n[2/4] Load VoxelUNet3D + SC-VAE...")
    model, mcfg = _load_imf(args.imf_ckpt, device)
    sc_vae = _load_scvae(args.sc_vae_ckpt, cfg, decode_device)
    stats = torch.load(args.slat_stats, map_location="cpu", weights_only=False)
    slat_mean = stats["mean"].view(1, 1, -1)
    slat_std = stats["std"].view(1, 1, -1)
    slat_shape = (1, int(mcfg["slat_length"]), int(mcfg["input_dim"]))

    # ---- 3. Sample slat (velocity CFG, multi-step) ----
    print(f"\n[3/4] Sampling slat ({args.steps}-step, ω={args.omega})...")
    torch.manual_seed(args.seed)
    B = 1
    z = torch.randn(slat_shape, device=device)
    om = torch.full((B,), args.omega, device=device)
    zc = torch.zeros(B, device=device)
    oc = torch.ones(B, device=device)
    null = torch.zeros_like(context)
    t1 = time.time()
    for k in range(args.steps):
        tv = 1.0 - k / args.steps
        tt = torch.full((B,), tv, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            v_c = model(z, tt, context, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
            if args.omega != 1.0:
                v_u = model(z, tt, null, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
                v = v_u + args.omega * (v_c - v_u)
            else:
                v = v_c
        z = z - (1.0 / args.steps) * v
    print(f"       sampled in {(time.time()-t1)*1000:.0f} ms")

    # ---- 4. Un-normalize → mask → SC-VAE decode → mesh ----
    print("\n[4/4] Decode → mesh...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    slat_raw = (z.float().to(decode_device) * slat_std.to(decode_device) + slat_mean.to(decode_device))
    rn = slat_raw[0].norm(dim=-1)
    mask = (rn > args.mask_threshold)
    print(f"       occupied (norm>{args.mask_threshold}): {int(mask.sum())}/{slat_shape[1]} (med={rn.median():.2f})")
    verts, faces, colors, n_voxels = slat_to_mesh(slat_raw, sc_vae, decode_device, mask=mask)
    ok = save_ply(verts, faces, colors, args.output)
    if ok:
        print(f"\n✓ Mesh: {args.output} ({len(verts)} verts, {len(faces)} faces)")
    else:
        print("\n✗ Mesh rỗng — thử giảm --mask-threshold hoặc đổi --seed.")


if __name__ == "__main__":
    main()
