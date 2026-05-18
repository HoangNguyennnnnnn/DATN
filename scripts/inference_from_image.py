"""
End-to-End inference: Raw image → 3D face mesh (.ply).

Pipeline:
  ImagePreprocessor → context [946]
  VoxelMamba + iMF  → slat tokens [1, 4096, 32]   (1-step hoặc multi-step)
  reverse normalize → slat_raw
  SC-VAE decode + DC → mesh (.ply)

Examples:
  python scripts/inference_from_image.py \\
    --input photo.jpg --output mesh.ply

  python scripts/inference_from_image.py \\
    --input photo.jpg --back back.jpg --output mesh.ply \\
    --imf-ckpt checkpoints/imf_unet/best.pt \\
    --steps 1 20
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TrainConfig
from src.data.image_preprocessor import ImagePreprocessor
from src.models.sc_vae import SC_VAE
from src.models.voxel_mamba import VoxelMamba
from scripts.test_e2e_inference import sample_n_step, slat_to_mesh, save_ply


def _load_imf(ckpt_path: str, device: torch.device):
    print(f"[Load] iMF checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    print(f"       epoch={ckpt['epoch']} loss={ckpt['loss']:.4f}")

    model = VoxelMamba(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        slat_length=mcfg["slat_length"],
        context_dim=mcfg["context_dim"],
        backend=mcfg.get("backend", "auto"),
        num_context_tokens=mcfg.get("num_context_tokens", 8),
        num_time_tokens=mcfg.get("num_time_tokens", 4),
        num_r_tokens=mcfg.get("num_r_tokens", 4),
        num_interval_tokens=mcfg.get("num_interval_tokens", 4),
        num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
        d_state=mcfg.get("d_state", 16),
        d_conv=mcfg.get("d_conv", 4),
        expand=mcfg.get("expand", 2),
    ).to(device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, mcfg


def _load_scvae(ckpt_path: str, device: torch.device):
    print(f"[Load] SC-VAE checkpoint: {ckpt_path} (device={device})")
    cfg = TrainConfig()
    sc_vae = SC_VAE(
        in_channels=int(cfg.sc_vae.in_channels),
        latent_dim=int(cfg.sc_vae.latent_dim),
        num_res_blocks=int(cfg.sc_vae.num_res_blocks),
        encoder_dims=list(cfg.sc_vae.encoder_dims),
    ).to(device)
    sc_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sc_state = sc_ckpt.get("model_state_dict", sc_ckpt)
    sc_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sc_state.items()}
    sc_vae.load_state_dict(sc_state, strict=False)
    sc_vae.eval()
    return sc_vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ảnh frontal (jpg/png).")
    ap.add_argument("--back", default=None, help="Ảnh back-of-head (optional).")
    ap.add_argument("--output", required=True, help="Mesh output (.ply).")
    ap.add_argument("--imf-ckpt", default="checkpoints/imf_unet/best.pt")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_500.pt")
    ap.add_argument("--slat-stats", default="data/slat_stats.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=[1],
                    help="Sampling steps (mỗi giá trị → 1 file output).")
    ap.add_argument("--omega", type=float, default=4.0, help="CFG guidance scale.")
    ap.add_argument("--device", default="cuda:0", help="Device cho preprocessing + iMF sample.")
    ap.add_argument("--decode-device", default="cpu",
                    help="Device cho SC-VAE decode + DC. CPU an toàn khi training đang chạy.")
    ap.add_argument("--debug-dir", default=None, help="Lưu ảnh preprocessing debug.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    decode_device = torch.device(args.decode_device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Preprocess image → context ----
    print(f"\n[1/4] Preprocessing image → context...")
    t0 = time.time()
    pp = ImagePreprocessor(device=str(device))
    print(f"       preprocessor loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    context = pp.process(args.input, back_image_path=args.back, save_debug_dir=args.debug_dir)
    print(f"       context extracted in {(time.time() - t0) * 1000:.0f} ms; "
          f"shape={tuple(context.shape)}")
    # Free preprocessor (giải phóng ~1.5 GB VRAM trước khi load iMF)
    del pp
    torch.cuda.empty_cache()

    context = context.unsqueeze(0).to(device)  # [1, 946]

    # ---- 2. Load iMF model ----
    print(f"\n[2/4] Loading iMF model...")
    model, mcfg = _load_imf(args.imf_ckpt, device)

    # ---- 3. Load SC-VAE decoder ----
    print(f"\n[3/4] Loading SC-VAE decoder...")
    sc_vae = _load_scvae(args.sc_vae_ckpt, decode_device)

    # ---- 4. Load slat stats ----
    stats_path = mcfg.get("slat_stats_path") or args.slat_stats
    print(f"[Load] slat stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    slat_mean = stats["mean"].to(device).view(1, 1, -1)
    slat_std = stats["std"].to(device).view(1, 1, -1)

    # ---- 5. Sample + Decode for each n_steps ----
    print(f"\n[4/4] Sampling + Decoding (steps={args.steps})...")
    slat_shape = (1, mcfg["slat_length"], mcfg["input_dim"])
    out_base, out_ext = os.path.splitext(args.output)
    if out_ext.lower() != ".ply":
        print(f"[WARN] Output ext '{out_ext}' không phải .ply — sẽ ép sang .ply")
        out_ext = ".ply"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    for n_steps in args.steps:
        print(f"\n--- {n_steps}-step sampling ---")
        torch.manual_seed(args.seed)
        t1 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            slat_norm = sample_n_step(
                model, context,
                shape=slat_shape,
                num_steps=n_steps,
                omega=args.omega,
            )
        slat_norm = slat_norm.float()
        slat_raw = slat_norm * slat_std + slat_mean
        print(f"   sample: {(time.time() - t1) * 1000:.0f} ms; "
              f"slat_raw std={slat_raw.std().item():.3f} (target ~0.37)")

        t1 = time.time()
        torch.cuda.empty_cache()
        try:
            verts, faces, colors, n_voxels = slat_to_mesh(slat_raw, sc_vae, decode_device)
        except Exception as e:
            print(f"   [ERROR] slat_to_mesh: {e}")
            import traceback
            traceback.print_exc()
            continue
        print(f"   decode: {(time.time() - t1):.1f} s; "
              f"verts={len(verts) if verts is not None else 0}, "
              f"faces={len(faces) if faces is not None else 0}, voxels={n_voxels}")

        # Output path: nếu nhiều steps thì suffix
        if len(args.steps) > 1:
            out_path = f"{out_base}__{n_steps}step{out_ext}"
        else:
            out_path = out_base + out_ext
        ok = save_ply(verts, faces, colors, out_path)
        if ok:
            print(f"   ✓ Saved: {out_path}")

    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
