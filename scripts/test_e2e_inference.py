"""
E2E inference test với TEST SET context (mesh KHÔNG có trong train).
Sinh mesh từ random noise + context, so sánh 1-step vs 20-step.

Pipeline:
  context (test set, không train) → VoxelMamba sample (1 or 20 step)
    → reverse slat_norm → SC-VAE decode → O-Voxel DC → mesh .ply

Outputs: outputs_e2e/<key>_<n_steps>step.ply
"""
import argparse
import io
import itertools
import os
import sys

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import TrainConfig
from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.sc_vae import SC_VAE
from src.models.voxel_mamba import VoxelMamba


# ============================================================
# Sampling
# ============================================================
@torch.no_grad()
def sample_n_step(
    model: torch.nn.Module,
    context: torch.Tensor,
    shape: tuple,
    num_steps: int,
    omega: float = 4.0,
    cfg_tmin: float = 0.2,
    cfg_tmax: float = 0.8,
) -> torch.Tensor:
    """Multi-step Euler ODE sampling (interpolates from noise z_1 to data z_0)."""
    device = context.device
    b = context.shape[0]
    omega_t = torch.full((b,), omega, device=device, dtype=context.dtype)
    tmin_t = torch.full((b,), cfg_tmin, device=device, dtype=context.dtype)
    tmax_t = torch.full((b,), cfg_tmax, device=device, dtype=context.dtype)

    z = torch.randn(shape, device=device, dtype=context.dtype)
    if num_steps == 1:
        # iMF 1-step: z_0 = z_1 − u_θ(z_1, r=0, t=1)
        t_cur = torch.ones(b, device=device, dtype=context.dtype)
        r_cur = torch.zeros(b, device=device, dtype=context.dtype)
        u = model(z, t_cur, context, r=r_cur, omega=omega_t, cfg_tmin=tmin_t, cfg_tmax=tmax_t)
        return z - u

    # Multi-step Euler: integrate from t=1 → t=0
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=context.dtype)
    for i in range(num_steps):
        t_cur = ts[i].expand(b)
        t_nxt = ts[i + 1].expand(b)
        # mean-flow: u_θ(z_t, r, t) ≈ avg velocity over [r, t]; use (r=t_nxt, t=t_cur)
        u = model(z, t_cur, context, r=t_nxt, omega=omega_t, cfg_tmin=tmin_t, cfg_tmax=tmax_t)
        z = z - (t_cur - t_nxt).view(-1, 1, 1) * u
    return z


# ============================================================
# Slat → Mesh via SC-VAE + DC
# ============================================================
@torch.no_grad()
def slat_to_mesh(
    slat_raw: torch.Tensor,  # [1, L, 32] in SC-VAE latent space (post-unnormalize)
    sc_vae: SC_VAE,
    decode_device: torch.device,
    slat_grid_size: int = 16,
    ovoxel_resolution: int = 256,
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
    voxel_feats, _, _, out_indices = sc_vae.decode(
        z_flat,
        original_indices=grid_indices,
        batch_size=1,
        return_indices=True,
    )

    # Extract mesh via DC (sử dụng helper từ test_sc_vae_recon_v2)
    from scripts.test_sc_vae_recon_v2 import extract_ovoxel_mesh

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


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf-ckpt", default="checkpoints/imf_unet/best.pt")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_500.pt")
    ap.add_argument("--context-lmdb", default="data/hybrid_context.lmdb")
    ap.add_argument("--slat-lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--slat-stats", default="data/slat_stats.pt")
    ap.add_argument("--n-samples", type=int, default=4, help="Số test samples")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 20], help="Sampling steps")
    ap.add_argument("--omega", type=float, default=4.0)
    ap.add_argument("--dataset-filter", default="faceverse",
                    help="faceverse|facescape|both — chọn nguồn test")
    ap.add_argument("--out-dir", default="outputs_e2e")
    ap.add_argument("--decode-device", default="cpu",
                    help="Device cho SC-VAE decode + DC. Mặc định cpu để né OOM khi training đang chạy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # VoxelMamba PHẢI ở GPU (mamba-ssm CUDA kernel). SC-VAE decode có thể ở CPU để né OOM.
    device = torch.device("cuda")
    decode_device = torch.device(args.decode_device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load checkpoints ---
    print(f"[1/5] Loading iMF checkpoint: {args.imf_ckpt}")
    ckpt = torch.load(args.imf_ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    print(f"   epoch={ckpt['epoch']} loss={ckpt['loss']:.4f}")

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

    # --- Load SC-VAE (decode device — có thể CPU) ---
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
    print(f"   in_channels={cfg.sc_vae.in_channels}, latent_dim={cfg.sc_vae.latent_dim}")

    # --- Load slat normalization stats ---
    stats_path = mcfg.get("slat_stats_path") or args.slat_stats
    print(f"[3/5] Loading slat stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    slat_mean = stats["mean"].to(device).view(1, 1, -1)
    slat_std = stats["std"].to(device).view(1, 1, -1)
    print(f"   mean range [{slat_mean.min():.4f}, {slat_mean.max():.4f}], "
          f"std range [{slat_std.min():.4f}, {slat_std.max():.4f}]")

    # --- Tìm test samples (có context, KHÔNG có slat trong train LMDB) ---
    print(f"[4/5] Finding {args.n_samples} test contexts...")
    ctx_env = lmdb.open(args.context_lmdb, readonly=True, lock=False)
    slat_env = lmdb.open(args.slat_lmdb, readonly=True, lock=False)

    test_keys = []
    with ctx_env.begin() as ctx_txn, slat_env.begin() as slat_txn:
        for k, _ in ctx_txn.cursor():
            if k == b"__meta__":
                continue
            key_str = k.decode()
            # Filter by dataset
            if args.dataset_filter != "both" and not key_str.startswith(args.dataset_filter):
                continue
            # Test = trong context nhưng KHÔNG trong slat (train)
            if slat_txn.get(k) is None:
                test_keys.append(k)
                if len(test_keys) >= args.n_samples * 5:
                    break

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(test_keys), size=min(args.n_samples, len(test_keys)), replace=False)
    test_keys = [test_keys[i] for i in picks]
    print(f"   Picked {len(test_keys)} test keys:")
    for k in test_keys:
        print(f"     - {k.decode()}")

    # Load contexts
    contexts, names = [], []
    with ctx_env.begin() as txn:
        for k in test_keys:
            raw = txn.get(k)
            ctx = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False).float()
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            contexts.append(ctx)
            names.append(k.decode().replace("/", "_").replace(".obj", ""))
    contexts = torch.cat(contexts, dim=0).to(device)
    print(f"   Context tensor: {contexts.shape}")

    # --- Sample + Decode + Save ---
    print(f"\n[5/5] Sampling + Decoding...")
    slat_shape = (1, mcfg["slat_length"], mcfg["input_dim"])

    for i, name in enumerate(names):
        ctx = contexts[i : i + 1]
        print(f"\n--- Sample {i+1}/{len(names)}: {name} ---")
        for n_steps in args.steps:
            print(f"  [{n_steps}-step] Sampling...")
            torch.manual_seed(args.seed + i * 100 + n_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                slat_norm = sample_n_step(
                    model, ctx, shape=slat_shape,
                    num_steps=n_steps, omega=args.omega,
                )
            slat_norm = slat_norm.float()
            # Reverse normalization: slat_raw = slat_norm * std + mean
            slat_raw = slat_norm * slat_std + slat_mean
            print(f"    slat_norm std={slat_norm.std():.3f}, "
                  f"slat_raw std={slat_raw.std():.3f} (target ~0.37)")

            print(f"    Decoding via SC-VAE + DC (device={decode_device})...")
            try:
                # Free GPU memory before decode
                torch.cuda.empty_cache()
                verts, faces, colors, n_voxels = slat_to_mesh(slat_raw, sc_vae, decode_device)
                print(f"    Mesh: {len(verts) if verts is not None else 0} verts, "
                      f"{len(faces) if faces is not None else 0} faces, "
                      f"{n_voxels} sparse voxels")
                out_path = os.path.join(args.out_dir, f"{name}__{n_steps}step.ply")
                ok = save_ply(verts, faces, colors, out_path)
                if ok:
                    print(f"    ✓ Saved: {out_path}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback
                traceback.print_exc()

    print(f"\n✓ Done. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
