#!/usr/bin/env python3
"""So sánh context (thô / sau MLP / prefix) với nhau và với nhiễu."""
from __future__ import annotations

import argparse
import io
import os
import sys

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.imf_diffusion import slice_contrastive_context
from src.models.voxel_mamba import VoxelMamba


def _cos_matrix(x: torch.Tensor) -> np.ndarray:
    x = F.normalize(x.float(), dim=-1)
    return (x @ x.t()).cpu().numpy()


def _offdiag_stats(mat: np.ndarray) -> dict:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    v = mat[mask]
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def load_contexts(lmdb_path: str, n: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
        keys = [keys[i] for i in picks]
        ctxs = []
        names = []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            ctxs.append(blob["context"].float().flatten())
            names.append(k.decode()[:48])
    env.close()
    return torch.stack(ctxs), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--ckpt", default="checkpoints/imf_v7_phaseB/epoch_40.pt")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-model", action="store_true", help="Chỉ phân tích raw ctx, không load VoxelMamba")
    args = ap.parse_args()

    ctx, names = load_contexts(args.lmdb, args.n, args.seed)
    n = ctx.shape[0]
    print("=" * 70)
    print(f"  CONTEXT vs NOISE  (n={n}, lmdb={args.lmdb})")
    print("=" * 70)

    # --- Raw hybrid 946-d ---
    mat_full = _cos_matrix(ctx)
    st = _offdiag_stats(mat_full)
    print("\n[1] Hybrid context thô (946-d, L2 norm ~√3 nếu balanced)")
    print(f"    ||ctx|| mean={ctx.norm(dim=-1).mean():.4f} std={ctx.norm(dim=-1).std():.4f}")
    print(f"    off-diag cos: mean={st['mean']:.4f} std={st['std']:.4f} "
          f"min={st['min']:.4f} max={st['max']:.4f}")

    arc = slice_contrastive_context(ctx, "arcface")
    flame = slice_contrastive_context(ctx, "flame")
    dino = ctx[:, 512 + 50 :]
    for label, block in [("ArcFace 512", arc), ("FLAME 50", flame), ("DINO 384", dino)]:
        st_b = _offdiag_stats(_cos_matrix(block))
        print(f"    [{label}] off-diag cos mean={st_b['mean']:.4f} "
              f"(||block|| mean={block.norm(dim=-1).mean():.3f})")

    # --- vs Gaussian noise same dim ---
    noise = torch.randn_like(ctx)
    noise = noise / noise.norm(dim=-1, keepdim=True) * ctx.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cn = F.normalize(ctx, dim=-1) @ F.normalize(noise, dim=-1).t()
    print("\n[2] Context vs nhiễu Gaussian (cùng dim, norm khớp từng mẫu)")
    print(f"    cos(ctx_i, noise_j) mean={cn.mean():.4f} std={cn.std():.4f} "
          f"min={cn.min():.4f} max={cn.max():.4f}")
    print(f"    cos(ctx_i, noise_i) diag mean={cn.diag().mean():.4f}")
    print(f"    (random unit 946-d: E[cos]≈0, std≈{1/np.sqrt(946):.4f})")

    # --- vs shuffle permuted ctx ---
    perm = torch.randperm(n)
    ctx_perm = ctx[perm]
    cp = F.normalize(ctx, dim=-1) @ F.normalize(ctx_perm, dim=-1).t()
    print("\n[3] Context vs bản shuffle (cùng phân phối, khác ghép cặp)")
    print(f"    cos(ctx_i, ctx_perm_i) diag mean={cp.diag().mean():.4f}")
    print(f"    cos(ctx_i, ctx_perm_j) off-diag mean={_offdiag_stats(cp.numpy())['mean']:.4f}")

    if args.no_model:
        return

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("stage2_model_config", {})
    seg_w = mcfg.get("context_segment_weights")
    if seg_w is not None:
        seg_w = tuple(float(x) for x in seg_w)
    else:
        seg_w = (1.5, 1.0, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoxelMamba(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        slat_length=mcfg["slat_length"],
        context_dim=mcfg["context_dim"],
        backend="gru" if device.type == "cpu" else "auto",
        num_context_tokens=mcfg.get("num_context_tokens", 8),
        num_time_tokens=mcfg.get("num_time_tokens", 4),
        num_r_tokens=mcfg.get("num_r_tokens", 4),
        num_interval_tokens=mcfg.get("num_interval_tokens", 4),
        num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
        use_per_layer_context=bool(mcfg.get("use_per_layer_context", True)),
        context_segment_weights=seg_w,
        dropout=0.0,
    ).to(device)
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt.get("ema_state_dict", ckpt["model_state_dict"]).items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"\n[4] Sau model (ckpt={args.ckpt}, seg_w={seg_w})")

    ctx_d = ctx.to(device)
    with torch.no_grad():
        ctx_scaled = model._scale_context_segments(ctx_d)
        ctx_cond = model.context_cond_mlp(ctx_scaled)
        ctx_tokens = model._make_prefix_tokens(
            model.context_tokenizer, ctx_d, model.num_context_tokens, n
        )
        if ctx_tokens.shape[1] > 0:
            tok_flat = ctx_tokens.reshape(n, -1)
        else:
            tok_flat = None

    st_mlp = _offdiag_stats(_cos_matrix(ctx_cond))
    print(f"    context_cond_mlp (512-d): off-diag cos mean={st_mlp['mean']:.4f} "
          f"min={st_mlp['min']:.4f} max={st_mlp['max']:.4f}")

    noise_d = torch.randn_like(ctx_d)
    noise_scaled = model._scale_context_segments(noise_d)
    with torch.no_grad():
        noise_cond = model.context_cond_mlp(noise_scaled)
    cn_mlp = F.normalize(ctx_cond, dim=-1) @ F.normalize(noise_cond, dim=-1).t()
    print(f"    cos(ctx_cond_i, noise_cond_j): mean={cn_mlp.mean():.4f} diag={cn_mlp.diag().mean():.4f}")

    if tok_flat is not None:
        st_tok = _offdiag_stats(_cos_matrix(tok_flat))
        print(f"    prefix ctx tokens flat ({tok_flat.shape[1]}-d): off-diag cos mean={st_tok['mean']:.4f}")

    # ctx_layer_proj[0]
    with torch.no_grad():
        ctx_l0 = model.ctx_layer_projs[0](ctx_cond)
    st_l0 = _offdiag_stats(_cos_matrix(ctx_l0))
    print(f"    ctx_layer_projs[0]: off-diag cos mean={st_l0['mean']:.4f}")

    # u velocity: same z, ctx vs noise vs zero ctx
    stats = torch.load("data/slat_stats.pt", map_location="cpu", weights_only=False)
    mean = stats["mean"].to(device).view(1, 1, -1)
    std = stats["std"].to(device).view(1, 1, -1)
    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        k0 = [k for k, _ in txn.cursor() if k != b"__meta__"][0]
        slat = torch.load(io.BytesIO(txn.get(k0)), map_location="cpu", weights_only=False)["slat"].float()
    env.close()
    x = ((slat.unsqueeze(0).expand(n, -1, -1) - mean) / std).to(device)
    z = 0.5 * x + 0.5 * torch.randn_like(x)
    t = torch.full((n,), 0.5, device=device)
    r = torch.zeros(n, device=device)
    o = torch.ones(n, device=device)
    z0 = torch.zeros(n, device=device)

    with torch.no_grad():
        u_real = model(z, t, ctx_d, r=r, omega=o, cfg_tmin=z0, cfg_tmax=o).float()
        u_noise_ctx = model(z, t, noise_d, r=r, omega=o, cfg_tmin=z0, cfg_tmax=o).float()
        u_zero = model(z, t, torch.zeros_like(ctx_d), r=r, omega=o, cfg_tmin=z0, cfg_tmax=o).float()

    u_flat = u_real.reshape(n, -1)
    un_flat = u_noise_ctx.reshape(n, -1)
    uz_flat = u_zero.reshape(n, -1)
    mat_u = _cos_matrix(u_flat)
    st_u = _offdiag_stats(mat_u)
    print(f"\n[5] u(z,t=0.5) — cùng z, khác context (velocity 4096×32 flat)")
    print(f"    u(correct ctx) off-diag cos mean={st_u['mean']:.4f}")
    cos_noise = F.cosine_similarity(u_flat, un_flat, dim=1)
    cos_zero = F.cosine_similarity(u_flat, uz_flat, dim=1)
    print(f"    cos(u_ctx_i, u_noise_ctx_i) per-sample mean={cos_noise.mean():.4f}")
    print(f"    cos(u_ctx_i, u_zero_ctx_i)  per-sample mean={cos_zero.mean():.4f}")
    cos_wrong = F.cosine_similarity(u_flat, u_flat[perm.to(device)], dim=1)
    print(f"    cos(u_ctx_i, u_ctx_perm_i) shuffle diag mean={cos_wrong.mean():.4f}")

    print("\n" + "=" * 70)
    print("  ĐỌC NHANH")
    print("  - Raw Arc off-diag thấp (~0.3–0.7) = identity khác nhau trong data")
    print("  - ctx_cond off-diag cao (~0.9+) nhưng u vẫn ~1 = Mamba/gate triệt tiêu")
    print("  - cos(u, u_noise) ≈ cos(u, u_ctx) → model không phân biệt ctx vs nhiễu")
    print("=" * 70)


if __name__ == "__main__":
    main()
