#!/usr/bin/env python3
"""Overfit 1 mẫu — VoxelMamba v7 + loss chuẩn (ctx_sep >= 0)."""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TrainConfig
from src.models.imf_diffusion import ImprovedMeanFlow, contrastive_target_dim
from src.models.voxel_mamba import VoxelMamba


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_sample(lmdb_path: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, str]:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
        k = keys[int(seed) % len(keys)]
        blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
    env.close()
    return blob["slat"].float(), blob["context"].float().flatten(), k.decode()


def load_second_context(lmdb_path: str, avoid_key: str, seed: int = 1) -> torch.Tensor:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__" and k.decode() != avoid_key]
        k = keys[int(seed) % len(keys)]
        blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
    env.close()
    return blob["context"].float().flatten()


def build_model(device: torch.device, seg_w: tuple[float, float, float] = (3.0, 2.0, 0.5)) -> VoxelMamba:
    ic = TrainConfig().imf
    return VoxelMamba(
        input_dim=ic.input_dim,
        hidden_dim=ic.mamba_hidden_dim,
        num_layers=ic.mamba_num_layers,
        slat_length=ic.slat_length,
        context_dim=ic.context_dim,
        backend="auto",
        strict=False,
        num_context_tokens=ic.mamba_num_context_tokens,
        num_time_tokens=ic.mamba_num_time_tokens,
        num_r_tokens=ic.mamba_num_r_tokens,
        num_interval_tokens=ic.mamba_num_interval_tokens,
        num_guidance_tokens=ic.mamba_num_guidance_tokens,
        use_per_layer_context=ic.mamba_use_per_layer_context,
        d_state=ic.mamba_d_state,
        d_conv=ic.mamba_d_conv,
        expand=ic.mamba_expand,
        dropout=0.0,
        context_segment_weights=seg_w,
    ).to(device)


@torch.no_grad()
def ctx_cos_at_t(model, x_norm, ctx_a, ctx_b, t_val: float, device) -> float:
    model.eval()
    ctx_a, ctx_b = ctx_a.to(device), ctx_b.to(device)
    t = torch.full((1,), t_val, device=device)
    r = torch.zeros(1, device=device)
    z = (1.0 - t_val) * x_norm + t_val * torch.randn_like(x_norm)
    o, z0, o1 = torch.ones(1, device=device), torch.zeros(1, device=device), torch.ones(1, device=device)
    u_a = model(z, t, ctx_a.unsqueeze(0), r=r, omega=o, cfg_tmin=z0, cfg_tmax=o1).float()
    u_b = model(z, t, ctx_b.unsqueeze(0), r=r, omega=o, cfg_tmin=z0, cfg_tmax=o1).float()
    return F.cosine_similarity(u_a.flatten(), u_b.flatten(), dim=0).item()


@torch.no_grad()
def one_step_cos(model, imf, x_raw, ctx, mean, std) -> float:
    device = next(model.parameters()).device
    x_n = (x_raw.to(device) - mean) / std
    pred = imf.sample_1_step(model, ctx.to(device).unsqueeze(0), shape=x_n.shape, omega=torch.ones(1, device=device))
    return F.cosine_similarity(pred.float().flatten(), x_n.float().flatten(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--iters-per-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ratio-r-neq-t", type=float, default=0.0)
    ap.add_argument("--contrastive-weight", type=float, default=0.2)
    ap.add_argument("--ctx-sep-weight", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slat, ctx, key = load_sample(args.lmdb)
    ctx_wrong = load_second_context(args.lmdb, avoid_key=key, seed=99)

    stats = torch.load("data/slat_stats.pt", map_location="cpu", weights_only=False)
    mean = stats["mean"].to(device).view(1, 1, -1)
    std = stats["std"].to(device).view(1, 1, -1)

    model = build_model(device)
    log(f"params={sum(p.numel() for p in model)/1e6:.2f}M prefix={model.total_prefix_tokens}")

    ctx_classifier = None
    if args.contrastive_weight > 0:
        ctx_classifier = nn.Sequential(
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, contrastive_target_dim(946, "arcface"))
        ).to(device)

    imf = ImprovedMeanFlow(
        sigma_min=1e-4, ratio_r_neq_t=args.ratio_r_neq_t,
        t_sampler="logit_normal", t_loc=-0.4, t_scale=1.0,
        adaptive_loss_weighting=False,
    )

    x = (slat.unsqueeze(0).to(device) - mean) / std
    x2 = torch.cat([x, x], dim=0)
    ctx2 = torch.stack([ctx, ctx_wrong], dim=0).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + (list(ctx_classifier.parameters()) if ctx_classifier else []),
        lr=args.lr, betas=(0.9, 0.95),
    )

    log(f"pre 1-step={one_step_cos(model, imf, slat, ctx, mean, std):.4f} ctx@.5={ctx_cos_at_t(model, x, ctx, ctx_wrong, 0.5, device):.4f}")
    log(f"{'ep':>4} {'loss':>8} {'bnd':>8} {'ctr':>8} {'ctxsep':>8} {'cos_u':>8} {'1step':>8} {'ctx@.5':>8}")

    best = {"1step": -1.0, "ctx": 2.0}
    for ep in range(1, args.epochs + 1):
        model.train()
        if ctx_classifier:
            ctx_classifier.train()
        sums = dict(loss=0.0, bnd=0.0, ctr=0.0, sep=0.0, cos=0.0)
        t0 = time.time()
        for _ in range(args.iters_per_epoch):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = imf.compute_loss(
                    model, x2, ctx2,
                    ctx_classifier=ctx_classifier,
                    contrastive_loss_weight=args.contrastive_weight,
                    contrastive_mode="arcface",
                    context_velocity_sep_weight=args.ctx_sep_weight,
                    return_components=True,
                )
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums["loss"] += float(loss.item())
            sums["bnd"] += float(out.get("loss_boundary", 0))
            sums["ctr"] += float(out.get("loss_contrastive", 0))
            sums["sep"] += float(out.get("loss_context_sep", 0))
            sums["cos"] += float(out.get("ctx_sep_cos", 0))

        n = args.iters_per_epoch
        if ep % args.eval_every == 0 or ep == 1:
            c1 = one_step_cos(model, imf, slat, ctx, mean, std)
            cc = ctx_cos_at_t(model, x, ctx, ctx_wrong, 0.5, device)
            best["1step"] = max(best["1step"], c1)
            best["ctx"] = min(best["ctx"], cc)
            log(f"{ep:>4} {sums['loss']/n:>8.4f} {sums['bnd']/n:>8.4f} {sums['ctr']/n:>8.4f} "
                f"{sums['sep']/n:>8.4f} {sums['cos']/n:>8.4f} {c1:>8.4f} {cc:>8.4f} ({time.time()-t0:.0f}s)")
            if c1 > 0.9 and cc < 0.5:
                log(f"PASS ep{ep}")
                return

    log(f"DONE best_1step={best['1step']:.4f} best_ctx@.5={best['ctx']:.4f}")


if __name__ == "__main__":
    main()
