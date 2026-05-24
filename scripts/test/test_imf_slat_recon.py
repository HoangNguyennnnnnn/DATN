#!/usr/bin/env python3
"""cos_sim(slat_GT, slat_generated) — nhiễu + context → sample → so với GT.

1-step: z_0 = z_1 - u(z_1, r=0, t=1)
N-step: Euler từ 1→0 với cùng ε cố định (per sample).
So sánh context đúng vs shuffle.
"""
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

from scripts.test.test_imf_identity_t0 import _load_voxel_mamba_from_ckpt
from src.models.imf_diffusion import ImprovedMeanFlow


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    z1: torch.Tensor,
    context: torch.Tensor,
    num_steps: int,
    omega: torch.Tensor,
    cfg_tmin: torch.Tensor | None = None,
    cfg_tmax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler từ t=1 → 0: z_{t-dt} = z_t - dt * u(z_t, r=t_next, t=t_cur)."""
    b = z1.shape[0]
    device = z1.device
    if cfg_tmin is None:
        cfg_tmin = torch.zeros(b, device=device)
    if cfg_tmax is None:
        cfg_tmax = torch.ones(b, device=device)
    if num_steps <= 1:
        t = torch.ones(b, device=device)
        r = torch.zeros(b, device=device)
        u = model(
            z1, t, context, r=r,
            omega=omega,
            cfg_tmin=cfg_tmin,
            cfg_tmax=cfg_tmax,
        )
        return z1 - u

    z = z1
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur = ts[i].expand(b)
        t_nxt = ts[i + 1].expand(b)
        dt = (t_cur - t_nxt).view(-1, 1, 1)
        u = model(
            z, t_cur, context, r=t_nxt,
            omega=omega,
            cfg_tmin=cfg_tmin,
            cfg_tmax=cfg_tmax,
        )
        z = z - dt * u
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description="Slat GT vs generated slat (noise+context)")
    ap.add_argument("--ckpt", "--checkpoint", default="checkpoints/imf_v8_lite/best.pt")
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20],
        help="Số bước Euler (1 = iMF 1-step)",
    )
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--cfg-tmin", type=float, default=0.0, help="CFG interval start (paper eval: 0.4)")
    ap.add_argument("--cfg-tmax", type=float, default=1.0, help="CFG interval end (paper eval: 0.65)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--wrong-ctx", choices=("shuffle", "none"), default="shuffle")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("  SLAT RECON: cos_sim(GT, slat từ nhiễu + context)")
    print("=" * 72)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 0):.4f}")

    model, mcfg = _load_voxel_mamba_from_ckpt(ckpt, device, use_ema=not args.no_ema)
    diffusion = ImprovedMeanFlow()

    slat_mean = slat_std = None
    stats_path = mcfg.get("slat_stats_path") or "data/slat_stats.pt"
    if stats_path and os.path.exists(stats_path):
        st = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_mean = st["mean"].to(device).view(1, 1, -1)
        slat_std = st["std"].to(device).view(1, 1, -1)
        print(f"  [slat norm] {stats_path}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
        picks = rng.choice(len(keys), size=min(args.num_samples, len(keys)), replace=False)
        keys = [keys[i] for i in picks]
        contexts, slats_gt_raw = [], []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            contexts.append(blob["context"].float())
            slats_gt_raw.append(blob["slat"].float())
    env.close()

    contexts = torch.stack(contexts).to(device)
    slats_gt_raw = torch.stack(slats_gt_raw).to(device)
    b = slats_gt_raw.shape[0]

    if slat_mean is not None:
        slats_gt_norm = (slats_gt_raw - slat_mean) / slat_std
    else:
        slats_gt_norm = slats_gt_raw
        slat_mean = slat_std = None

    perm = torch.randperm(b, device=device)
    if (perm == torch.arange(b, device=device)).all() and b > 1:
        perm = torch.roll(torch.arange(b, device=device), 1)
    ctx_wrong = contexts[perm] if args.wrong_ctx == "shuffle" else None

    torch.manual_seed(args.seed)
    z1 = torch.randn_like(slats_gt_norm)  # cùng ε mọi num_steps

    omega = torch.full((b,), args.omega, device=device)
    cfg_tmin = torch.full((b,), args.cfg_tmin, device=device)
    cfg_tmax = torch.full((b,), args.cfg_tmax, device=device)

    print(f"\n  samples={b}, omega={args.omega}, cfg=[{args.cfg_tmin},{args.cfg_tmax}], steps={args.num_steps}")
    print(f"  z_1 ~ N(0,I) trong không gian slat đã normalize\n")

    def unnorm(z: torch.Tensor) -> torch.Tensor:
        if slat_mean is None:
            return z
        return z * slat_std + slat_mean

    print(f"  {'steps':>6} {'cos_ok':>10} {'cos_bad':>10} {'Δcos':>8} {'mse_ok':>10} {'mse_bad':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for n_steps in sorted(set(args.num_steps)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if n_steps == 1:
                z0_ok = diffusion.sample_1_step(
                    model, contexts, shape=tuple(slats_gt_norm.shape),
                    omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                ).float()
                if ctx_wrong is not None:
                    z0_bad = diffusion.sample_1_step(
                        model, ctx_wrong, shape=tuple(slats_gt_norm.shape),
                        omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                else:
                    z0_bad = None
            else:
                z0_ok = sample_euler(
                    model, z1, contexts, n_steps, omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                ).float()
                if ctx_wrong is not None:
                    z0_bad = sample_euler(
                        model, z1, ctx_wrong, n_steps, omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                else:
                    z0_bad = None

        pred_ok = unnorm(z0_ok)
        pred_bad = unnorm(z0_bad) if z0_bad is not None else None

        cos_ok_l, mse_ok_l = [], []
        cos_bad_l, mse_bad_l = [], []
        for i in range(b):
            gt = slats_gt_raw[i]
            pr = pred_ok[i]
            cos_ok_l.append(F.cosine_similarity(pr.flatten(), gt.flatten(), dim=0).item())
            mse_ok_l.append(F.mse_loss(pr, gt).item())
            if pred_bad is not None:
                pb = pred_bad[i]
                cos_bad_l.append(F.cosine_similarity(pb.flatten(), gt.flatten(), dim=0).item())
                mse_bad_l.append(F.mse_loss(pb, gt).item())

        mean_cos_ok = float(np.mean(cos_ok_l))
        mean_mse_ok = float(np.mean(mse_ok_l))
        if pred_bad is not None:
            mean_cos_bad = float(np.mean(cos_bad_l))
            mean_mse_bad = float(np.mean(mse_bad_l))
            delta = mean_cos_ok - mean_cos_bad
            print(
                f"  {n_steps:>6} {mean_cos_ok:10.4f} {mean_cos_bad:10.4f} {delta:8.4f} "
                f"{mean_mse_ok:10.4f} {mean_mse_bad:10.4f}"
            )
        else:
            print(f"  {n_steps:>6} {mean_cos_ok:10.4f} {'—':>10} {'—':>8} {mean_mse_ok:10.4f} {'—':>10}")

        if n_steps == 1 or n_steps == max(args.num_steps):
            print(f"\n  Per-sample cos (ctx đúng), {n_steps} step(s):")
            for i in range(min(8, b)):
                print(f"    [{i}] cos={cos_ok_l[i]:.4f}  mse={mse_ok_l[i]:.4f}  key={keys[i][-50:]}")

    # cos(pred_ok, pred_bad) — slat sinh có khác identity không
    if ctx_wrong is not None:
        print(f"\n  cos_sim(slat_pred đúng ctx, slat_pred shuffle) theo num_steps:")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for n_steps in sorted(set(args.num_steps)):
                if n_steps == 1:
                    z0_ok = diffusion.sample_1_step(
                        model, contexts, shape=tuple(slats_gt_norm.shape),
                        omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                    z0_bad = diffusion.sample_1_step(
                        model, ctx_wrong, shape=tuple(slats_gt_norm.shape),
                        omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                else:
                    z0_ok = sample_euler(
                        model, z1, contexts, n_steps, omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                    z0_bad = sample_euler(
                        model, z1, ctx_wrong, n_steps, omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
                    ).float()
                p_ok = unnorm(z0_ok)
                p_bad = unnorm(z0_bad)
                cu = float(np.mean([
                    F.cosine_similarity(p_ok[i].flatten(), p_bad[i].flatten(), dim=0).item()
                    for i in range(b)
                ]))
                print(f"    steps={n_steps:>2}  cos(pred_ok, pred_wrong)={cu:.4f}")

    print(f"\n{'='*72}")
    print("  ĐỌC KẾT QUẢ")
    print("  - cos → 1: slat sinh trùng hướng GT (không đảm bảo decode mesh đẹp).")
    print("  - cos_ok > cos_bad: context đúng giúp reconstruct đúng người hơn.")
    print("  - 1-step iMF là mục tiêu train; nhiều step Euler có thể tốt hơn hoặc tệ hơn.")
    print("=" * 72)


if __name__ == "__main__":
    main()
