#!/usr/bin/env python3
"""cos_sim(u_pred, v_gt) với v_gt = ε − x (FM target), nhiều bước t.

So sánh context đúng vs shuffle (sai identity) trên cùng z_t, cùng ε.
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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="u_pred vs GT velocity (e-x) across t")
    ap.add_argument("--ckpt", "--checkpoint", default="checkpoints/imf_v8_lite/best.pt")
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--num-t-steps", type=int, default=21, help="Số điểm t trong [0,1]")
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wrong-ctx", choices=("shuffle", "zero", "none"), default="shuffle")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("  VELOCITY vs GT: cos_sim(u_pred, ε − x) theo t")
    print("=" * 72)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 0):.4f}")

    model, mcfg = _load_voxel_mamba_from_ckpt(
        ckpt, device, use_ema=not args.no_ema,
    )
    print(f"  arcface_only={mcfg.get('context_use_arcface_only', True)}")

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
        contexts, slats = [], []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            contexts.append(blob["context"].float())
            slats.append(blob["slat"].float())
    env.close()

    contexts = torch.stack(contexts).to(device)
    x_raw = torch.stack(slats).to(device)
    b = x_raw.shape[0]
    x = (x_raw - slat_mean) / slat_std if slat_mean is not None else x_raw

    # Cùng ε cho mọi t (per-sample)
    torch.manual_seed(args.seed)
    e = torch.randn_like(x)

    perm = torch.randperm(b, device=device)
    if (perm == torch.arange(b, device=device)).all() and b > 1:
        perm = torch.roll(torch.arange(b, device=device), 1)
    if args.wrong_ctx == "shuffle":
        ctx_wrong = contexts[perm]
    elif args.wrong_ctx == "zero":
        ctx_wrong = torch.zeros_like(contexts)
    else:
        ctx_wrong = None

    t_values = np.linspace(args.t_min, args.t_max, args.num_t_steps)
    omega = torch.ones(b, device=device)
    zcfg = torch.zeros(b, device=device)
    ocfg = torch.ones(b, device=device)

    cos_ok, cos_bad, mse_ok, mse_bad = [], [], [], []
    cos_per_t_ok: list[list[float]] = [[] for _ in range(b)]

    print(f"\n  samples={b}, t_steps={len(t_values)}, wrong_ctx={args.wrong_ctx}")
    print(f"  z_t = (1-t)*x + t*ε,  r=t (boundary),  v_gt = ε - x\n")
    print(f"  {'t':>6} {'cos_ok':>10} {'cos_bad':>10} {'Δcos':>10} {'mse_ok':>10} {'mse_bad':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for t_val in t_values:
            t = torch.full((b,), float(t_val), device=device)
            r = t.clone()
            t_view = t.view(-1, 1, 1)
            z_t = (1.0 - t_view) * x + t_view * e
            v_gt = e - x

            u_ok = model(
                z_t, t, contexts, r=r,
                omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg,
            ).float()

            if ctx_wrong is not None:
                u_bad = model(
                    z_t, t, ctx_wrong, r=r,
                    omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg,
                ).float()
            else:
                u_bad = None

            # Per-batch metrics
            c_ok = [
                F.cosine_similarity(u_ok[i].flatten(), v_gt[i].flatten(), dim=0).item()
                for i in range(b)
            ]
            m_ok = [F.mse_loss(u_ok[i], v_gt[i]).item() for i in range(b)]
            mean_c_ok = float(np.mean(c_ok))
            mean_m_ok = float(np.mean(m_ok))
            cos_ok.append(mean_c_ok)
            mse_ok.append(mean_m_ok)
            for i in range(b):
                cos_per_t_ok[i].append(c_ok[i])

            if u_bad is not None:
                c_bad = [
                    F.cosine_similarity(u_bad[i].flatten(), v_gt[i].flatten(), dim=0).item()
                    for i in range(b)
                ]
                m_bad = [F.mse_loss(u_bad[i], v_gt[i]).item() for i in range(b)]
                mean_c_bad = float(np.mean(c_bad))
                mean_m_bad = float(np.mean(m_bad))
                cos_bad.append(mean_c_bad)
                mse_bad.append(mean_m_bad)
                delta = mean_c_ok - mean_c_bad
                print(
                    f"  {t_val:6.3f} {mean_c_ok:10.4f} {mean_c_bad:10.4f} {delta:10.4f} "
                    f"{mean_m_ok:10.4f} {mean_m_bad:10.4f}"
                )
            else:
                cos_bad.append(float("nan"))
                mse_bad.append(float("nan"))
                print(f"  {t_val:6.3f} {mean_c_ok:10.4f} {'—':>10} {'—':>10} {mean_m_ok:10.4f} {'—':>10}")

    print(f"\n{'='*72}")
    print("  TÓM TẮT (trung bình trên t)")
    print(f"{'='*72}")
    print(f"  cos_ok:  mean={np.nanmean(cos_ok):.4f}  min={np.nanmin(cos_ok):.4f}  max={np.nanmax(cos_ok):.4f}")
    if ctx_wrong is not None:
        print(f"  cos_bad: mean={np.nanmean(cos_bad):.4f}  min={np.nanmin(cos_bad):.4f}  max={np.nanmax(cos_bad):.4f}")
        print(f"  margin (ok-bad): mean={np.nanmean(np.array(cos_ok)-np.array(cos_bad)):+.4f}")
    print(f"  mse_ok:  mean={np.nanmean(mse_ok):.4f}")

    # Vùng t train (logit-normal ~0.2-0.6)
    mask_mid = (t_values >= 0.15) & (t_values <= 0.65)
    if mask_mid.any():
        print(f"\n  Vùng t∈[0.15,0.65] (train-heavy):")
        print(f"    cos_ok mean={np.mean(np.array(cos_ok)[mask_mid]):.4f}")
        if ctx_wrong is not None:
            print(f"    cos_bad mean={np.mean(np.array(cos_bad)[mask_mid]):.4f}")

    print(f"\n  Per-sample cos_ok @ t=0.5: {[f'{cos_per_t_ok[i][len(t_values)//2]:.3f}' for i in range(min(8,b))]}")

    # cos(u_ok, u_bad) theo t — model output có phụ thuộc ctx không
    if ctx_wrong is not None:
        print(f"\n{'='*72}")
        print("  cos_sim(u_correct_ctx, u_wrong_ctx) theo t (không so GT)")
        print(f"{'='*72}")
        cos_uu = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for t_val in t_values:
                t = torch.full((b,), float(t_val), device=device)
                r = t.clone()
                t_view = t.view(-1, 1, 1)
                z_t = (1.0 - t_view) * x + t_view * e
                u_ok = model(z_t, t, contexts, r=r, omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
                u_bad = model(z_t, t, ctx_wrong, r=r, omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
                cu = float(np.mean([
                    F.cosine_similarity(u_ok[i].flatten(), u_bad[i].flatten(), dim=0).item()
                    for i in range(b)
                ]))
                cos_uu.append(cu)
        for t_val, cu in zip(t_values, cos_uu):
            print(f"  t={t_val:.3f}  cos(u_ok,u_bad)={cu:.4f}")
        print(f"  mean={np.mean(cos_uu):.4f}")

    print(f"\n{'='*72}")
    print("  ĐỌC KẾT QUẢ")
    print("  - cos_ok → 1: u_pred cùng hướng với ε−x (GT flow). Thấp @ mọi t → chưa học field.")
    print("  - cos_ok > cos_bad: context đúng giúp khớp GT hơn shuffle.")
    print("  - t≈0: z_t≈x, v_gt=ε−x; t≈1: z_t≈ε — cos thường cao hơn ở giữa nếu train logit-normal.")
    print("=" * 72)


if __name__ == "__main__":
    main()
