#!/usr/bin/env python3
"""Test ở SCALE: checkpoint train trên 20K, sample held-out với NOISE RIÊNG per-identity.
Phát hiện 30/05: dùng chung noise → collapse (lỗi đo). Noise riêng → identity-specific.
Đây là phép thử quyết định model combined có sinh đúng identity ở scale không.

Đo cross-matrix: gen[i] vs GT[j]. diag (đúng) phải >> off (sai)."""
import argparse, io, os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
import torch.nn.functional as F
from src.models.unet3d import voxel_unet3d_from_stage2_config


def load_model(ckpt, dev, no_ema=True):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = voxel_unet3d_from_stage2_config(c.get("stage2_model_config", {})).to(dev)
    sk = "ema_state_dict" if ("ema_state_dict" in c and not no_ema) else "model_state_dict"
    st = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in c[sk].items()}
    m.load_state_dict(st, strict=False); m.eval()
    print(f"  ckpt epoch={c.get('epoch')} using={sk}")
    return m


def load_samples(p, n, skip=0):
    env = lmdb.open(p, readonly=True, lock=False, readahead=False)
    sl, cx = [], []
    with env.begin() as t:
        cur = t.cursor(); cur.first(); c = 0; seen = 0
        for k, v in cur:
            if k == b"__meta__": continue
            if seen < skip: seen += 1; continue
            b = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
            sl.append(b["slat"].float()); cx.append(b["context"].float().flatten()); c += 1
            if c >= n: break
    env.close()
    return torch.stack(sl), torch.stack(cx)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/latest_step.pt")
    ap.add_argument("--lmdb", default="data/slat_context_faceverse_balanced.lmdb")
    ap.add_argument("--stats", default="data/slat_stats_both.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--prediction-type", default="velocity", choices=["velocity", "x0"])
    args = ap.parse_args()
    dev = torch.device("cuda")

    m = load_model(args.ckpt, dev)
    sl, ctx = load_samples(args.lmdb, args.n, args.skip)
    st = torch.load(args.stats, map_location="cpu", weights_only=False)
    mean = st["mean"].view(1, 1, -1).to(dev); std = st["std"].view(1, 1, -1).to(dev)
    x = (sl.to(dev) - mean) / std; ctx = ctx.to(dev); B = x.shape[0]
    om = torch.full((B,), args.omega, device=dev); zc = torch.zeros(B, device=dev); oc = torch.ones(B, device=dev)
    null = torch.zeros_like(ctx)

    # NOISE RIÊNG mỗi identity
    g = torch.Generator(device=dev).manual_seed(1234)
    noise = torch.randn(x.shape, generator=g, device=dev)

    z = noise.clone()
    N = args.steps
    x0_mode = (args.prediction_type == "x0")
    for k in range(N):
        tv = 1.0 - k / N
        tt = torch.full((B,), tv, device=dev)
        out_c = m(z, tt, ctx, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
        if x0_mode:
            # model dự đoán x0 → velocity = (z - x0)/t
            if args.omega != 1.0:
                out_u = m(z, tt, null, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
                x0h = out_u + args.omega * (out_c - out_u)
            else:
                x0h = out_c
            v = (z - x0h) / max(tv, 1e-3)
        else:
            if args.omega != 1.0:
                out_u = m(z, tt, null, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()
                v = out_u + args.omega * (out_c - out_u)
            else:
                v = out_c
        z = z - (1.0 / N) * v

    zn = F.normalize(z.flatten(1), dim=1); xn = F.normalize(x.flatten(1), dim=1)
    sim = zn @ xn.t()  # [B,B]: gen[i] vs GT[j]
    diag = sim.diag().mean().item()
    off = ((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item()
    x_off = ((xn @ xn.t()).sum() - (xn @ xn.t()).diag().sum()) / (B * (B - 1))

    print(f"\n  === SCALE gen (noise RIÊNG, ω={args.omega}, N={N}) ===")
    print(f"  GT baseline (2 mặt khác): cos={float(x_off):.3f}")
    print(f"  gen-self diag = {diag:.4f}  (cao = sinh đúng identity)")
    print(f"  cross    off  = {off:.4f}  (thấp = không lẫn identity)")
    print(f"  diag - off    = {diag - off:.4f}")
    print(f"  per-identity gen-self: {[f'{c:.2f}' for c in sim.diag().tolist()]}")
    if diag > 0.5 and (diag - off) > 0.2:
        print("  ==> ✅ Model SINH ĐÚNG identity ở scale → sẵn sàng decode→mesh")
    elif diag > 0.3:
        print("  ==> ⚠ có tín hiệu identity, chưa sắc")
    else:
        print("  ==> ❌ chưa sinh được ở scale (khác overfit)")


if __name__ == "__main__":
    main()
