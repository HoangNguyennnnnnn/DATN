#!/usr/bin/env python3
"""GĐ0 gate: conditioning + generation của UNet pretrain checkpoint ở SCALE.

Load checkpoint production (arch=unet3d), test trên N mẫu held-out FaceVerse:
  1. Conditioning: cos(u ctx đúng, u ctx sai) ở t=0.5, 0.9 — thấp = identity vào model.
  2. Generation: multi-step Euler từ noise → cos(gen, GT) diag vs off-diag.

PASS: ctx_cos < 0.7 VÀ gen diag-off > 0.2 → conditioning sống ở scale → cày tiếp.
FAIL: ctx_cos ~1.0 → conditioning sụp → bật CFG relaunch.
"""
import argparse, io, os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
import torch.nn.functional as F
from src.models.unet3d import voxel_unet3d_from_stage2_config


def load_model(ckpt_path, device, no_ema=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("stage2_model_config", {})
    assert mcfg.get("arch") == "unet3d", f"arch={mcfg.get('arch')} (cần unet3d)"
    model = voxel_unet3d_from_stage2_config(mcfg).to(device)
    sk = "ema_state_dict" if ("ema_state_dict" in ckpt and not no_ema) else "model_state_dict"
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt[sk].items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  ckpt epoch={ckpt.get('epoch')} using={sk} miss={len(miss)} unexp={len(unexp)}")
    return model


def load_samples(lmdb_path, n, skip=0):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    slats, ctxs = [], []
    with env.begin() as t:
        cur = t.cursor(); cur.first(); c = 0; seen = 0
        for k, v in cur:
            if k == b"__meta__":
                continue
            if seen < skip:
                seen += 1; continue
            b = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
            slats.append(b["slat"].float()); ctxs.append(b["context"].float().flatten()); c += 1
            if c >= n:
                break
    env.close()
    return torch.stack(slats), torch.stack(ctxs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/latest_step.pt")
    ap.add_argument("--lmdb", default="data/slat_context_faceverse_balanced.lmdb")
    ap.add_argument("--stats", default="data/slat_stats_both.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0, help="bỏ qua N mẫu đầu (lấy held-out)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda", help="cuda | cpu (cpu để không tranh VRAM với training)")
    args = ap.parse_args()
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = load_model(args.ckpt, dev, no_ema=args.no_ema)
    slats, ctx = load_samples(args.lmdb, args.n, args.skip)
    st = torch.load(args.stats, map_location="cpu", weights_only=False)
    mean = st["mean"].view(1, 1, -1).to(dev); std = st["std"].view(1, 1, -1).to(dev)
    x = (slats.to(dev) - mean) / std
    ctx = ctx.to(dev)
    B = x.shape[0]
    om = torch.ones(B, device=dev); zc = torch.zeros(B, device=dev); oc = torch.ones(B, device=dev)
    ctx_w = torch.roll(ctx, 1, 0)
    g = torch.Generator(device=dev).manual_seed(0)
    noise = torch.randn(x.shape, generator=g, device=dev)

    def fwd(z, tv, c):
        tt = torch.full((B,), tv, device=dev)
        return model(z, tt, c, r=tt, omega=om, cfg_tmin=zc, cfg_tmax=oc).float()

    print("\n  [1] Conditioning: cos(u ctx đúng, u ctx sai) — thấp = tốt")
    for tv in [0.3, 0.5, 0.7, 0.9]:
        z = (1 - tv) * x + tv * noise
        c = F.cosine_similarity(fwd(z, tv, ctx).flatten(1), fwd(z, tv, ctx_w).flatten(1), dim=1).mean().item()
        print(f"    t={tv}: cos={c:.4f}")

    print(f"\n  [2] Generation: multi-step Euler N={args.sample_steps}")
    z = noise.clone()
    N = args.sample_steps
    for k in range(N):
        z = z - (1.0 / N) * fwd(z, 1.0 - k / N, ctx)
    cos_self = F.cosine_similarity(z.flatten(1), x.flatten(1), dim=1)
    zn = F.normalize(z.flatten(1), dim=1); xn = F.normalize(x.flatten(1), dim=1)
    sim = zn @ xn.t(); diag = sim.diag().mean().item()
    off = ((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item() if B > 1 else 0.0
    print(f"    cos(gen,GT) self={cos_self.mean().item():.4f}  diag={diag:.4f} off={off:.4f} (chênh={diag-off:.4f})")

    # verdict dựa trên conditioning t=0.9 (gần điểm sinh) + gen chênh
    z9 = 0.1 * x + 0.9 * noise
    ctx_cos9 = F.cosine_similarity(fwd(z9, 0.9, ctx).flatten(1), fwd(z9, 0.9, ctx_w).flatten(1), dim=1).mean().item()
    print("\n  " + "=" * 50)
    if ctx_cos9 < 0.7 and (diag - off) > 0.2:
        print("  VERDICT: ✅ Conditioning SỐNG ở scale → cày tiếp pretrain.")
    elif ctx_cos9 < 0.9:
        print("  VERDICT: ⚠ Conditioning yếu — cân nhắc bật CFG sớm.")
    else:
        print("  VERDICT: ❌ Conditioning SỤP (cos~1) → DỪNG, bật CFG relaunch.")


if __name__ == "__main__":
    main()
