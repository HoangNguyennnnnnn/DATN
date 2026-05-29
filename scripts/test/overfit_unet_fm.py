#!/usr/bin/env python3
"""A/B test backbone: 3D UNet (conv, data-efficient) + standard FM v-pred + multi-step.

Slat [B,4096,32] = grid 16³×32 (raster) → reshape [B,32,16,16,16] cho conv3d.
Conditioning: time + hybrid context (946) qua FiLM (kiểu DDPM class-cond).

So với VoxelMamba (overfit_multistep_fm.py) trên CÙNG harness:
  n=1: UNet sinh được (cos>0.9) mà Mamba không → backbone là thủ phạm.
"""
import argparse, io, math, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
import torch.nn as nn
import torch.nn.functional as F

G = 16          # grid size per dim (16³ = 4096)
CH = 32         # slat channels


def load_samples(lmdb_path, n):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    slats, ctxs = [], []
    with env.begin() as txn:
        cur = txn.cursor(); cur.first(); c = 0
        while c < n:
            k, v = cur.item()
            if k != b"__meta__":
                blob = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
                slats.append(blob["slat"].float()); ctxs.append(blob["context"].float().flatten()); c += 1
            if not cur.next():
                break
    env.close()
    return torch.stack(slats), torch.stack(ctxs)


def sinusoidal(t, dim):
    # t: [B] in [0,1]
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        s, b = self.film(cond)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + s) + b
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(self, base=128, ctx_dim=946, cond_dim=512):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(256, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.c_mlp = nn.Sequential(nn.Linear(ctx_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        c1, c2, c3 = base, base * 2, base * 4
        self.in_conv = nn.Conv3d(CH, c1, 3, padding=1)
        self.d1 = ResBlock3D(c1, c1, cond_dim)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)   # 16->8
        self.d2 = ResBlock3D(c2, c2, cond_dim)
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)   # 8->4
        self.mid1 = ResBlock3D(c3, c3, cond_dim)
        self.mid2 = ResBlock3D(c3, c3, cond_dim)
        self.up2 = ResBlock3D(c3 + c2, c2, cond_dim)
        self.up1 = ResBlock3D(c2 + c1, c1, cond_dim)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv3d(c1, CH, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, ctx):
        cond = self.t_mlp(sinusoidal(t, 256)) + self.c_mlp(ctx)
        h0 = self.in_conv(x)
        h0 = self.d1(h0, cond)                       # [B,c1,16,16,16]
        h1 = self.d2(self.down1(h0), cond)           # [B,c2,8,8,8]
        h2 = self.mid2(self.mid1(self.down2(h1), cond), cond)  # [B,c3,4,4,4]
        u2 = F.interpolate(h2, scale_factor=2, mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h1], dim=1), cond)        # [B,c2,8,8,8]
        u1 = F.interpolate(u2, scale_factor=2, mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h0], dim=1), cond)        # [B,c1,16,16,16]
        return self.out_conv(F.silu(self.out_norm(u1)))


def to_grid(s):   # [B,4096,32] -> [B,32,16,16,16]
    B = s.shape[0]
    return s.transpose(1, 2).reshape(B, CH, G, G, G)


def to_seq(g):    # [B,32,16,16,16] -> [B,4096,32]
    B = g.shape[0]
    return g.reshape(B, CH, G * G * G).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context_faceverse_balanced.lmdb")
    ap.add_argument("--stats", default="data/slat_stats_faceverse.pt")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--sample-steps", type=int, default=50)
    args = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dev = torch.device("cuda")

    slats, ctx = load_samples(args.lmdb, args.n)
    st = torch.load(args.stats, map_location="cpu", weights_only=False)
    mean = st["mean"].view(1, 1, -1).to(dev); std = st["std"].view(1, 1, -1).to(dev)
    x = to_grid(((slats.to(dev) - mean) / std))      # [B,32,16,16,16]
    ctx = ctx.to(dev)
    B = x.shape[0]

    model = UNet3D(base=args.base, ctx_dim=ctx.shape[-1]).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {B} | 3D UNet base={args.base} ({nparam:.1f}M) | FM v-pred + multistep({args.sample_steps})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    print(f"{'Step':>5} | {'mse_v':>8}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        e = torch.randn_like(x)
        t = torch.rand(B, device=dev)
        z_t = (1 - t).view(B, 1, 1, 1, 1) * x + t.view(B, 1, 1, 1, 1) * e
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_hat = model(z_t, t, ctx)
            loss = F.mse_loss(v_hat, e - x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"{step:>5} | {loss.item():>8.4f}  ({time.time()-t0:.0f}s)")
            t0 = time.time()

    # EVAL multi-step
    model.eval()
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(0)
        noise = torch.randn(x.shape, generator=g, device=dev)
        def sample(N):
            z = noise.clone()
            for k in range(N):
                tv = torch.full((B,), 1.0 - k / N, device=dev)
                z = z - (1.0 / N) * model(z, tv, ctx).float()
            return z
        steps_cmp = {NN: F.cosine_similarity(sample(NN).flatten(1), x.flatten(1), dim=1).mean().item()
                     for NN in [1, 4, 16, args.sample_steps]}
        z0 = sample(args.sample_steps)
        cos_self = F.cosine_similarity(z0.flatten(1), x.flatten(1), dim=1)
        z0n = F.normalize(z0.flatten(1), dim=1); xn = F.normalize(x.flatten(1), dim=1)
        sim = z0n @ xn.t(); diag = sim.diag().mean().item()
        off = ((sim.sum() - sim.diag().sum()) / (B * (B - 1))) if B > 1 else 0.0

    print("\n" + "=" * 56)
    print(f"  3D UNET RESULT (N={args.sample_steps})")
    print("=" * 56)
    print(f"  cos(gen, x_GT) self = {cos_self.mean().item():.4f}  (per: {[f'{c:.2f}' for c in cos_self.tolist()]})")
    print(f"  diag={diag:.4f}  off-diag={off:.4f}  (chênh={diag-off:.4f})")
    print(f"  theo bước: " + "  ".join(f"N={k}:{v:.3f}" for k, v in steps_cmp.items()))
    if cos_self.mean().item() > 0.8:
        print("  VERDICT: ✅ UNet SINH ĐƯỢC → backbone là thủ phạm. Chuyển UNet+iMF.")
    elif cos_self.mean().item() > 0.4:
        print("  VERDICT: ⚠ UNet tốt hơn Mamba rõ nhưng chưa đủ.")
    else:
        print("  VERDICT: ❌ UNet cũng fail → vấn đề KHÔNG ở backbone (data/setup).")


if __name__ == "__main__":
    main()
