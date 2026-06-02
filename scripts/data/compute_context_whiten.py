#!/usr/bin/env python3
"""PCA-whiten transform cho context, tính trên 1 LMDB. Loại chiều variance gần-0 (FLAME-constant)."""
import io, sys, lmdb, torch

LMDB = sys.argv[1] if len(sys.argv) > 1 else "data/slat_context_both_balanced.lmdb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/context_whiten.pt"

env = lmdb.open(LMDB, readonly=True, lock=False, readahead=False)
cx = []
with env.begin() as t:
    cur = t.cursor(); cur.first()
    for k, v in cur:
        if k == b"__meta__":
            continue
        b = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
        cx.append(b["context"].float().flatten())
env.close()
ctx = torch.stack(cx)
print(f"context: {ctx.shape}", flush=True)
mean = ctx.mean(0)
Xc = ctx - mean
U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
eps = S.max() * 1e-3
keep = S > eps
W = (Vh[keep] / S[keep].unsqueeze(1)) * (Xc.shape[0] ** 0.5)
torch.save({"mean": mean, "W": W, "out_dim": int(keep.sum())}, OUT)
ctx_w = (ctx - mean) @ W.t()

def offdiag(M):
    Mn = torch.nn.functional.normalize(M, dim=-1); s = Mn @ Mn.t(); B = M.shape[0]
    return ((s.sum() - s.diag().sum()) / (B * (B - 1))).item()

print(f"out_dim={int(keep.sum())}/{len(S)} std={ctx_w.std():.3f} "
      f"raw_cos={offdiag(ctx[:64]):.4f} whitened_cos={offdiag(ctx_w[:64]):.4f} → saved {OUT}", flush=True)
