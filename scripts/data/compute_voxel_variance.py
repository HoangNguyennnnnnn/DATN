#!/usr/bin/env python3
"""Per-voxel variance qua identity (trên slat normalized) → voxel nào mang identity.
Dùng cho variance-weighted velocity loss (Bước 3 / Hướng B)."""
import io, sys, lmdb, torch

LMDB = sys.argv[1] if len(sys.argv) > 1 else "data/slat_context_both_balanced.lmdb"
STATS = sys.argv[2] if len(sys.argv) > 2 else "data/slat_stats_both.pt"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/voxel_variance.pt"

ss = torch.load(STATS, weights_only=False)
mean = ss["mean"].view(1, -1); std = ss["std"].view(1, -1)
env = lmdb.open(LMDB, readonly=True, lock=False, readahead=False)
N = 0
sum_v = torch.zeros(4096, dtype=torch.float64)
sum_v2 = torch.zeros(4096, dtype=torch.float64)
with env.begin() as t:
    cur = t.cursor(); cur.first()
    for k, v in cur:
        if k == b"__meta__":
            continue
        b = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
        s = (b["slat"].float() - mean) / std
        vn = s.norm(dim=-1).double()
        sum_v += vn; sum_v2 += vn * vn; N += 1
        if N % 2000 == 0:
            print(f"  ...{N}", flush=True)
env.close()
mean_v = sum_v / N
var_v = (sum_v2 / N - mean_v * mean_v).clamp(min=0)
torch.save({"voxel_variance": var_v.float(), "n_samples": N}, OUT)
q75 = var_v.quantile(0.75)
print(f"N={N} median={var_v.median():.4f} q75={q75:.4f} max={var_v.max():.4f} "
      f"top25={int((var_v>q75).sum())} → saved {OUT}", flush=True)
