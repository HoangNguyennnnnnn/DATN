#!/usr/bin/env python3
"""Audit ArcFace / FLAME / DINO blocks in context LMDBs."""
from __future__ import annotations

import argparse
import io
import os
import random
import sys
from collections import defaultdict

import lmdb
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ARC, FLAME, DINO = 512, 50, 384


def load_ctx(blob) -> torch.Tensor:
    if isinstance(blob, (bytes, bytearray)):
        obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            return obj["context"].float().flatten()
        return obj.float().flatten()
    return blob["context"].float().flatten()


def split(ctx: torch.Tensor):
    ctx = ctx.flatten()
    return ctx[:ARC], ctx[ARC : ARC + FLAME], ctx[ARC + FLAME :]


def stats(name: str, vecs: torch.Tensor):
    v = vecs.float()
    n = v.shape[0]
    norms = v.norm(dim=-1)
    nz = (v.abs().sum(dim=-1) > 1e-6).float().mean().item() * 100
    uniq = len(torch.unique(v.round(decimals=4), dim=0))
    print(f"  [{name}] n={n}")
    print(f"    ||v|| mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    print(f"    nonzero rows={nz:.1f}%  unique(4dp)={uniq}/{n}")
    print(f"    per-dim std mean={v.std(dim=0).mean():.6f} max={v.std(dim=0).max():.6f}")
    if n >= 2:
        v_n = F.normalize(v, dim=-1)
        sim = v_n @ v_n.T
        mask = ~torch.eye(n, dtype=torch.bool)
        print(f"    pairwise cos: mean={sim[mask].mean():.4f} min={sim[mask].min():.4f} max={sim[mask].max():.4f}")


def audit_lmdb(path: str, max_samples: int, seed: int):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    keys = [k for k, _ in env.begin().cursor() if k != b"__meta__"]
    random.seed(seed)
    if len(keys) > max_samples:
        keys = random.sample(keys, max_samples)

    arcs, flames, dinos, full = [], [], [], []
    with env.begin() as txn:
        for k in keys:
            raw = txn.get(k)
            if raw is None:
                continue
            ctx = load_ctx(raw)
            if ctx.numel() != ARC + FLAME + DINO:
                continue
            a, f, d = split(ctx)
            arcs.append(a)
            flames.append(f)
            dinos.append(d)
            full.append(ctx)

    env.close()
    print(f"\n{'='*60}\n  {path}\n  samples={len(full)} / keys={len(keys)}\n{'='*60}")
    if not full:
        print("  (no valid samples)")
        return

    stats("ArcFace", torch.stack(arcs))
    stats("FLAME", torch.stack(flames))
    stats("DINO", torch.stack(dinos))
    stats("Full 946", torch.stack(full))

    # Same person different expr (facescape path heuristic)
    by_subject = defaultdict(list)
    for k, ctx in zip(keys[: len(full)], full):
        parts = k.decode().split("/")
        subj = "/".join(parts[:3]) if len(parts) >= 3 else parts[0]
        by_subject[subj].append(ctx)

    multi = [v for v in by_subject.values() if len(v) >= 3][:20]
    if multi:
        arc_sims, flame_sims, dino_sims = [], [], []
        for group in multi:
            C = torch.stack(group)
            a, f, d = split(C[0])
            A = torch.stack([split(c)[0] for c in group])
            Fm = torch.stack([split(c)[1] for c in group])
            D = torch.stack([split(c)[2] for c in group])
            An = F.normalize(A, dim=-1)
            Fn = F.normalize(Fm, dim=-1)
            Dn = F.normalize(D, dim=-1)
            arc_sims.append((An @ An.T)[torch.triu(torch.ones(len(group), len(group)), 1) == 1].mean().item())
            fn = Fm.norm(dim=-1)
            if fn.max() > 1e-4:
                flame_sims.append((Fn @ Fn.T)[torch.triu(torch.ones(len(group), len(group)), 1) == 1].mean().item())
            dino_sims.append((Dn @ Dn.T)[torch.triu(torch.ones(len(group), len(group)), 1) == 1].mean().item())
        print("  [Same subject, multi expression — expect Arc high, FLAME lower, DINO mid]")
        print(f"    ArcFace cos mean={sum(arc_sims)/len(arc_sims):.4f}")
        if flame_sims:
            print(f"    FLAME cos mean={sum(flame_sims)/len(flame_sims):.4f} (n={len(flame_sims)})")
        else:
            print("    FLAME: all near-zero in these groups")
        print(f"    DINO cos mean={sum(dino_sims)/len(dino_sims):.4f}")

    # Cross-block correlation (full vectors)
    C = torch.stack(full)
    Cn = (C - C.mean(0)) / (C.std(0).clamp(min=1e-8))
    a_idx = slice(0, ARC)
    f_idx = slice(ARC, ARC + FLAME)
    d_idx = slice(ARC + FLAME, None)
    def block_corr(idx):
        X = Cn[:, idx]
        return (X.T @ X).abs().mean().item()
    print(f"  [Block internal corr proxy] Arc={block_corr(a_idx):.4f} FLAME={block_corr(f_idx):.4f} DINO={block_corr(d_idx):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for p in [
        "data/slat_context.lmdb",
        "data/slat_context_balanced.lmdb",
        "data/hybrid_context.lmdb",
    ]:
        if os.path.isdir(p):
            audit_lmdb(p, args.max_samples, args.seed)


if __name__ == "__main__":
    main()
