#!/usr/bin/env python3
"""Phân tích context chỉ tập FaceVerse trong slat_context LMDB."""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from collections import defaultdict

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.imf_diffusion import slice_contrastive_context


def parse_identity(key: str) -> str:
    # faceverse/011_01/011_01.obj -> 011
    parts = key.split("/")
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.split("_")[0]


def load_faceverse_contexts(lmdb_path: str, max_samples: int | None = None):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    key_bytes: list[bytes] = []
    with env.begin() as txn:
        for k, _ in txn.cursor():
            if k == b"__meta__":
                continue
            if k.decode().startswith("faceverse/"):
                key_bytes.append(bytes(k))
    keys = [k.decode() for k in key_bytes]
    keys.sort()
    key_bytes.sort()
    if max_samples is not None and len(keys) > max_samples:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(keys), size=max_samples, replace=False))
        keys = [keys[i] for i in idx]
        key_bytes = [key_bytes[i] for i in idx]

    ctxs = []
    with env.begin() as txn:
        for kb in key_bytes:
            blob = torch.load(io.BytesIO(txn.get(kb)), map_location="cpu", weights_only=False)
            ctxs.append(blob["context"].float().flatten())
    env.close()
    return keys, torch.stack(ctxs)


def offdiag_mean(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(mat[mask].mean())


def within_cross_identity(keys: list[str], ctx: torch.Tensor, block_fn):
    """Trung bình cos: cùng identity khác expression vs khác identity."""
    block = block_fn(ctx)
    block = F.normalize(block.float(), dim=-1)
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_id[parse_identity(k)].append(i)

    same_cos, diff_cos = [], []
    n = len(keys)
    for i in range(n):
        idi = parse_identity(keys[i])
        for j in range(i + 1, n):
            idj = parse_identity(keys[j])
            c = float((block[i] @ block[j]).item())
            if idi == idj:
                same_cos.append(c)
            else:
                diff_cos.append(c)
    return {
        "same_id_mean": float(np.mean(same_cos)) if same_cos else float("nan"),
        "same_id_std": float(np.std(same_cos)) if same_cos else float("nan"),
        "same_id_n": len(same_cos),
        "diff_id_mean": float(np.mean(diff_cos)) if diff_cos else float("nan"),
        "diff_id_std": float(np.std(diff_cos)) if diff_cos else float("nan"),
        "diff_id_n": len(diff_cos),
        "margin": float(np.mean(diff_cos) - np.mean(same_cos))
        if same_cos and diff_cos
        else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = toàn bộ FaceVerse")
    ap.add_argument("--compare-facescape", type=int, default=200, help="Số mẫu FaceScape để so sánh")
    args = ap.parse_args()

    max_n = None if args.max_samples <= 0 else args.max_samples
    keys, ctx = load_faceverse_contexts(args.lmdb, max_n)
    n = ctx.shape[0]
    n_id = len({parse_identity(k) for k in keys})

    print("=" * 72)
    print("  FACEVERSE CONTEXT ANALYSIS")
    print(f"  LMDB: {args.lmdb}")
    print(f"  samples={n}  identities≈{n_id}  (tổng FaceVerse trong LMDB: 2100)")
    print("=" * 72)

    print(f"\n||ctx|| mean={ctx.norm(dim=-1).mean():.4f} std={ctx.norm(dim=-1).std():.4f}")
    print(f"  (balanced LMDB: norm ~√3 ≈ 1.732)")

    blocks = [
        ("Hybrid 946-d", lambda c: c),
        ("ArcFace 512", lambda c: slice_contrastive_context(c, "arcface")),
        ("FLAME 50", lambda c: slice_contrastive_context(c, "flame")),
        ("DINO 384", lambda c: c[:, 512 + 50 :]),
    ]

    print("\n--- A) Ma trận cos ngẫu nhiên (off-diag mọi cặp) ---")
    print(f"  {'block':<14} {'off-diag mean':>14} {'min':>8} {'max':>8}")
    for label, fn in blocks:
        x = fn(ctx)
        mat = (F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).t()).cpu().numpy()
        m = offdiag_mean(mat)
        mask = ~np.eye(n, dtype=bool)
        v = mat[mask]
        print(f"  {label:<14} {m:>14.4f} {v.min():>8.4f} {v.max():>8.4f}")

    print("\n--- B) Cùng identity (khác expression) vs khác identity ---")
    print(f"  {'block':<14} {'same ID':>10} {'diff ID':>10} {'margin':>10}  (margin cao = identity tách tốt)")
    for label, fn in blocks:
        st = within_cross_identity(keys, ctx, fn)
        print(
            f"  {label:<14} {st['same_id_mean']:>10.4f} {st['diff_id_mean']:>10.4f} "
            f"{st['margin']:>10.4f}"
        )

    # Per-identity ArcFace spread
    arc = F.normalize(slice_contrastive_context(ctx, "arcface"), dim=-1)
    print("\n--- C) ArcFace: cos giữa các expression CÙNG một người (mẫu 5 ID) ---")
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_id[parse_identity(k)].append(i)
    shown = 0
    for pid in sorted(by_id.keys())[:8]:
        idxs = by_id[pid]
        if len(idxs) < 2:
            continue
        sub = arc[idxs]
        sm = (sub @ sub.t()).cpu().numpy()
        mask = ~np.eye(len(idxs), dtype=bool)
        mean_same = float(sm[mask].mean()) if mask.any() else 1.0
        exprs = [keys[i].split("/")[1] for i in idxs[:4]]
        print(f"  ID {pid} ({len(idxs)} expr): mean cos={mean_same:.4f}  e.g. {exprs}")
        shown += 1

    if args.compare_facescape > 0:
        print(f"\n--- D) So sánh nhanh FaceVerse vs FaceScape (n_fv={n}, n_fs={args.compare_facescape}) ---")
        env = lmdb.open(args.lmdb, readonly=True, lock=False)
        fs_keys = []
        with env.begin() as txn:
            for k, _ in txn.cursor():
                if k == b"__meta__":
                    continue
                ks = k.decode()
                if ks.startswith("facescape/"):
                    fs_keys.append(ks)
        env.close()
        rng = np.random.default_rng(0)
        fs_keys = [fs_keys[i] for i in rng.choice(len(fs_keys), size=min(args.compare_facescape, len(fs_keys)), replace=False)]
        fs_bytes = [k.encode() for k in fs_keys]
        fs_ctx = []
        env2 = lmdb.open(args.lmdb, readonly=True, lock=False)
        with env2.begin() as txn:
            for kb in fs_bytes:
                blob = torch.load(io.BytesIO(txn.get(kb)), map_location="cpu", weights_only=False)
                fs_ctx.append(blob["context"].float().flatten())
        env2.close()
        fs_ctx = torch.stack(fs_ctx)

        for label, fn in blocks:
            fv_b = fn(ctx)
            fs_b = fn(fs_ctx)
            fv_m = offdiag_mean((F.normalize(fv_b, dim=-1) @ F.normalize(fv_b, dim=-1).t()).cpu().numpy())
            fs_m = offdiag_mean((F.normalize(fs_b, dim=-1) @ F.normalize(fs_b, dim=-1).t()).cpu().numpy())
            print(f"  {label:<14}  FV off-diag={fv_m:.4f}   FS off-diag={fs_m:.4f}")

    print("\n" + "=" * 72)
    print("  ĐỌC KẾT QUẢ FACEVERSE")
    print("  - same ID cao + diff ID thấp (FLAME/DINO) → expression/geometry trùng, khó học identity riêng")
    print("  - ArcFace: cần margin (diff - same) lớn → mới đủ signal cho iMF")
    print("  - Vai (shoulders) trong mesh FaceVerse → DINO có thể đồng nhất hơn FaceScape")
    print("=" * 72)


if __name__ == "__main__":
    main()
