#!/usr/bin/env python3
"""
Rebalance context vectors trong slat_context.lmdb (giữ slat, chỉ sửa context).

Áp dụng balance_hybrid_context_segments: L2 norm từng khối ArcFace/FLAME/DINO
trước khi concat → MLP thấy identity (~33% năng lượng mỗi khối thay vì 0.05% ArcFace).

Usage:
    python scripts/rebalance_slat_context_lmdb.py \\
        --in-lmdb data/slat_context.lmdb \\
        --out-lmdb data/slat_context_balanced.lmdb
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys

import lmdb
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.flame_adapter import balance_hybrid_context_segments


def rebalance_ctx(ctx: torch.Tensor) -> torch.Tensor:
    ctx = ctx.float().flatten()
    if ctx.numel() != 946:
        raise ValueError(f"expected 946-d context, got {ctx.numel()}")
    return balance_hybrid_context_segments(
        ctx[:512].unsqueeze(0),
        ctx[512:562].unsqueeze(0),
        ctx[562:].unsqueeze(0),
    ).squeeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--out-lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--map-size-gb", type=int, default=15)
    args = ap.parse_args()

    if os.path.abspath(args.in_lmdb) == os.path.abspath(args.out_lmdb):
        raise SystemExit("in-lmdb and out-lmdb must differ (or remove out-lmdb first)")

    if os.path.isdir(args.out_lmdb):
        print(f"Removing existing {args.out_lmdb}")
        shutil.rmtree(args.out_lmdb)

    in_env = lmdb.open(args.in_lmdb, readonly=True, lock=False, readahead=False)
    out_env = lmdb.open(
        args.out_lmdb,
        map_size=args.map_size_gb * 1024**3,
        subdir=True,
        sync=False,
        writemap=True,
    )

    keys = []
    with in_env.begin() as txn:
        for k, _ in txn.cursor():
            if k != b"__meta__":
                keys.append(k)

    print(f"Rebalancing {len(keys)} entries: {args.in_lmdb} -> {args.out_lmdb}")

    packed = errors = 0
    txn_out = out_env.begin(write=True)
    with in_env.begin() as txn_in:
        for k in tqdm(keys, desc="Rebalance"):
            try:
                blob = torch.load(io.BytesIO(txn_in.get(k)), map_location="cpu", weights_only=False)
                slat = blob["slat"]
                ctx = rebalance_ctx(blob["context"])
                buf = io.BytesIO()
                torch.save({"slat": slat, "context": ctx}, buf)
                txn_out.put(k, buf.getvalue())
                packed += 1
                if packed % 1000 == 0:
                    txn_out.commit()
                    txn_out = out_env.begin(write=True)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [ERROR] {k.decode(errors='replace')}: {e}")

    meta = {
        "packed": packed,
        "errors": errors,
        "rebalanced_from": os.path.abspath(args.in_lmdb),
        "note": "Per-segment L2 balance: ArcFace+FLAME+DINO equal unit-norm blocks",
    }
    txn_out.put(b"__meta__", json.dumps(meta).encode("utf-8"))
    txn_out.commit()
    out_env.sync()
    out_env.close()
    in_env.close()

    print(f"Done: packed={packed}, errors={errors}")
    print(f"Output: {args.out_lmdb}")


if __name__ == "__main__":
    main()
