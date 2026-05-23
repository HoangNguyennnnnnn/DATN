"""
Remix slat_context.lmdb với context mới: giữ slat cũ, thay context bằng context
mới từ hybrid_context.lmdb (sau khi fix FLAME bug).

Use case:
  - Đã backup slat_context.lmdb cũ (chứa slat đúng + context broken-FLAME)
  - Đã chạy build_context_lmdb.py với new FLAME → hybrid_context.lmdb mới
  - Cần tạo slat_context.lmdb mới = slat cũ + context mới (không cần encode lại SC-VAE)

Usage:
    python scripts/remix_slat_lmdb_with_new_context.py \\
        --old-slat-lmdb data/backup_broken_flame_XXX/slat_context.lmdb \\
        --new-context-lmdb data/hybrid_context.lmdb \\
        --out-lmdb data/slat_context.lmdb
"""
import argparse
import io
import json
import os
import sys

import lmdb
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-slat-lmdb", required=True,
                    help="LMDB chứa {slat, context_OLD} — chỉ đọc slat")
    ap.add_argument("--new-context-lmdb", default="data/hybrid_context.lmdb",
                    help="LMDB chứa context MỚI (với MediaPipe FLAME)")
    ap.add_argument("--out-lmdb", default="data/slat_context.lmdb")
    args = ap.parse_args()

    print(f"[1/3] Opening LMDBs...")
    print(f"  OLD slat: {args.old_slat_lmdb}")
    print(f"  NEW context: {args.new_context_lmdb}")
    print(f"  OUT: {args.out_lmdb}")

    old_env = lmdb.open(args.old_slat_lmdb, readonly=True, lock=False, readahead=False)
    new_ctx_env = lmdb.open(args.new_context_lmdb, readonly=True, lock=False, readahead=False)

    # Pre-scan all keys in OLD slat_lmdb
    keys = []
    with old_env.begin() as t:
        for k, _ in t.cursor():
            if k != b"__meta__":
                keys.append(k)
    print(f"[2/3] {len(keys)} slat entries to remix")

    # Verify context coverage
    missing = 0
    with new_ctx_env.begin() as t:
        for k in keys[:100]:  # spot check
            if t.get(k) is None:
                missing += 1
    if missing > 0:
        print(f"  [WARN] {missing}/100 spot-check keys missing in new context LMDB!")

    # Output LMDB (drop existing if present)
    if os.path.exists(args.out_lmdb):
        print(f"  Removing existing output: {args.out_lmdb}")
        import shutil
        shutil.rmtree(args.out_lmdb)
    os.makedirs(os.path.dirname(args.out_lmdb) or ".", exist_ok=True)

    out_env = lmdb.open(args.out_lmdb, map_size=int(15 * 1024 ** 3), subdir=True)

    print(f"[3/3] Remixing...")
    packed = 0
    no_context = 0
    with out_env.begin(write=True) as out_txn, \
         old_env.begin() as old_txn, \
         new_ctx_env.begin() as ctx_txn:
        for k in tqdm(keys, desc="Remix"):
            # Read slat from OLD LMDB
            blob = torch.load(io.BytesIO(old_txn.get(k)), map_location="cpu", weights_only=False)
            slat = blob["slat"]
            # Read NEW context
            ctx_raw = ctx_txn.get(k)
            if ctx_raw is None:
                no_context += 1
                continue
            new_ctx = torch.load(io.BytesIO(ctx_raw), map_location="cpu", weights_only=False).float()
            if new_ctx.ndim == 0:
                new_ctx = new_ctx.unsqueeze(0)
            # Save merged
            buf = io.BytesIO()
            torch.save({"slat": slat, "context": new_ctx}, buf)
            out_txn.put(k, buf.getvalue())
            packed += 1

        # Save meta
        meta = {
            "packed": packed,
            "errors": no_context,
            "remixed_from": args.old_slat_lmdb,
            "new_context_from": args.new_context_lmdb,
            "note": "Slat unchanged, context replaced with MediaPipe-FLAME context",
        }
        out_txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    out_env.close()
    print(f"\n✓ Done. Packed {packed} entries. Missing context: {no_context}")
    print(f"  Output: {args.out_lmdb}")


if __name__ == "__main__":
    main()
