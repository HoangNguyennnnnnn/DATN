#!/usr/bin/env python3
"""
Pack precomputed slat+context .pt files into a single LMDB for fast iMF training.
=================================================================================
Reads from ``data/slat_cache/`` (FaceVerse) and ``data/slat_cache_facescape/``
(FaceScape), merges each ``{slat, context}`` entry into one LMDB at
``data/slat_context.lmdb/``.

Usage:
    conda activate facediff
    python scripts/pack_slat_lmdb.py [--output data/slat_context.lmdb]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import lmdb
import torch
from tqdm import tqdm


def _scan_pt_files(cache_dir: str, dataset_name: str) -> list[tuple[str, str, str]]:
    """Return list of (abs_path, lmdb_key, dataset_name) for all .pt files."""
    results = []
    if not os.path.isdir(cache_dir):
        print(f"[WARN] Cache dir not found: {cache_dir}")
        return results
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".pt"):
            continue
        abs_path = os.path.join(cache_dir, fname)
        # Key format: dataset_name/filename (without cache_tag suffix for readability)
        lmdb_key = f"{dataset_name}/{fname}"
        results.append((abs_path, lmdb_key, dataset_name))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack slat .pt caches into LMDB")
    parser.add_argument("--fv-cache-dir", type=str, default="data/slat_cache")
    parser.add_argument("--fs-cache-dir", type=str, default="data/slat_cache_facescape")
    parser.add_argument("--output", type=str, default="data/slat_context.lmdb")
    parser.add_argument("--map-size-gb", type=int, default=100,
                        help="LMDB map size in GB (pre-allocated virtual, not physical)")
    args = parser.parse_args()

    # Scan .pt files
    entries = []
    entries.extend(_scan_pt_files(args.fv_cache_dir, "faceverse"))
    entries.extend(_scan_pt_files(args.fs_cache_dir, "facescape"))

    if not entries:
        print("[ERROR] No .pt files found. Run precompute_slat_cache.py first.")
        sys.exit(1)

    print(f"[pack_slat_lmdb] Found {len(entries)} .pt files")
    print(f"[pack_slat_lmdb] Output: {args.output}")

    os.makedirs(args.output, exist_ok=True)
    env = lmdb.open(
        args.output,
        map_size=args.map_size_gb * 1024 * 1024 * 1024,
        sync=False,
        writemap=True,
    )
    txn = env.begin(write=True)

    packed = 0
    errors = 0
    total_bytes = 0
    t0 = time.time()

    for abs_path, lmdb_key, dataset_name in tqdm(entries, desc="Packing"):
        try:
            payload = torch.load(abs_path, map_location="cpu", weights_only=False)
            slat = payload["slat"]
            context = payload["context"]

            # Serialize {slat, context} to bytes
            buf = io.BytesIO()
            torch.save({"slat": slat, "context": context}, buf)
            value = buf.getvalue()

            txn.put(lmdb_key.encode("utf-8"), value)
            packed += 1
            total_bytes += len(value)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] {lmdb_key}: {e}")

        # Commit every 1000 entries to limit memory
        if packed % 1000 == 0 and packed > 0:
            txn.commit()
            txn = env.begin(write=True)

    # Store metadata entry
    import json
    meta = {
        "packed": packed,
        "errors": errors,
        "fv_cache_dir": args.fv_cache_dir,
        "fs_cache_dir": args.fs_cache_dir,
    }
    txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    txn.commit()
    env.sync()
    env.close()

    elapsed = time.time() - t0
    print(f"\n[pack_slat_lmdb] Done in {elapsed:.1f}s")
    print(f"  packed={packed}, errors={errors}")
    print(f"  total_bytes={total_bytes / 1024 / 1024:.1f}MB")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
