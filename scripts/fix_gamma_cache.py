#!/usr/bin/env python3
"""
Fix Gamma Bug in O-Voxel Cache
===============================
The disk cache (.pt files) was generated with old converter code that set
gamma=1.0 constant. The correct formula is:
    gamma = (1 - var(dv_local, dim=1)).clamp(0, 1)

This script reads all cached .pt files, recomputes gamma from the stored
dv_local values, and fixes aabb to the standard [-0.5, 0.5]³.

Usage:
    python scripts/fix_gamma_cache.py --cache-dir data/ovoxel_cache_recached --dry-run
    python scripts/fix_gamma_cache.py --cache-dir data/ovoxel_cache_recached
"""
import argparse
import os
import glob
import torch
from tqdm import tqdm


def fix_payload(payload: dict) -> dict:
    """Recompute gamma from dv_local and fix aabb."""
    feat = payload["features"]
    if feat.ndim != 2 or feat.shape[1] < 10:
        return payload  # Skip non-10ch

    # Recompute gamma from dv_local (channels 0:3)
    dv_local = feat[:, 0:3].float()
    dv_var = dv_local.var(dim=1, keepdim=True)
    gamma_fixed = (1.0 - dv_var).clamp(0.0, 1.0)

    # Write back gamma (channel 6)
    feat = feat.clone()
    feat[:, 6:7] = gamma_fixed.to(feat.dtype)
    payload["features"] = feat

    # Fix aabb to standard o-voxel grid
    payload["aabb"] = torch.tensor(
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32
    )

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/ovoxel_cache_recached")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't write")
    parser.add_argument("--limit", type=int, default=-1, help="Process only N files (for testing)")
    args = parser.parse_args()

    pt_files = sorted(glob.glob(os.path.join(args.cache_dir, "**", "*.pt"), recursive=True))
    print(f"Found {len(pt_files)} .pt files in {args.cache_dir}")

    if args.limit > 0:
        pt_files = pt_files[: args.limit]
        print(f"Limited to {len(pt_files)} files")

    fixed = 0
    skipped = 0
    errors = 0
    gamma_stats = {"old_min": 1e9, "old_max": -1e9, "new_min": 1e9, "new_max": -1e9}

    for pt_path in tqdm(pt_files, desc="Fixing gamma"):
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)

            if not isinstance(payload, dict) or "features" not in payload:
                skipped += 1
                continue

            feat = payload["features"]
            if feat.ndim != 2 or feat.shape[1] < 10:
                skipped += 1
                continue

            # Track old gamma
            old_gamma = feat[:, 6].float()
            gamma_stats["old_min"] = min(gamma_stats["old_min"], old_gamma.min().item())
            gamma_stats["old_max"] = max(gamma_stats["old_max"], old_gamma.max().item())

            # Fix
            payload = fix_payload(payload)

            # Track new gamma
            new_gamma = payload["features"][:, 6].float()
            gamma_stats["new_min"] = min(gamma_stats["new_min"], new_gamma.min().item())
            gamma_stats["new_max"] = max(gamma_stats["new_max"], new_gamma.max().item())

            if not args.dry_run:
                torch.save(payload, pt_path)

            fixed += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n[ERROR] {pt_path}: {e}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Results:")
    print(f"  Fixed: {fixed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Gamma old: [{gamma_stats['old_min']:.6f}, {gamma_stats['old_max']:.6f}]")
    print(f"  Gamma new: [{gamma_stats['new_min']:.6f}, {gamma_stats['new_max']:.6f}]")


if __name__ == "__main__":
    main()
