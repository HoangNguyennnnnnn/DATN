#!/usr/bin/env python3
"""
Fill missing slat+context entries by borrowing context from same identity's neutral expression.
Encodes meshes with SC-VAE, then appends to slat_context.lmdb.

Usage:
    python scripts/fill_missing_slat.py \
        --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
        --slat-lmdb data/slat_context.lmdb \
        --context-lmdb data/hybrid_context.lmdb \
        --ovoxel-lmdb data/ovoxel_cache_lmdb
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmdb
import torch

# Missing meshes (from precompute error log)
MISSING_MESHES = [
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/13_lip_funneler.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/14_sadness.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/15_lip_roll.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/16_grin.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/17_cheek_blowing.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/18_eye_closed.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/19_brow_raiser.obj",
    "/mnt/16TData/Datasets/FaceScape/148/models_reg/20_brow_lower.obj",
    "/mnt/16TData/Datasets/FaceScape/452/models_reg/18_eye_closed.obj",
    "/mnt/16TData/Datasets/FaceScape/488/models_reg/8_mouth_left.obj",
    "/mnt/16TData/Datasets/FaceScape/510/models_reg/8_mouth_left.obj",
    "/mnt/16TData/Datasets/FaceScape/655/models_reg/13_lip_funneler.obj",
    "/mnt/16TData/Datasets/FaceScape/655/models_reg/20_brow_lower.obj",
    "/mnt/16TData/Datasets/FaceScape/66/models_reg/3_mouth_stretch.obj",
    "/mnt/16TData/Datasets/FaceScape/facescape_trainset_001_100-004/66/models_reg/3_mouth_stretch.obj",
]

FACESCAPE_ROOT = "/mnt/16TData/Datasets/FaceScape"


def get_identity_dir(obj_path: str) -> str:
    """Extract identity directory (e.g., .../148/models_reg/) from obj path."""
    return os.path.dirname(obj_path)


def get_identity_id(obj_path: str) -> str:
    """Extract identity number from path like .../148/models_reg/..."""
    parts = obj_path.split("/")
    for i, p in enumerate(parts):
        if p == "models_reg" and i > 0:
            return parts[i - 1]
    return "unknown"


def find_donor_context_key(identity_id: str) -> str:
    """Build the LMDB key for the neutral expression of this identity."""
    return f"facescape/{identity_id}/models_reg/1_neutral.obj"


def get_rel_path(obj_path: str) -> str:
    """Get relative path from FaceScape root for LMDB key."""
    return os.path.relpath(obj_path, FACESCAPE_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-vae-ckpt", required=True)
    parser.add_argument("--slat-lmdb", default="data/slat_context.lmdb")
    parser.add_argument("--context-lmdb", default="data/hybrid_context.lmdb")
    parser.add_argument("--ovoxel-lmdb", default="data/ovoxel_cache_lmdb")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- Load SC-VAE ---
    print("[fill_missing] Loading SC-VAE...")
    from src.models.sc_vae import SCVAE
    ckpt = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    sc_vae = SCVAE(
        in_channels=model_cfg.get("in_channels", 10),
        latent_dim=model_cfg.get("latent_dim", 32),
        encoder_channels=model_cfg.get("encoder_channels", [64, 128, 256, 512]),
        decoder_channels=model_cfg.get("decoder_channels", [512, 256, 128, 64]),
    )
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    sc_vae.load_state_dict(state)
    sc_vae = sc_vae.to(device).eval()
    print(f"[fill_missing] SC-VAE loaded on {device}")

    # --- Open O-Voxel LMDB ---
    print("[fill_missing] Opening O-Voxel LMDB...")
    ovoxel_env = lmdb.open(args.ovoxel_lmdb, readonly=True, lock=False,
                           readahead=True, meminit=False, max_readers=512)
    ovoxel_txn = ovoxel_env.begin(write=False)

    # --- Open Context LMDB (for borrowing) ---
    print("[fill_missing] Opening Context LMDB...")
    ctx_env = lmdb.open(args.context_lmdb, readonly=True, lock=False,
                        readahead=True, meminit=False, max_readers=512)
    ctx_txn = ctx_env.begin(write=False)

    # --- Open Slat+Context LMDB (for writing) ---
    print("[fill_missing] Opening Slat LMDB for writing...")
    slat_env = lmdb.open(args.slat_lmdb, map_size=100 * 1024**3,
                         sync=False, writemap=True)
    slat_txn = slat_env.begin(write=True)

    filled = 0
    skipped = 0
    errors = 0

    for obj_path in MISSING_MESHES:
        identity_id = get_identity_id(obj_path)
        rel_path = get_rel_path(obj_path)
        slat_key = f"facescape/{rel_path}"

        # Check if already in slat LMDB
        if slat_txn.get(slat_key.encode()) is not None:
            print(f"  [SKIP] Already in LMDB: {slat_key}")
            skipped += 1
            continue

        # 1) Borrow context from neutral expression
        donor_key = find_donor_context_key(identity_id)
        ctx_data = ctx_txn.get(donor_key.encode())
        if ctx_data is None:
            # Try other expressions
            fallback_exprs = ["2_smile", "3_mouth_stretch", "4_anger",
                              "5_jaw_left", "10_dimpler", "11_chin_raiser"]
            for expr in fallback_exprs:
                alt_key = f"facescape/{identity_id}/models_reg/{expr}.obj"
                ctx_data = ctx_txn.get(alt_key.encode())
                if ctx_data is not None:
                    donor_key = alt_key
                    break
            if ctx_data is None:
                print(f"  [ERROR] No donor context for identity {identity_id}")
                errors += 1
                continue

        context = torch.load(io.BytesIO(ctx_data), map_location="cpu",
                             weights_only=False).float()
        if context.ndim == 0:
            context = context.unsqueeze(0)
        print(f"  Borrowed context from {donor_key} → {slat_key}")

        # 2) Encode mesh with SC-VAE via O-Voxel LMDB
        # Build ovoxel key: facescape/{rel_path}
        ovoxel_key = f"facescape/{rel_path}".encode()
        ovoxel_data = ovoxel_txn.get(ovoxel_key)
        if ovoxel_data is None:
            # Try alternate key formats
            alt_key = rel_path.encode()
            ovoxel_data = ovoxel_txn.get(alt_key)
        if ovoxel_data is None:
            print(f"  [ERROR] O-Voxel not found in LMDB for {rel_path}")
            errors += 1
            continue

        try:
            ovoxel_payload = torch.load(io.BytesIO(ovoxel_data), map_location="cpu",
                                        weights_only=False)
            # Extract features and coords
            if isinstance(ovoxel_payload, dict):
                features = ovoxel_payload.get("features") or ovoxel_payload.get("feats")
                coords = ovoxel_payload.get("coords") or ovoxel_payload.get("coordinates")
            else:
                features, coords = ovoxel_payload[0], ovoxel_payload[1]

            features = features.float()
            coords = coords.int()

            # Pad/slice to 10 channels if needed
            if features.shape[-1] < 10:
                pad = torch.zeros(features.shape[0], 10 - features.shape[-1])
                features = torch.cat([features, pad], dim=-1)
            elif features.shape[-1] > 10:
                features = features[:, :10]

            # Encode with SC-VAE
            import spconv.pytorch as spconv
            N = coords.shape[0]
            batch_idx = torch.zeros(N, 1, dtype=torch.int32)
            coords_batch = torch.cat([batch_idx, coords], dim=1)

            sp_tensor = spconv.SparseConvTensor(
                features=features.to(device),
                indices=coords_batch.to(device),
                spatial_shape=[256, 256, 256],
                batch_size=1,
            )

            with torch.no_grad():
                enc_out = sc_vae.encode(sp_tensor)
                mu = enc_out["mu"]  # [1, N_latent, latent_dim]
                slat = mu.squeeze(0).cpu()  # [4096, 32]

            # 3) Pack into slat LMDB
            buf = io.BytesIO()
            torch.save({"slat": slat, "context": context}, buf)
            slat_txn.put(slat_key.encode(), buf.getvalue())
            filled += 1
            print(f"  [OK] {slat_key} — slat={list(slat.shape)}, ctx={list(context.shape)}")

        except Exception as e:
            print(f"  [ERROR] Encoding {rel_path}: {e}")
            errors += 1

    # Update meta
    import json
    old_meta = slat_txn.get(b"__meta__")
    if old_meta:
        meta = json.loads(old_meta.decode())
    else:
        meta = {}
    meta["filled_missing"] = filled
    meta["packed"] = meta.get("packed", 0) + filled
    slat_txn.put(b"__meta__", json.dumps(meta).encode())

    slat_txn.commit()
    slat_env.sync()
    slat_env.close()
    ctx_env.close()
    ovoxel_env.close()

    print(f"\n[fill_missing] Done: filled={filled}, skipped={skipped}, errors={errors}")
    print(f"  Total entries in LMDB should now be: {20370 + filled}")


if __name__ == "__main__":
    main()
