#!/usr/bin/env python3
"""
Offline O-Voxel Cache Generator v2.1
====================================
Converts ALL FaceScape meshes to TRUE O-Voxel features.
Saves:
  - coords: [N, 3] int32
    - features: [N, 10] float16 (v, delta, gamma, rgb)
  - aabb: [2, 3] float32 (CRITICAL for reconstruction)
    - resolution: int (voxel grid resolution used for encode/decode)
"""
import argparse
import os
import sys
import glob
import traceback
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Thêm PYTHONPATH
sys.path.append(os.getcwd())


@dataclass
class MeshSidecarStatus:
    has_mtllib_decl: bool
    has_mtl_file: bool
    has_map_kd: bool
    has_texture_file: bool
    mtl_path: str
    texture_path: str


def _validate_data_root(data_root: str, allow_workspace_data_root: bool) -> str:
    root = os.path.abspath(data_root)
    if not os.path.isdir(root):
        raise RuntimeError(f"[OVoxel-Gen] data-root does not exist: {root}")

    ws_data_root = os.path.abspath(os.path.join(os.getcwd(), "data"))
    in_workspace_data = root == ws_data_root or root.startswith(ws_data_root + os.sep)
    if in_workspace_data and not allow_workspace_data_root:
        raise RuntimeError(
            "[OVoxel-Gen] data-root points to workspace data cache folder. "
            "Please use the real mounted dataset path (e.g. /mnt/16TData/Datasets/...). "
            "If this is intentional, pass --allow-workspace-data-root."
        )
    return root


def _inspect_mesh_sidecars(obj_path: str) -> MeshSidecarStatus:
    obj_dir = os.path.dirname(obj_path)
    mtl_path = ""
    has_mtllib_decl = False

    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("mtllib "):
                    has_mtllib_decl = True
                    mtl_name = line.split(maxsplit=1)[1].strip()
                    mtl_path = os.path.join(obj_dir, mtl_name)
                    break
    except Exception:
        pass

    if not mtl_path:
        fallback_mtl = f"{obj_path}.mtl"
        if os.path.exists(fallback_mtl):
            mtl_path = fallback_mtl

    has_mtl_file = bool(mtl_path) and os.path.exists(mtl_path)
    has_map_kd = False
    texture_path = ""

    if has_mtl_file:
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("map_kd "):
                        has_map_kd = True
                        tex_rel = line.split(maxsplit=1)[1].strip()
                        texture_path = os.path.join(obj_dir, tex_rel)
                        break
        except Exception:
            pass

    if not texture_path:
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidate = os.path.join(obj_dir, f"{stem}{ext}")
            if os.path.exists(candidate):
                texture_path = candidate
                break

    has_texture_file = bool(texture_path) and os.path.exists(texture_path)
    return MeshSidecarStatus(
        has_mtllib_decl=has_mtllib_decl,
        has_mtl_file=has_mtl_file,
        has_map_kd=has_map_kd,
        has_texture_file=has_texture_file,
        mtl_path=mtl_path,
        texture_path=texture_path,
    )

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-root", default="/mnt/16TData/Datasets/FaceScape")
    pa.add_argument("--output-dir", default="data/ovoxel_cache_shape_native_v2")
    pa.add_argument("--resolution", type=int, default=256)
    pa.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    pa.add_argument("--max-voxels", type=int, default=350000)
    pa.add_argument(
        "--force-subsample",
        action="store_true",
        help="Allow subsampling to --max-voxels when cache is too dense (can hurt DC topology).",
    )
    pa.add_argument("--skip-existing", action="store_true")
    pa.add_argument("--limit-meshes", type=int, default=-1, help="Stop after N meshes (useful for verification)")
    pa.add_argument("--subject-filter", type=str, default=None, help="Only process directories containing this string")
    pa.add_argument(
        "--allow-workspace-data-root",
        action="store_true",
        help="Allow using workspace ./data as data-root (disabled by default to avoid using cache folders as raw dataset input).",
    )
    pa.add_argument(
        "--strict-sidecar",
        action="store_true",
        help="Require OBJ sidecar files (MTL + texture) for every processed mesh.",
    )
    pa.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    pa.add_argument("--shard-id", type=int, default=0, help="ID of this shard (0 to num-shards-1).")
    pa.add_argument("--file-list", type=str, default=None, help="Path to a text file containing the list of OBJ files to process.")
    args = pa.parse_args()

    args.data_root = _validate_data_root(args.data_root, args.allow_workspace_data_root)

    # Import native converter
    from src.data.ovoxel_converter import OVoxelConverter
    converter = OVoxelConverter(resolution=args.resolution, device=args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optimization: If file-list is provided, load from there. Otherwise glob.
    if args.file_list and os.path.exists(args.file_list):
        print(f"[OVoxel-Gen] Loading OBJ files from list: {args.file_list}")
        with open(args.file_list, "r") as f:
            obj_files = [line.strip() for line in f if line.strip() and line.strip().endswith(".obj")]
        obj_files = sorted(obj_files)
    elif args.subject_filter:
        pattern = os.path.join(args.data_root, f"*{args.subject_filter}*", "**", "*.obj")
        print(f"[OVoxel-Gen] Searching with filter pattern: {pattern}")
        obj_files = sorted(glob.glob(pattern, recursive=True))
    else:
        print(f"[OVoxel-Gen] Scanning directory for OBJ files: {args.data_root} (this may take a minute)...")
        obj_files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.obj"), recursive=True))
    
    total_found = len(obj_files)
    if args.num_shards > 1:
        # Sharding: each shard takes a subset of files
        obj_files = obj_files[args.shard_id :: args.num_shards]
        print(f"[OVoxel-Gen] Sharding enabled: Shard {args.shard_id+1}/{args.num_shards}")
        print(f"[OVoxel-Gen] Processing {len(obj_files)} of {total_found} matching OBJ files.")
    else:
        print(f"[OVoxel-Gen] Found {total_found} matching OBJ files.")

    if len(obj_files) > 0:
        sample_n = min(128, len(obj_files))
        sample_paths = obj_files[:sample_n]
        sample_stats = [_inspect_mesh_sidecars(p) for p in sample_paths]
        mtl_ok = sum(1 for s in sample_stats if s.has_mtl_file)
        tex_ok = sum(1 for s in sample_stats if s.has_texture_file)
        print(
            f"[OVoxel-Gen] Sidecar sample audit (first {sample_n}): "
            f"MTL {mtl_ok}/{sample_n}, texture {tex_ok}/{sample_n}"
        )

    if args.skip_existing:
        print("[OVoxel-Gen] Resume mode enabled (--skip-existing): existing cache files are reused and counted as SkippedExisting.")

    success = 0
    fail = 0
    skipped = 0
    processed = 0

    for obj_path in tqdm(obj_files, desc="Converting"):
        if 0 < args.limit_meshes <= processed:
            print(f"[INFO] Limit reached ({args.limit_meshes}). Stopping.")
            break

        rel = os.path.relpath(obj_path, args.data_root)
        
        # Filtering
        if args.subject_filter and args.subject_filter not in rel:
            continue
        
        print(f"\n[DEBUG] Matched: {rel} | Full Path: {obj_path}")

        sidecar = _inspect_mesh_sidecars(obj_path)
        if args.strict_sidecar and (not sidecar.has_mtl_file or not sidecar.has_texture_file):
            raise RuntimeError(
                "Missing sidecar asset under --strict-sidecar for "
                f"{obj_path}. mtl={sidecar.has_mtl_file}, texture={sidecar.has_texture_file}"
            )

        safe_name = rel.replace("/", "_").replace("\\", "_").replace(".obj", "")
        # Suffix updated to c10 (10 channels: 7 shape + 3 mat)
        suffix = f".c10.shape_mat.mx{args.max_voxels}.pt"
        cache_path = out_dir / f"{safe_name}{suffix}"

        if args.skip_existing and cache_path.exists():
            skipped += 1
            processed += 1
            continue

        try:
            # Using the unified 10-channel converter
            result = converter.process_mesh(obj_path)
            
            coords = result["coords"]
            features = result["shape_mat_features"]
            aabb = result["aabb"]

            if features.ndim != 2 or features.shape[1] != 10:
                raise RuntimeError(
                    f"Invalid feature shape for {safe_name}: expected [N, 10], got {tuple(features.shape)}"
                )
            if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
                raise RuntimeError(f"Invalid coords for {safe_name}: got {tuple(coords.shape)}")

            delta_sum = float(features[:, 3:6].to(torch.float32).abs().sum().item())
            if delta_sum <= 0.0:
                raise RuntimeError(
                    f"Invalid dual-grid cache for {safe_name}: delta channels are all zero"
                )

            # Topology note:
            # Randomly dropping active voxels breaks dual-grid connectivity and can
            # produce blocky/sparse meshes. Keep full cache by default.
            n = features.shape[0]
            if args.max_voxels > 0 and n > args.max_voxels:
                if args.force_subsample:
                    rng = np.random.RandomState(hash(safe_name) % (2**31))
                    indices = rng.choice(n, args.max_voxels, replace=False)
                    indices.sort()
                    indices = torch.from_numpy(indices)
                    features = features[indices]
                    coords = coords[indices]
                else:
                    print(
                        f"[WARN] {safe_name}: n_vox={n} > max_voxels={args.max_voxels}. "
                        "Keeping full cache to preserve dual-contouring topology. "
                        "Use --force-subsample to override."
                    )

            payload = {
                "features": features.to(torch.float32),
                "coords": coords,
                "aabb": aabb.cpu(),
                "resolution": int(args.resolution),
                "norm_params": result.get("norm_params")
            }

            torch.save(payload, str(cache_path))
            success += 1
            processed += 1

        except Exception as e:
            fail += 1
            processed += 1
            if fail <= 50: # Increased limit for large dataset
                 traceback.print_exc()

    print(
        f"\n[OVoxel-Gen] Done! Processed={processed}, NewlyCached={success}, "
        f"Fail={fail}, SkippedExisting={skipped}"
    )
    print(f"Results in {out_dir}")

if __name__ == "__main__":
    main()
