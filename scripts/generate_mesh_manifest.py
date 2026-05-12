#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Tạo manifest danh sách mesh để train offline không cần file .obj")
    parser.add_argument("--faceverse-root", type=str, default="/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse")
    parser.add_argument("--facescape-root", type=str, default="/mnt/16TData/Datasets/FaceScape")
    parser.add_argument("--out", type=str, default="data/mesh_manifest.json")
    args = parser.parse_args()

    manifest = {
        "faceverse": [],
        "facescape": []
    }

    # Quét FaceVerse
    if os.path.isdir(args.faceverse_root):
        print(f"Scanning FaceVerse: {args.faceverse_root}")
        for root, _, files in os.walk(args.faceverse_root):
            for f in files:
                if f.endswith(".obj"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, args.faceverse_root)
                    manifest["faceverse"].append(rel_path)
        print(f"  Found {len(manifest['faceverse'])} meshes.")

    # Quét FaceScape
    if os.path.isdir(args.facescape_root):
        print(f"Scanning FaceScape: {args.facescape_root}")
        for root, _, files in os.walk(args.facescape_root):
            for f in files:
                if f.endswith(".obj"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, args.facescape_root)
                    manifest["facescape"].append(rel_path)
        print(f"  Found {len(manifest['facescape'])} meshes.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Manifest saved to: {args.out}")
    print("Bạn chỉ cần mang file JSON này lên Cloud GPU cùng với các file LMDB.")

if __name__ == "__main__":
    main()
