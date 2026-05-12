#!/usr/bin/env python3
"""
Tạo hybrid_context.lmdb chứa các vector đặc trưng cho VoxelMamba.
Script này chạy qua tất cả các file .obj, render 2 mặt, và trích xuất:
- ArcFace (Identity): [1, 512]
- FLAME (Expression): [1, 50]
- DINOv2 (Shape/Hair): [1, 384]
Ghép lại thành vector [946] và lưu vào LMDB với key là đường dẫn tuyệt đối của file .obj.
"""

import os
import io
import argparse
from tqdm import tqdm
import torch
import lmdb

from src.data.mesh_renderer import MeshRenderer
from src.data.arcface_extractor import ArcFaceExtractor
from src.data.flame_adapter import FLAMEExpressionAdapter
from src.data.feature_extractor import DinoV3Extractor

def find_obj_files(directories):
    obj_files = []
    for d in directories:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.obj'):
                    obj_files.append(os.path.join(root, f))
    return sorted(obj_files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-lmdb", type=str, default="/mnt/16TData/hybrid_context.lmdb")
    parser.add_argument("--dirs", nargs="+", default=[
        "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse",
        "/mnt/16TData/Datasets/FaceScape"
    ])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    obj_files = find_obj_files(args.dirs)
    print(f"Found {len(obj_files)} .obj files.")
    if len(obj_files) == 0:
        return

    # Khởi tạo mô hình
    print("Loading models...")
    renderer = MeshRenderer(device=args.device, image_size=512)
    arcface = ArcFaceExtractor(device=args.device)
    flame = FLAMEExpressionAdapter(expression_dim=50, device=args.device)
    dino = DinoV3Extractor(model_name="facebook/dinov2-small", device=args.device)

    # Khởi tạo LMDB (dung lượng map_size lớn một chút, 50GB, nhưng nó sẽ chỉ dùng đúng kích thước thực tế)
    env = lmdb.open(args.out_lmdb, map_size=50 * 1024 * 1024 * 1024)

    success_count = 0
    fail_count = 0
    batch_size = 500

    txn = env.begin(write=True)
    for i, obj_path in enumerate(tqdm(obj_files, desc="Building Context")):
        # Tạo portable key: 'dataset_name/relative_path'
        dataset_name = "unknown"
        rel_path = obj_path
        for d in args.dirs:
            if obj_path.startswith(d):
                dataset_name = os.path.basename(d).lower()
                if dataset_name == "faceverse_3d": # Chuẩn hóa tên folder
                    dataset_name = "faceverse"
                rel_path = os.path.relpath(obj_path, d)
                break
        
        key = f"{dataset_name}/{rel_path}".encode('utf-8')
        
        if txn.get(key) is not None:
            success_count += 1
            continue

        try:
            # 1. Render
            front, back = renderer.render_front_and_back(obj_path)
            
            # 2. Extract
            id_vec = arcface.extract_identity(front)          # [1, 512]
            exp_vec = flame.extract_from_image(front)         # [1, 50]
            
            # DINOv2
            shape_vec = dino.extract_features(back)           # [1, 384]
            
            # 3. Ghép vector [1, 946]
            context_vec = torch.cat([id_vec, exp_vec, shape_vec], dim=-1).squeeze(0).cpu() # [946]
            
            # Lưu vào bytes
            buffer = io.BytesIO()
            torch.save(context_vec.half(), buffer) # Lưu dạng float16
            
            txn.put(key, buffer.getvalue())
            success_count += 1
            
        except Exception as e:
            print(f"Failed {obj_path}: {e}")
            fail_count += 1

        if (i + 1) % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    print(f"Done! Success: {success_count}, Failed: {fail_count}")
    print(f"LMDB saved at {args.out_lmdb}")


if __name__ == "__main__":
    main()
