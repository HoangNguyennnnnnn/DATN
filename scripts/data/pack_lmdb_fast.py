import os
import sys
import lmdb
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

sys.path.insert(0, '/mnt/18TData/facediff')
from src.config import TrainConfig
from src.scvae_train.data import VoxelDataset

cfg = TrainConfig()
lmdb_dir = "data/ovoxel_cache_lmdb_new"

# Xóa LMDB cũ nếu có
if os.path.exists(lmdb_dir):
    print(f"Removing old LMDB: {lmdb_dir}")
    shutil.rmtree(lmdb_dir)
os.makedirs(lmdb_dir, exist_ok=True)

# Khởi tạo dataset để lấy list obj paths
ds = VoxelDataset(
    data_root=cfg.data.facescape_root,
    dataset_name="facescape",
    feature_mode="shape_mat",
    target_in_channels=10,
    max_voxels=350000,
    lmdb_dir=None,
)

ds_fv = VoxelDataset(
    data_root=cfg.data.faceverse_root,
    dataset_name="faceverse",
    feature_mode="shape_mat",
    target_in_channels=10,
    max_voxels=350000,
    lmdb_dir=None,
)

all_samples = []
for i in range(len(ds)):
    all_samples.append((ds.samples[i], ds.data_root))
for i in range(len(ds_fv)):
    all_samples.append((ds_fv.samples[i], ds_fv.data_root))

print(f"Total samples to pack: {len(all_samples)}")

# Tạo LMDB (map_size 200GB)
env = lmdb.open(lmdb_dir, map_size=400 * 1024 * 1024 * 1024, sync=False, writemap=True)
txn = env.begin(write=True)

cache_dir = "data/ovoxel_cache_recached"
packed = 0
missing = 0

print("Packing LMDB...")
for i, (obj_path, data_root) in enumerate(tqdm(all_samples)):
    rel = os.path.relpath(obj_path, data_root)
    # Tên file cho max_voxels=350000
    safe_name = rel.replace(os.sep, '_').replace('.obj', '.c10.shape_mat.mx350000.pt')
    
    # Tìm file trong cache (thường là trong cache_dir/facescape hoặc cache_dir/faceverse)
    # VoxelDataset có hàm _get_cache_paths, ta tự implement đơn giản:
    dataset_folder = "facescape" if "FaceScape" in data_root else "faceverse"
    pt_path = os.path.join(cache_dir, dataset_folder, safe_name)
    
    if os.path.exists(pt_path):
        with open(pt_path, "rb") as f:
            data = f.read()
        # Lưu vào LMDB với key là tên an toàn
        txn.put(safe_name.encode("utf-8"), data)
        packed += 1
    else:
        missing += 1
        
    # Commit mỗi 1000 file để giảm RAM usage
    if (i + 1) % 1000 == 0:
        txn.commit()
        txn = env.begin(write=True)

# Final commit
txn.commit()
env.close()

print(f"\n[DONE] Packed: {packed} | Missing: {missing}")

if packed > 0:
    print(f"\nReplacing old LMDB directory...")
    old_lmdb = "data/ovoxel_cache_lmdb"
    if os.path.exists(old_lmdb):
        shutil.rmtree(old_lmdb)
    os.rename(lmdb_dir, old_lmdb)
    print(f"Success! LMDB is ready at {old_lmdb}")
