"""
Trình tải Tập dữ liệu FaceScape (FaceScape Dataset Loader)
========================
Trình tải (Loader) cho tập dữ liệu FaceScape (847 danh tính × 20 biểu cảm).
Cấu trúc: Lưới đồng nhất về mặt tô pô (Topologically Uniform - TU meshes) — tất cả các lưới chia sẻ cùng một topology.
Đặc điểm: Không có mắt và vai (khác với FaceVerse).

Định dạng (Format): {identity}_{expression}.obj + .mtl + ảnh dịch chuyển (displacement) .png + kết cấu (texture) .jpg
"""

import os
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, ConcatDataset


class FaceScapeDataset(Dataset):
    """
    PyTorch Dataset cho các lưới khuôn mặt 3D (3D face meshes) của FaceScape.
    
    Cấu trúc thư mục kỳ vọng:
        root_dir/
        ├── models_reg/
        │   ├── 1_neutral.obj
        │   ├── 1_smile.obj
        │   └── ...
        └── textures/
            ├── 1_neutral.jpg
            └── ...
    
    Hoặc dạng phẳng (flat):
        root_dir/
        ├── 001_01.obj
        ├── 001_01.mtl
        ├── 001_01.jpg
        └── ...
    """
    
    def __init__(self, root_dir: str, scan_mode: str = "auto"):
        """
        Tham số:
            root_dir: Đường dẫn gốc tới tập dữ liệu FaceScape
            scan_mode: 'auto' (tự động phát hiện), 'flat' (cùng thư mục), 'nested' (thư mục con - sub-dirs)
        """
        super().__init__()
        self.root_dir = root_dir
        self.samples: List[Dict[str, Any]] = []
        
        if not os.path.isdir(root_dir):
            print(f"[FaceScape] Warning: Directory not found: {root_dir}")
            return
        
        if scan_mode == "auto":
            # Tự động phát hiện (detect) cấu trúc
            self.samples = self._scan_auto()
        elif scan_mode == "flat":
            self.samples = self._scan_flat()
        else:
            self.samples = self._scan_nested()
        
        print(f"[FaceScape] Loaded {len(self.samples)} meshes from {root_dir}")
    
    def _scan_auto(self) -> List[Dict[str, Any]]:
        """Tự động phát hiện (detect) cấu trúc thư mục."""
        # Thử dạng phẳng (flat) trước
        samples = self._scan_flat()
        if len(samples) > 0:
            return samples
        # Thử dạng lồng nhau (nested)
        samples = self._scan_nested()
        return samples
    
    def _scan_flat(self) -> List[Dict[str, Any]]:
        """Quét thư mục phẳng (tất cả các file nằm cùng một cấp)."""
        samples = []
        obj_files = sorted([
            f for f in os.listdir(self.root_dir)
            if f.endswith('.obj') and not f.startswith('.')
        ])
        
        for obj_file in obj_files:
            basename = obj_file.replace('.obj', '')
            obj_path = os.path.join(self.root_dir, obj_file)
            mtl_path = os.path.join(self.root_dir, f"{basename}.mtl")
            jpg_path = os.path.join(self.root_dir, f"{basename}.jpg")
            png_path = os.path.join(self.root_dir, f"{basename}.png")
            
            samples.append({
                "id": basename,
                "dataset": "facescape",
                "obj_path": obj_path,
                "mtl_path": mtl_path if os.path.exists(mtl_path) else None,
                "img_path": jpg_path if os.path.exists(jpg_path) else (
                    png_path if os.path.exists(png_path) else None
                ),
            })
        
        return samples
    
    def _scan_nested(self) -> List[Dict[str, Any]]:
        """Quét thư mục lồng nhau (nested) (1/models_reg/*.obj)."""
        samples = []
        subdirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.root_dir, subdir)
            models_reg_dir = os.path.join(subdir_path, 'models_reg')
            dpmap_dir = os.path.join(subdir_path, 'dpmap')
            
            if not os.path.isdir(models_reg_dir):
                continue
                
            obj_files = [f for f in os.listdir(models_reg_dir) if f.endswith('.obj')]
            
            for obj_file in obj_files:
                basename = obj_file.replace('.obj', '')
                obj_path = os.path.join(models_reg_dir, obj_file)
                mtl_path = os.path.join(models_reg_dir, f"{basename}.obj.mtl")
                if not os.path.exists(mtl_path):
                    mtl_path = os.path.join(models_reg_dir, f"{basename}.mtl")
                
                # Tìm kết cấu (texture) (.jpg hoặc .png)
                img_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = os.path.join(models_reg_dir, f"{basename}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                
                # Tìm bản đồ dịch chuyển (displacement map) (dpmap)
                dpmap_path = None
                if os.path.isdir(dpmap_dir):
                    for ext in ['.png', '.tif', '.jpg']:
                        candidate_dp = os.path.join(dpmap_dir, f"{basename}{ext}")
                        if os.path.exists(candidate_dp):
                            dpmap_path = candidate_dp
                            break
                            
                samples.append({
                    "id": f"facescape_{subdir}_{basename}",
                    "dataset": "facescape",
                    "obj_path": obj_path,
                    "mtl_path": mtl_path if os.path.exists(mtl_path) else None,
                    "img_path": img_path,
                    "dpmap_path": dpmap_path,
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def create_mixed_dataset(
    faceverse_root: str = "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse",
    facescape_root: str = "/mnt/16TData/Datasets/FaceScape",
) -> ConcatDataset:
    """
    Tạo Mixed DataLoader kết hợp FaceVerse và FaceScape.
    
    Trả về:
        ConcatDataset chứa cả 2 tập dữ liệu (datasets), sẵn sàng cho DataLoader
    """
    from src.data.faceverse_dataset import FaceVerseDataset
    
    fv_ds = FaceVerseDataset(faceverse_root)
    fs_ds = FaceScapeDataset(facescape_root)
    
    print(f"[Mixed] FaceVerse: {len(fv_ds)} | FaceScape: {len(fs_ds)} | "
          f"Total: {len(fv_ds) + len(fs_ds)}")
    
    return ConcatDataset([fv_ds, fs_ds])


if __name__ == "__main__":
    # Kiểm thử độc lập (Test standalone)
    print("=" * 50)
    print(" FaceScape Dataset Test")
    print("=" * 50)
    
    # Kiểm thử với đường dẫn (path) mặc định
    facescape_root = "/mnt/16TData/Datasets/FaceScape"
    ds = FaceScapeDataset(facescape_root)
    print(f"Samples: {len(ds)}")
    if len(ds) > 0:
        print(f"Sample 0: {ds[0]}")
    
    print("\n✅ FaceScape Dataset Loader hoạt động!")
