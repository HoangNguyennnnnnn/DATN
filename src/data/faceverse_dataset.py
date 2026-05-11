import os
import glob
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset

class FaceVerseDataset(Dataset):
    """
    PyTorch Dataset cho các lưới (meshes) FaceVerse_3D.
    Tập dữ liệu chứa 2310 thư mục (110 danh tính x 21 biểu cảm),
    mỗi thư mục chứa một tệp .obj, .mtl, và tệp kết cấu (texture) .jpg.
    
    Lớp này cung cấp các đường dẫn tệp một cách lười biếng (lazily) để tránh tải tất cả 2310 
    lưới vào RAM CPU cùng một lúc, giúp bảo toàn bộ nhớ.
    """
    def __init__(self, root_dir: str = "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse"):
        """
        Tham số:
            root_dir (str): Đường dẫn đến thư mục gốc của tập dữ liệu FaceVerse.
        """
        super().__init__()
        self.root_dir = root_dir
        self.samples: List[Dict[str, str]] = self._scan_dataset()
        
    def _scan_dataset(self) -> List[Dict[str, str]]:
        """
        Quét thư mục gốc và thu thập đường dẫn cho các tệp .obj, .mtl, và .jpg.
        """
        samples = []
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            
        # Cấu trúc của tập dữ liệu là root_dir/<id_expr>/<id_expr>.[obj|mtl|jpg]
        subdirs = sorted([
            os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        
        for subdir in subdirs:
            basename = os.path.basename(subdir)
            obj_path = os.path.join(subdir, f"{basename}.obj")
            mtl_path = os.path.join(subdir, f"{basename}.mtl")
            img_path = os.path.join(subdir, f"{basename}.jpg")
            
            # Đảm bảo ít nhất tệp obj tồn tại trước khi thêm nó vào danh sách các mẫu
            if os.path.exists(obj_path):
                samples.append({
                    "id": basename,
                    "obj_path": obj_path,
                    "mtl_path": mtl_path if os.path.exists(mtl_path) else None,
                    "img_path": img_path if os.path.exists(img_path) else None
                })
                
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Trả về đường dẫn dữ liệu cho một lưới cụ thể.
        Việc tải lưới bổ sung (ví dụ: thông qua trimesh hoặc pytorch3d) nên được thực hiện 
        hoặc trong một hàm transform hoặc trực tiếp trong vòng lặp DataLoader để
        tối ưu hóa việc sử dụng bộ nhớ và giữ cho mức tiêu thụ VRAM ở mức thấp.
        """
        return self.samples[idx]

if __name__ == "__main__":
    # Logic kiểm thử đơn giản để xác minh tập dữ liệu
    dataset = FaceVerseDataset()
    print(f"Loaded {len(dataset)} samples from FaceVerse_3D.")
    if len(dataset) > 0:
        print(f"Sample 0: {dataset[0]}")
