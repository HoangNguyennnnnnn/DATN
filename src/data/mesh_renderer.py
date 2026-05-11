import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
import numpy as np
from typing import Tuple
import trimesh
import pyrender
from PIL import Image

def get_look_at(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat

class MeshRenderer:
    """
    Kết xuất lưới 3D (3D meshes) thành ảnh 2D (mặt trước và mặt sau) để cung cấp đầu vào 
    cho DINOv3 cho mục đích điều kiện hóa qua Cross-Attention.
    Được triển khai lại bằng Pyrender để đảm bảo chất lượng kết xuất màu và tránh các lỗi biên dịch của PyTorch3D.
    """
    def __init__(self, device: str = "cuda:0", image_size: int = 512):
        self.device = torch.device(device)
        self.image_size = image_size
        
    def render_front_and_back(self, obj_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing OBJ file: {obj_path}")

        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / (1024**2) if self.device.type == 'cuda' else 0

        mesh = trimesh.load(obj_path, process=True)
        
        # Xoay lưới FaceVerse 180 độ quanh trục X (vì lưới bị chổng ngược và lật mặt)
        is_faceverse = "FaceVerse" in obj_path or "faceverse" in obj_path.lower()
        if is_faceverse:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
            
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        # Tải kết cấu (texture) theo cách thủ công
        img_path = obj_path.replace(".obj", ".jpg")
        img = Image.open(img_path).convert("RGB")
        tex = pyrender.Texture(source=img, source_channels='RGB')
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0, alphaMode='OPAQUE'
        )
        for prim in py_mesh.primitives:
            prim.material = mat
            
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        scene.add(py_mesh)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        dist = max(bounds[1] - bounds[0]) * 1.5
        
        up = np.array([0, 1, 0])
        
        # Thiết lập góc nhìn mặt trước (Front Setup)
        eye_front = center + np.array([0, 0, dist])
        cam_node = scene.add(camera, pose=get_look_at(eye_front, center, up))
        
        r = pyrender.OffscreenRenderer(self.image_size, self.image_size)
        color_front, _ = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
        
        # Thiết lập góc nhìn mặt sau (Back Setup)
        scene.remove_node(cam_node)
        eye_back = center + np.array([0, 0, -dist])
        scene.add(camera, pose=get_look_at(eye_back, center, up))
        
        color_back, _ = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
        
        # Dọn dẹp bộ nhớ (Memory Cleanup)
        r.delete()
        del scene, mesh, py_mesh, img
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / (1024**2) if self.device.type == 'cuda' else 0
        
        print(f"[MeshRenderer] Memory Delta: {mem_after - mem_before:.2f} MB")
        
        # Chuyển đổi thành định dạng torch tensor [1, 3, H, W]
        front_tensor = torch.from_numpy(color_front[..., :3].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        back_tensor = torch.from_numpy(color_back[..., :3].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        return front_tensor.to(self.device), back_tensor.to(self.device)
