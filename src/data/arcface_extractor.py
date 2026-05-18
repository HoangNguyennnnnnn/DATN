"""
Bộ trích xuất Danh tính ArcFace (ArcFace Identity Extractor)
===========================
Thay thế DINOv2 bằng ArcFace để trích xuất vector danh tính khuôn mặt 512 chiều (512-dim).
ArcFace sử dụng hàm mất mát Additive Angular Margin Loss, tạo ra không gian đặc trưng có tính
phân tách danh tính cao giữa các lớp (high inter-class separability) và độ tập trung cao trong cùng một lớp (high intra-class compactness).

Kiến trúc: Mạng cơ sở (backbone) ResNet-100 hoặc ResNet-50 thông qua thư viện insightface.
Dự phòng (Fallback): Nếu chưa cài đặt insightface, sử dụng mạng ResNet đã huấn luyện trước (pretrained) từ torchvision
với khối chiếu (projection head) 512 chiều.

VRAM: ~50-100 MB (nhẹ hơn DINOv2 186 MB).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ============================================================
# Cố gắng import insightface (dựa trên ONNX, tiêu chuẩn sản phẩm (production-grade))
# ============================================================
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

# Dự phòng (Fallback): torchvision ResNet
TORCHVISION_AVAILABLE = False
try:
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass


class ArcFaceExtractor:
    """
    Trích xuất vector danh tính khuôn mặt 512 chiều từ ảnh 2D.
    
    So với DINOv2 (1536 chiều được ghép nối - concat, đa mục đích - general-purpose):
    - ArcFace: 512 chiều, chuyên biệt cho nhận dạng khuôn mặt (face identity) → không bị sai lệch danh tính (identity drift)
    - Hàm mất mát Angular margin → phân tách danh tính siêu cao 
    - Nhỏ hơn 3 lần nhưng chính xác hơn nhiều lần cho nhiệm vụ tạo sinh khuôn mặt (face generation)
    
    Luồng xử lý (Pipeline): Ảnh chính diện (Front image) → ArcFace → Vector danh tính 512 chiều
    (Không cần góc nhìn phía sau - back view - vì ArcFace chỉ cần 1 ảnh chính diện là đủ mã hóa danh tính)
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "buffalo_l",  # insightface model pack
        det_size: int = 512,
    ):
        self.device = torch.device(device)
        self.embedding_dim = 512  # Số chiều tiêu chuẩn của ArcFace = 512 chiều
        
        if INSIGHTFACE_AVAILABLE:
            print(f"[ArcFace] Khởi tạo InsightFace model pack: {model_name}")
            self.mode = "insightface"
            self.use_cuda_provider = self.device.type == "cuda" and torch.cuda.is_available()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_cuda_provider else ['CPUExecutionProvider']
            ctx_id = 0 if self.use_cuda_provider else -1
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
            self.providers = list(providers)
            provider_desc = " + ".join(providers)
            print(f"[ArcFace] InsightFace đã sẵn sàng (providers={provider_desc})")
        elif TORCHVISION_AVAILABLE:
            print(f"[ArcFace] Không tìm thấy InsightFace, sử dụng dự phòng ResNet-50")
            self.mode = "resnet_fallback"
            self._build_resnet_fallback()
            self.providers = ["torchvision"]
        else:
            print(f"[ArcFace] Không có thư viện nào, sử dụng giá trị nhúng ngẫu nhiên (random embedding)")
            self.mode = "mock"
            self.providers = ["mock"]
    
    def _build_resnet_fallback(self):
        """
        Xây dựng bộ trích xuất dựa trên mạng ResNet-50 đã huấn luyện trước + khối chiếu (projection head) 512 chiều.
        Không mạnh bằng ArcFace thật nhưng đủ dùng cho luồng xử lý kiểm thử (testing pipeline).
        """
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Loại bỏ lớp FC cuối cùng (1000 phân lớp) → bộ trích xuất đặc trưng 2048 chiều
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        # Phép chiếu (Projection): 2048 → 512 (kích thước tiêu chuẩn của ArcFace)
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        ).to(self.device).eval()
        
        # Vô hiệu hóa tính toán gradient (chỉ dùng cho suy luận - inference)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False
            
        self.preprocess = transforms.Compose([
            transforms.Resize(112),  # Tiêu chuẩn của ArcFace sử dụng 112×112
            transforms.CenterCrop(112),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        mem = torch.cuda.memory_allocated(self.device) / (1024**2)
        print(f"[ArcFace] ResNet-50 Fallback loaded. VRAM: {mem:.1f} MB")
    
    @torch.no_grad()
    def extract_identity(
        self, 
        front_img: torch.Tensor,
        back_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Trích xuất vector danh tính 512 chiều từ ảnh khuôn mặt.
        
        Tham số:
            front_img: Tensor [B, 3, H, W] hoặc [1, 3, H, W], giá trị trong khoảng [0, 1]
            back_img: Không bắt buộc (ArcFace chỉ yêu cầu ảnh mặt trước)
            
        Trả về:
            Tensor [B, 512] — vector danh tính đã được chuẩn hóa L2 (L2-normalize)
        """
        mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)
        
        if self.mode == "insightface":
            embedding = self._extract_insightface(front_img)
        elif self.mode == "resnet_fallback":
            embedding = self._extract_resnet(front_img)
        else:
            # Giả lập (Mock): vector 512 chiều ngẫu nhiên
            batch_size = front_img.shape[0] if len(front_img.shape) == 4 else 1
            embedding = torch.randn(batch_size, 512, device=self.device)
        
        # Chuẩn hóa L2 (L2 normalize) — chiếu lên siêu cầu (hypersphere) (đặc trưng của ArcFace)
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        mem_after = torch.cuda.memory_allocated(self.device) / (1024**2)
        print(f"[ArcFace] Identity vector: {list(embedding.shape)}, "
              f"VRAM Delta: {mem_after - mem_before:.2f} MB")
        
        return embedding
    
    @torch.no_grad()
    def detect_face(self, img_rgb_np: np.ndarray) -> Optional[dict]:
        """Detect khuôn mặt lớn nhất trong ảnh, trả bbox + landmarks + embedding.

        Args:
            img_rgb_np: ảnh numpy RGB uint8 shape [H, W, 3].

        Returns:
            {bbox: [x1, y1, x2, y2], kps: [[x, y]×5], embedding: tensor[512]} hoặc None
            nếu không detect được face (hoặc đang ở chế độ fallback).
        """
        if self.mode != "insightface":
            return None
        img_bgr = img_rgb_np[..., ::-1].copy()
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = torch.tensor(face.embedding, dtype=torch.float32, device=self.device)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return {
            "bbox": [float(x) for x in face.bbox],
            "kps": [[float(p[0]), float(p[1])] for p in face.kps],
            "embedding": emb,
        }

    def _extract_insightface(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Sử dụng thời gian chạy ONNX (ONNX runtime) của InsightFace để trích xuất embedding.
        InsightFace yêu cầu ảnh numpy định dạng BGR, và trả về một embedding 512 chiều.
        """
        results = []
        # Xử lý chiều lô dữ liệu (batch dimension)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        for i in range(img_tensor.shape[0]):
            # Chuyển đổi tensor [C, H, W] float [0,1] → numpy [H, W, C] kiểu uint8 BGR
            img = img_tensor[i].cpu()
            if img.shape[0] == 3:  # [C, H, W] → [H, W, C]
                img = img.permute(1, 2, 0)
            img_np = (img.numpy() * 255).astype(np.uint8)
            # Đổi hệ màu RGB → BGR (insightface yêu cầu đầu vào là BGR)
            img_bgr = img_np[:, :, ::-1].copy()
            
            faces = self.app.get(img_bgr)
            if len(faces) > 0:
                # Lấy khuôn mặt lớn nhất (dựa trên diện tích hộp bao - bbox area - lớn nhất)
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                emb = torch.tensor(face.embedding, dtype=torch.float32, device=self.device)
            else:
                # Không phát hiện được khuôn mặt → sử dụng vector toàn số 0 (zero vector)
                print(f"[ArcFace] Cảnh báo (Warning): Không phát hiện được khuôn mặt trong ảnh {i}, sử dụng vector nhúng toàn số 0")
                emb = torch.zeros(512, dtype=torch.float32, device=self.device)
            results.append(emb)
        
        return torch.stack(results)  # [B, 512]
    
    def _extract_resnet(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fallback: ResNet-50 backbone + 512-dim projection.
        """
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Preprocess
        x = img_tensor.to(self.device, dtype=torch.float32)
        x = self.preprocess(x)
        
        # Extract features
        features = self.backbone(x)  # [B, 2048, 1, 1]
        features = features.flatten(1)  # [B, 2048]
        embedding = self.proj(features)  # [B, 512]
        
        return embedding


# ============================================================
# Chạy kiểm thử độc lập (Standalone test)
# ============================================================
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Test] Device: {device}")
    
    extractor = ArcFaceExtractor(device=device)
    
    # Tạo ảnh giả [1, 3, 512, 512]
    fake_img = torch.rand(1, 3, 512, 512, device=device)
    identity_vec = extractor.extract_identity(fake_img)
    
    print(f"\n[Kết quả]")
    print(f"  Mode: {extractor.mode}")
    print(f"  Identity vector shape: {list(identity_vec.shape)}")
    print(f"  Embedding dim: {extractor.embedding_dim}")
    print(f"  L2 norm: {identity_vec.norm(dim=-1).item():.4f}")
    
    if device != "cpu":
        print(f"  VRAM: {torch.cuda.memory_allocated(device)/(1024**2):.1f} MB")
    
    print(f"\n✅ ArcFace Extractor hoạt động!")
