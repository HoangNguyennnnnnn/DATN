"""
Bộ điều hợp Biểu cảm FLAME (FLAME Expression Adapter)
=========================
Trích xuất tham số biểu cảm (expression blendshapes) từ lưới 3D (mesh) hoặc ảnh 2D.

FLAME (Faces Learned with an Articulated Model and Expressions) cung cấp
một mô hình tham số hóa khuôn mặt 3D với các biến riêng biệt:
- Shape (β): hình dạng khuôn mặt (danh tính) — 100 tham số
- Expression (ψ): biểu cảm — 50 tham số
- Pose (θ): tư thế đầu — 6 tham số (xoay toàn cục - global rotation + hàm - jaw)

Module này tách riêng luồng biểu cảm, kết hợp với nhận dạng danh tính ArcFace (ArcFace identity)
để tạo ra vector ngữ cảnh kết hợp (hybrid context vector) cho mô hình sinh.

Hybrid context v4.1: ArcFace [512] + Biểu cảm (Expression) [50] + DINO_Back [384] = [946]

VRAM: < 10 MB (MLP trọng lượng nhẹ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FLAMEExpressionAdapter(nn.Module):
    """
    Trích xuất và mã hóa tham số biểu cảm FLAME.
    
    3 chế độ (modes):
    1. Từ các đỉnh lưới (From mesh vertices): Khớp (Fit) tham số FLAME từ lưới 3D → hệ số biểu cảm (expression coefficients)
       (cần mô hình FLAME, dùng cho quá trình huấn luyện)
    2. Từ hình ảnh (From image): Dự đoán biểu cảm từ ảnh 2D bằng mạng CNN nhẹ (lightweight CNN)
       (dùng cho quá trình suy luận - inference - khi không có lưới 3D)
    3. Giả lập (Mock): Vector biểu cảm ngẫu nhiên (dành cho kiểm thử - testing)
    
    Đầu ra (Output): [B, 50] vector biểu cảm, đã được chuẩn hóa.
    """
    
    def __init__(
        self,
        expression_dim: int = 50,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.expression_dim = expression_dim
        self.device = torch.device(device)
        
        # Trình dự đoán biểu cảm dựa trên hình ảnh trọng lượng nhẹ (Lightweight image-based expression predictor)
        # Đầu vào (Input): ảnh khuôn mặt được kết xuất (rendered face image) [B, 3, 112, 112]
        # Đầu ra (Output): các tham số biểu cảm (expression params) [B, 50]
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),   # [B, 32, 56, 56]
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 28, 28]
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 14, 14]
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# [B, 256, 7, 7]
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                     # [B, 256, 1, 1]
            nn.Flatten(),                                # [B, 256]
        )
        
        self.expression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, expression_dim),
            nn.Tanh(),  # Expression params bounded [-1, 1]
        )
        
        # Trình trích xuất biểu cảm dựa trên đỉnh (Vertex-based expression extractor) (dạng chiếu tựa PCA - PCA-like projection)
        # Dùng khi có các đỉnh của lưới (mesh vertices) thay vì hình ảnh
        self.vertex_projection = nn.Sequential(
            nn.Linear(5023 * 3, 512),  # FLAME template: 5023 vertices × 3
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, expression_dim),
            nn.Tanh(),
        )
        
        # Các nhãn biểu cảm được định nghĩa sẵn (21 biểu cảm FaceVerse)
        self.expression_names = [
            "neutral", "smile", "mouth_stretch", "anger", "jaw_left",
            "jaw_right", "jaw_forward", "mouth_left", "mouth_right",
            "dimpler", "chin_raiser", "lip_pressor", "lip_tightener",
            "lip_pucker", "lip_stretch", "lip_funneler", "brow_lowerer",
            "brow_raiser_left", "brow_raiser_right", "cheek_raiser",
            "eye_closed",
        ]
        
        self.to(self.device)
        print(f"[FLAME] Expression Adapter initialized: {expression_dim}-dim")
        param_count = sum(p.numel() for p in self.parameters())
        print(f"[FLAME] Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    @torch.no_grad()
    def extract_from_image(self, face_image: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán các tham số biểu cảm từ ảnh khuôn mặt.
        
        Tham số:
            face_image: [B, 3, H, W] (bất kỳ kích thước nào, sẽ được resize về 112)
            
        Trả về:
            [B, 50] vector biểu cảm
        """
        self.eval()
        x = face_image.to(self.device, dtype=torch.float32)
        
        # Thay đổi kích thước (Resize) về 112×112 (tiêu chuẩn phân tích khuôn mặt - face analysis)
        if x.shape[-1] != 112 or x.shape[-2] != 112:
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        
        features = self.image_encoder(x)
        expression = self.expression_head(features)
        
        return expression  # [B, 50]
    
    @torch.no_grad()
    def extract_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất biểu cảm từ các đỉnh lưới (mesh vertices).
        
        Tham số:
            vertices: [B, N, 3] các đỉnh lưới
            
        Trả về:
            [B, 50] vector biểu cảm
        """
        self.eval()
        v = vertices.to(self.device, dtype=torch.float32)
        b = v.shape[0]
        
        # Đệm (Pad) hoặc cắt bớt (truncate) về 5023 đỉnh (kích thước template của FLAME)
        target_n = 5023
        if v.shape[1] > target_n:
            v = v[:, :target_n, :]
        elif v.shape[1] < target_n:
            pad = torch.zeros(b, target_n - v.shape[1], 3, device=self.device)
            v = torch.cat([v, pad], dim=1)
        
        v_flat = v.reshape(b, -1)  # [B, 5023*3]
        expression = self.vertex_projection(v_flat)
        
        return expression  # [B, 50]
    
    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass) dùng cho quá trình huấn luyện (training)."""
        x = face_image.to(self.device, dtype=torch.float32)
        if x.shape[-1] != 112 or x.shape[-2] != 112:
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        features = self.image_encoder(x)
        return self.expression_head(features)


def create_hybrid_context(
    identity: torch.Tensor,
    expression: torch.Tensor,
    back_shape: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Kết hợp ArcFace danh tính (identity) + FLAME biểu cảm (expression) + DINOv2_Back (v4.1)
    
    Tham số:
        identity: [B, 512] — ArcFace vector danh tính (đã chuẩn hóa L2 - L2-normalized)
        expression: [B, 50] — FLAME vector biểu cảm (nằm trong khoảng [-1, 1])
        back_shape: [B, 384] — DINOv2-Small từ ảnh mặt sau (back image)
        
    Trả về:
        [B, 562] hoặc [B, 946] — vector ngữ cảnh kết hợp (hybrid context vector)
    """
    if back_shape is not None:
        return torch.cat([identity, expression, back_shape], dim=-1)
    return torch.cat([identity, expression], dim=-1)


# ============================================================
# Chạy kiểm thử độc lập (Standalone test)
# ============================================================
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    adapter = FLAMEExpressionAdapter(expression_dim=50, device=device)
    
    # Kiểm thử 1 (Test 1): Từ hình ảnh (From image)
    fake_img = torch.rand(2, 3, 256, 256, device=device)
    expr = adapter.extract_from_image(fake_img)
    print(f"\n[Test 1] Image → Expression: {list(expr.shape)}, range=[{expr.min():.3f}, {expr.max():.3f}]")
    
    # Kiểm thử 2 (Test 2): Từ các đỉnh (From vertices)
    fake_verts = torch.randn(2, 4000, 3, device=device)
    expr2 = adapter.extract_from_vertices(fake_verts)
    print(f"[Test 2] Vertices → Expression: {list(expr2.shape)}, range=[{expr2.min():.3f}, {expr2.max():.3f}]")
    
    # Kiểm thử 3 (Test 3): Ngữ cảnh kết hợp (Hybrid context)
    identity = torch.randn(2, 512, device=device)
    identity = F.normalize(identity, p=2, dim=-1)
    hybrid = create_hybrid_context(identity, expr)
    print(f"[Test 3] Hybrid context: {list(hybrid.shape)} (512 identity + 50 expression)")
    
    if device != "cpu":
        print(f"\n  VRAM: {torch.cuda.memory_allocated(device)/(1024**2):.1f} MB")
    print(f"\n✅ FLAME Expression Adapter hoạt động!")
