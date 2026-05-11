"""
Hướng dẫn Ngữ cảnh Tiêu cực (Negative Context Guidance) (Lấy mẫu Far-Neg)
==============================================
Triển khai chiến lược Far-Neg từ bài báo NegFaceDiff (2025) để tăng cường khả năng bảo toàn danh tính (identity preservation).

Chiến lược (Strategy):
1. Khối lượng dữ liệu huấn luyện (training data): Thu thập N giá trị nhúng (embeddings) ArcFace khác nhau
2. Cho mỗi ngữ cảnh tích cực (positive context) (danh tính gốc), tìm "Far-Neg" = giá trị nhúng cách xa nhất (theo khoảng cách cosine - cosine distance)
3. Suy luận (Inference): Pha trộn (Blend) hướng dẫn tích cực + tiêu cực (positive + negative guidance): ctx_guided = ctx + scale * (ctx - ctx_neg)

Lợi ích:
- Khả năng phân tách giữa các lớp (Inter-class separability): Mô hình học cách phân biệt giữa các danh tính
- Sự biến thiên trong cùng một lớp (Intra-class variability): Nhưng vẫn cho phép sự biến thiên trong cùng một lớp (biểu cảm, tư thế)
- Bảo toàn danh tính (Identity preservation): Cực kỳ chặt chẽ, gần như ánh xạ 1:1 (1:1 mapping)
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class FarNegSelector:
    """
    Tìm các ngữ cảnh Far-Neg (các giá trị nhúng danh tính cách xa nhất).
    
    Cách sử dụng (Usage):
        selector = FarNegSelector(enable=True, device="cuda")
        # Sau khi tích lũy (accumulate) N lô dữ liệu (batches):
        far_neg = selector.select_far_neg(positive_context)
    """
    
    def __init__(self, enable: bool = False, device: str = "cuda", 
                 num_candidates: int = 32, metric: str = "cosine"):
        """
        Tham số:
            enable: Bật tính năng chọn Far-Neg
            device: Thiết bị GPU dùng cho tính toán độ tương đồng (similarity computation)
            num_candidates: Số lượng giá trị nhúng cần tích lũy trước khi chọn far-neg
            metric: "cosine" hoặc "euclidean"
        """
        self.enable = enable
        self.device = device
        self.num_candidates = num_candidates
        self.metric = metric
        self.embedding_pool = []  # Tích lũy các giá trị nhúng ArcFace (ArcFace embeddings)
        self.max_pool_size = 1024  # Giới hạn kích thước của pool
    
    def add_embeddings(self, embeddings: torch.Tensor) -> None:
        """Thêm một lô các giá trị nhúng (batch of embeddings) vào pool."""
        if not self.enable:
            return
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Tách đồ thị (Detach) + di chuyển sang CPU để tiết kiệm VRAM
        embeddings_cpu = embeddings.detach().cpu()
        
        for emb in embeddings_cpu:
            self.embedding_pool.append(emb)
        
        # Giới hạn kích thước của pool
        if len(self.embedding_pool) > self.max_pool_size:
            # Giữ lại các giá trị nhúng gần đây nhất (Cơ chế FIFO - Vào trước ra trước)
            self.embedding_pool = self.embedding_pool[-self.max_pool_size:]
    
    def select_far_neg(self, positive_context: torch.Tensor) -> torch.Tensor:
        """
        Chọn giá trị nhúng Far-Neg cho từng ngữ cảnh tích cực (positive context).
        
        Tham số:
            positive_context: [B, D] hoặc [D] tensor giá trị nhúng ArcFace
        
        Trả về:
            [B, D] hoặc [D] các ngữ cảnh Far-Neg tương ứng
        """
        if not self.enable or len(self.embedding_pool) < self.num_candidates:
            # Dự phòng (Fallback): trả về giá trị nhúng ngẫu nhiên (random embedding)
            return torch.randn_like(positive_context)
        
        if positive_context.dim() == 1:
            positive_context = positive_context.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = positive_context.shape[0]
        feature_dim = positive_context.shape[1]
        
        # Xếp chồng (Stack) các giá trị nhúng từ pool
        pool_tensor = torch.stack(self.embedding_pool).to(self.device)  # [N, D]
        positive_on_device = positive_context.to(self.device)  # [B, D]
        
        # Tính toán độ tương đồng (Compute similarity): [B, N]
        if self.metric == "cosine":
            # Chuẩn hóa (Normalize)
            pos_norm = torch.nn.functional.normalize(positive_on_device, p=2, dim=-1)
            pool_norm = torch.nn.functional.normalize(pool_tensor, p=2, dim=-1)
            similarity = torch.mm(pos_norm, pool_norm.T)  # [B, N]
        else:  # euclidean
            # Tính toán khoảng cách theo cặp (Compute pairwise distances)
            similarity = -torch.cdist(positive_on_device, pool_tensor)  # [B, N]
        
        # Chọn Far-Neg: độ tương đồng thấp nhất (lowest similarity) / khoảng cách lớn nhất (highest distance)
        far_neg_indices = torch.argmin(similarity, dim=1)  # [B]
        far_neg = pool_tensor[far_neg_indices]  # [B, D]
        
        if squeeze_output:
            far_neg = far_neg.squeeze(0)
        
        return far_neg.to(positive_context.device)
    
    def clear(self) -> None:
        """Xóa sạch (Clear) embedding pool."""
        self.embedding_pool = []


def create_negative_context_batch(
    positive_contexts: torch.Tensor,
    selector: Optional[FarNegSelector] = None,
    fallback: str = "random"
) -> torch.Tensor:
    """
    Tạo một lô (batch) các ngữ cảnh tiêu cực từ các ngữ cảnh tích cực.
    
    Tham số:
        positive_contexts: [B, D] các ngữ cảnh danh tính tích cực
        selector: Phiên bản FarNegSelector (không bắt buộc)
        fallback: "random" hoặc "zero" khi selector không khả dụng (not available)
    
    Trả về:
        [B, D] các ngữ cảnh tiêu cực
    """
    if selector is not None and selector.enable:
        return selector.select_far_neg(positive_contexts)
    
    # Các chiến lược dự phòng (Fallback strategies)
    if fallback == "random":
        return torch.randn_like(positive_contexts)
    elif fallback == "zero":
        return torch.zeros_like(positive_contexts)
    else:
        # Mặc định (Default): ngẫu nhiên (random)
        return torch.randn_like(positive_contexts)


def apply_negative_guidance(
    latent: torch.Tensor,
    positive_context: torch.Tensor,
    negative_context: torch.Tensor,
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """
    Áp dụng hướng dẫn ngữ cảnh tiêu cực (điều hướng điều kiện - conditioning steering).
    
    Công thức: z_guided = z - guidance_scale * (ctx_neg - ctx_pos)
    Hay tương đương: z_guided = z + guidance_scale * (ctx_pos - ctx_neg)
    
    Quá trình này đẩy kết quả tạo sinh TRÁNH XA khỏi danh tính tiêu cực và HƯỚNG TỚI danh tính tích cực.
    
    Tham số:
        latent: [B, L, D] biểu diễn tiềm ẩn (latent representation)
        positive_context: [B, C] danh tính tích cực
        negative_context: [B, C] danh tính tiêu cực
        guidance_scale: Cường độ hướng dẫn (Guidance strength) (>0)
    
    Trả về:
        [B, L, D] biểu diễn tiềm ẩn đã được hướng dẫn (guided latent)
    """
    if guidance_scale <= 0.0:
        return latent
    
    # Đẩy ra xa khỏi tiêu cực, hướng tới tích cực
    guidance_direction = positive_context - negative_context  # [B, C]
    
    # Phát sóng (Broadcast) để khớp với hình dạng của latent [B, L, D]
    if guidance_direction.dim() == 2:
        guidance_direction = guidance_direction.unsqueeze(1)  # [B, 1, C]
    
    # Tỷ lệ và áp dụng (Scale and apply)
    guided_latent = latent + guidance_scale * guidance_direction
    
    return guided_latent
