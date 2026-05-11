"""
Trình tạo Cấu trúc Thưa thớt Giai đoạn 1 (Stage-1 Sparse Structure Generator)
=================================
Dự đoán bố cục không gian chiếm dụng thưa thớt (sparse latent occupancy layout) ở độ phân giải thô (mặc định 16^3)
được điều kiện hóa dựa trên vector ngữ cảnh kết hợp (hybrid context vector).

Module này được thiết kế cố ý với trọng lượng rất nhẹ để có thể huấn luyện và xác thực
độc lập trước khi tích hợp với các giai đoạn tạo hình học/vật liệu (geometry/material generation stages).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _infer_grid_size(slat_length: int) -> int:
    grid = int(round(float(slat_length) ** (1.0 / 3.0)))
    if grid ** 3 != int(slat_length):
        raise ValueError(
            f"slat_length={slat_length} is not a perfect cube. "
            "Sparse structure tokens must form a 3D grid."
        )
    return grid


class SparseStructureGenerator(nn.Module):
    """
    Trình dự đoán cấu trúc thưa thớt được điều kiện hóa bởi ngữ cảnh.

    Đầu vào:
        context: [B, context_dim]
    Đầu ra:
        logits:  [B, slat_length]
    """

    def __init__(
        self,
        context_dim: int = 946,
        slat_length: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_context_tokens: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.slat_length = int(slat_length)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_context_tokens = int(num_context_tokens)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_context_tokens <= 0:
            raise ValueError("num_context_tokens must be > 0")

        if self.hidden_dim % max(1, int(num_heads)) != 0:
            # Giữ cho MHA (Multi-Head Attention) hợp lệ mà không ép buộc người gọi (caller) phải tinh chỉnh thủ công từng checkpoint.
            num_heads = math.gcd(self.hidden_dim, int(num_heads))
            num_heads = max(1, num_heads)
        self.num_heads = int(num_heads)

        grid_size = _infer_grid_size(self.slat_length)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        coords = (coords / max(1, grid_size - 1)) * 2.0 - 1.0
        self.register_buffer("token_coords", coords, persistent=False)

        self.context_proj = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim * self.num_context_tokens),
            nn.SiLU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.query_tokens = nn.Parameter(
            torch.randn(self.slat_length, self.hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.out_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"Expected context shape [B, C], got {tuple(context.shape)}")

        b = context.shape[0]
        ctx_tokens = self.context_proj(context).view(
            b, self.num_context_tokens, self.hidden_dim
        )

        query = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        pos = self.coord_proj(self.token_coords).unsqueeze(0)
        data_tokens = query + pos

        x = torch.cat([ctx_tokens, data_tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x[:, self.num_context_tokens :])
        logits = self.out_head(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_mask(self, context: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(context)
        probs = torch.sigmoid(logits)
        mask = probs >= float(threshold)
        if mask.ndim == 2:
            # Đảm bảo có ít nhất một token kích hoạt (active) cho mỗi mẫu (sample).
            none_active = (~mask).all(dim=1)
            if bool(none_active.any().item()):
                top1 = torch.argmax(probs, dim=1)
                for i in torch.where(none_active)[0].tolist():
                    mask[i, top1[i]] = True
        return mask
