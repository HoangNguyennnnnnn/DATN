"""
FaceDiff Generative U-Net 1D — v5.0 (Hybrid U-DiT)
=====================================================
Nâng cấp từ v4.0:
1. Nút thắt cổ chai Hybrid U-DiT (Hybrid U-DiT Bottleneck): N lớp SelfAttention với 3D-RoPE
2. Điều kiện hóa trong ngữ cảnh (In-context Conditioning): nối (concat) các token ngữ cảnh vào chuỗi (iMF §4.3)
3. 3D-RoPE: Nhúng vị trí xoay (Rotary Position Embedding) trên tọa độ 3D
4. RMSNorm + QK-Norm: ổn định cơ chế chú ý (attention) đối với các chuỗi 3D dài (update3 §4.5)
5. OneNet PixelShuffle1D: tăng/giảm lấy mẫu không mất mát thông tin (lossless up/downsampling) (update3 §4.1)
6. U-ViT Long Skip Connections: kết nối phần dư dạng cộng (additive skips) (update3 §4.2)

Tài liệu tham khảo (References):
- Điều kiện hóa trong ngữ cảnh (In-context conditioning): Bài báo iMF (Geng et al., arXiv:2512.02012v1), Mục 4.3
- 3D-RoPE: TRELLIS (pe_mode="rope"), RoPE (Su et al., 2021)
- U-ViT/Hybrid: U-ViT (Bao et al., CVPR 2023)
- OneNet: PixelUnshuffle/PixelShuffle (2024)
- QK-Norm: Dehghani et al., "Scaling ViT" (2023)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

from src.utils import RMSNorm
from src.hilbert import get_hilbert_permutation_tensors


class TimestepEmbedding(nn.Module):
    """Nhúng Dấu thời gian Hình sin Tiêu chuẩn (Standard Sinusoidal Timestep Embedding)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Backend SageAttention (Hỗ trợ v1/v2 với lượng tử hóa INT4)
SAGE_ATTN_AVAILABLE = False
SAGE_ATTN_VERSION = None  # v1 or v2
_DISABLE_SAGE_ATTN = os.environ.get("FACEDIFF_DISABLE_SAGEATTN", "0").lower() in {
    "1", "true", "yes", "on"
}

if not _DISABLE_SAGE_ATTN:
    try:
        from sageattention import sageattn
        SAGE_ATTN_AVAILABLE = True
        SAGE_ATTN_VERSION = "v1"
    except ImportError:
        pass

# Thử sử dụng SageAttention2 (lượng tử hóa INT4, tương thích phần cứng tốt hơn)
SAGE_ATTN_V2_AVAILABLE = False
if not _DISABLE_SAGE_ATTN:
    try:
        from sageattention import sageattn_qk_int4 as sageattn_v2
        SAGE_ATTN_V2_AVAILABLE = True
        SAGE_ATTN_AVAILABLE = True
        SAGE_ATTN_VERSION = "v2"
    except ImportError:
        pass


def _use_sage(head_dim: int, training: bool = False) -> bool:
    """
    Kiểm tra xem SageAttention có thể được sử dụng với cấu hình phù hợp hay không.
    
    v2 (INT4): Hoạt động tốt cho cả quá trình huấn luyện (training) và suy luận (inference), khắc phục lỗi tính gradient tại chỗ (inplace grad error).
    v1: Chỉ an toàn cho quá trình suy luận (không yêu cầu gradient - no_grad), sẽ gặp lỗi gradient khi huấn luyện.
    
    Tham số:
        head_dim: Số chiều của mỗi head trong cơ chế attention
        training: Xác định xem mô hình có đang ở chế độ huấn luyện hay không
    """
    if not SAGE_ATTN_AVAILABLE:
        return False
    
    # SageAttention v2 có thể dùng cho cả training & inference
    if SAGE_ATTN_V2_AVAILABLE:
        return head_dim in (64, 96, 128)
    
    # SageAttention v1 chỉ an toàn cho inference (no_grad)
    return (SAGE_ATTN_VERSION == "v1" 
            and head_dim in (64, 96, 128) 
            and not training
            and not torch.is_grad_enabled())


# ============================================================
# Cải tiến 4: RMSNorm (update3.md §4.5)
# ============================================================
# RMSNorm is now imported from src.utils


# ============================================================
# Cải tiến 5: Lossless PixelShuffle1D (update3.md §4.1 — OneNet)
# ============================================================
class PixelUnshuffle1D(nn.Module):
    """Giảm lấy mẫu 1D không mất mát (Lossless 1D downsample): [B, C, L] → [B, C*r, L//r].
    Bảo toàn 100% thông tin không gian trong chiều kênh (channel dimension).
    """
    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert L % self.r == 0, f"Sequence length {L} not divisible by {self.r}"
        return x.reshape(B, C, L // self.r, self.r).permute(0, 1, 3, 2).reshape(B, C * self.r, L // self.r)


class PixelShuffle1D(nn.Module):
    """Tăng lấy mẫu 1D không mất mát (Lossless 1D upsample): [B, C*r, L] → [B, C, L*r].
    Ngược lại với PixelUnshuffle1D.
    """
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cr, L = x.shape
        C = Cr // self.r
        return x.reshape(B, C, self.r, L).permute(0, 1, 3, 2).reshape(B, C, L * self.r)


# ============================================================
# Cải tiến 3: 3D Rotary Position Embedding (3D-RoPE)
# ============================================================
class RotaryPositionEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding cho các token Slat.
    
    Thay thế hàm PE cộng (additive PE) cũ bằng RoPE để mã hóa khoảng cách tương đối
    giữa các token trong không gian 3D. RoPE sẽ xoay các vector Q/K
    theo tọa độ xyz → điểm số attention (attention score) tự nhiên phản ánh vị trí 3D.
    
    Tài liệu tham khảo:
    - RoPE: Su et al., "RoFormer", 2021 (arxiv 2104.09864)
    - TRELLIS: pe_mode="rope" trong ModulatedSparseTransformerCrossBlock
    
    Chia head_dim thành 3 phần bằng nhau cho các trục x, y, z.
    Nếu head_dim không chia hết cho 6, phần dư sẽ sử dụng phép xoay bằng 0 (zero-rotation).
    """
    def __init__(self, head_dim: int, grid_size: int = 16, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.grid_size = grid_size
        
        # Chia head_dim cho 3 trục (mỗi trục cần 2 chiều cho cặp sin/cos)
        self.dims_per_axis = (head_dim // 6) * 2  # Phải là số chẵn
        self.total_rope_dims = self.dims_per_axis * 3
        
        # Tính toán trước (Pre-compute) các tần số cho mỗi trục
        dim_per = self.dims_per_axis // 2
        freqs = 1.0 / (theta ** (torch.arange(0, dim_per, dtype=torch.float32) / dim_per))
        self.register_buffer('freqs', freqs)  # [dim_per]
        
        # Tính toán trước tọa độ lưới 3D (Pre-compute 3D grid coordinates)
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)  # [grid_size³, 3]
        self.register_buffer('coords', coords)
    
    def _get_rotary_emb(self, seq_len: int, device: torch.device):
        """Tính toán giá trị nhúng sin/cos cho các vị trí (positions) seq_len."""
        n = self.coords.shape[0]
        if seq_len <= n:
            positions = self.coords[:seq_len]  # [L, 3]
        else:
            repeats = (seq_len // n) + 1
            positions = self.coords.repeat(repeats, 1)[:seq_len]
        
        # [L, dim_per] for each axis
        all_sin, all_cos = [], []
        for axis in range(3):
            pos = positions[:, axis:axis+1]  # [L, 1]
            angles = pos * self.freqs.to(device)  # [L, dim_per]
            all_sin.append(angles.sin())
            all_cos.append(angles.cos())
        
        # Nối (Concat) 3 trục: [L, dims_per_axis/2 * 3] = [L, total_rope_dims/2]
        sin_emb = torch.cat(all_sin, dim=-1)  # [L, total_rope_dims/2]
        cos_emb = torch.cat(all_cos, dim=-1)
        
        return sin_emb, cos_emb
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Áp dụng phép xoay 3D-RoPE vào tensor Q hoặc K.
        x: [B, heads, L, head_dim]
        """
        b, h, l, d = x.shape
        sin_emb, cos_emb = self._get_rotary_emb(l, x.device)
        
        # Ép dtype đồng bộ với x (để tránh trả về float32 làm hỏng hàm sageattn)
        sin_emb = sin_emb.to(dtype=x.dtype)
        cos_emb = cos_emb.to(dtype=x.dtype)
        
        # Tách thành các chiều rope và các chiều giữ nguyên (passthrough dims)
        rope_d = self.total_rope_dims
        x_rope = x[..., :rope_d]         # các chiều bị xoay
        x_pass = x[..., rope_d:]         # các chiều được giữ nguyên không đổi
        
        # Biến đổi hình dạng để xoay: ghép cặp các chiều liên tiếp
        half = rope_d // 2
        x1 = x_rope[..., :half]
        x2 = x_rope[..., half:]
        
        # [1, 1, L, half] broadcast
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
        out1 = x1 * cos_emb - x2 * sin_emb
        out2 = x1 * sin_emb + x2 * cos_emb
        
        x_rotated = torch.cat([out1, out2], dim=-1)
        return torch.cat([x_rotated, x_pass], dim=-1)


# ============================================================
# Các khối chú ý (Attention modules) (có hỗ trợ 3D-RoPE)
# ============================================================
class CrossAttention1D(nn.Module):
    """
    Cross-Attention dùng để tiêm điều kiện hóa (conditioning) vào các đặc trưng của U-Net.
    Sử dụng SageAttention backend khi suy luận (inference).
    """
    def __init__(self, channels, context_dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.to_out = nn.Linear(channels, channels)
        
    def forward(self, x, context):
        b, c, l = x.shape
        x_proj = x.permute(0, 2, 1)
        q = self.to_q(x_proj).view(b, l, self.num_heads, -1).transpose(1, 2)
        
        if context.ndim == 2:
            context = context.unsqueeze(1)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(b, t.shape[1], self.num_heads, -1).transpose(1, 2), kv)
        
        # Ưu tiên SageAttention v2 (lượng tử hóa INT4) cho cả quá trình huấn luyện & suy luận
        if SAGE_ATTN_V2_AVAILABLE and _use_sage(self.head_dim, training=self.training):
            out = sageattn_v2(q, k, v, is_causal=False)
        elif _use_sage(self.head_dim, training=self.training):
            out = sageattn(q, k, v, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).reshape(b, l, c)
        out = self.to_out(out).permute(0, 2, 1)
        return x + out


class SelfAttention1D(nn.Module):
    """
    Self-Attention giữa các token Slat với 3D-RoPE + QK-Norm.
    
    Các nâng cấp (update3.md §4.5):
    - RMSNorm thay thế GroupNorm (nhẹ hơn, đã được kiểm chứng cho DiT)
    - QK-Norm: Sử dụng RMSNorm trên Q/K trước khi tính tích vô hướng (dot-product) để ngăn chặn bùng nổ logit
    - 3D-RoPE xoay các vector Q/K theo tọa độ 3D
    """
    def __init__(self, channels: int, num_heads: int = 8, use_rope: bool = True, grid_size: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = RMSNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        
        # QK-Norm (update3.md §4.5): ổn định attention logits đối với các chuỗi 3D dài
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # 3D-RoPE (Cải tiến 3)
        self.rope = RotaryPositionEmbedding3D(self.head_dim, grid_size) if use_rope else None

    def forward(self, x):
        b, c, l = x.shape
        h = self.norm(x.permute(0, 2, 1)).contiguous()  # RMSNorm operates on last dim
        
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(b, l, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )
        
        # QK-Norm áp dụng trước RoPE (ngăn chặn hiện tượng bùng nổ attention logit)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Áp dụng 3D-RoPE cho Q và K
        if self.rope is not None:
            q = self.rope.rotate(q)
            k = self.rope.rotate(k)
        
        # SageAttention v2 (lượng tử hóa INT4) hoạt động cho cả huấn luyện & suy luận
        if SAGE_ATTN_V2_AVAILABLE and _use_sage(self.head_dim, training=self.training):
            out = sageattn_v2(q, k, v, is_causal=False)
        elif _use_sage(self.head_dim, training=self.training):
            out = sageattn(q, k, v, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).reshape(b, l, c)
        out = self.to_out(out).permute(0, 2, 1)
        return x + out


# ============================================================
# Cải tiến 2: Điều kiện hóa trong ngữ cảnh (In-context Conditioning) (iMF paper §4.3)
# ============================================================
class ContextTokenizer(nn.Module):
    """
    Chuyển đổi vector ngữ cảnh (context vector) → các token tiền tố (prefix tokens) để nối vào chuỗi.
    
    Theo bài báo iMF (Geng et al., arXiv:2512.02012v1, Mục 4.3):
    - Các token phân loại (Class tokens): mã hóa điều kiện hóa danh tính/biểu cảm (identity/expression conditioning)
    - Các token thời gian (Time tokens): mã hóa thông tin dấu thời gian (timestep)
    
    Ngữ cảnh [B, ctx_dim] → num_tokens × [B, model_dim] tokens
    """
    def __init__(self, context_dim: int, model_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(context_dim, model_dim * num_tokens),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(model_dim)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        ngữ cảnh (context): [B, ctx_dim]
        Trả về: [B, num_tokens, model_dim]
        """
        tokens = self.proj(context)  # [B, model_dim * num_tokens]
        tokens = tokens.view(context.shape[0], self.num_tokens, -1)  # [B, num_tokens, model_dim]
        return self.norm(tokens)


class TimeTokenizer(nn.Module):
    """Mã hóa dấu thời gian thành các token tiền tố (theo bài báo iMF: 4 time tokens)."""
    def __init__(self, time_dim: int, model_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(time_dim, model_dim * num_tokens),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(model_dim)
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(t_emb).view(t_emb.shape[0], self.num_tokens, -1)
        return self.norm(tokens)


class GuidanceTokenizer(nn.Module):
    """Mã hóa các điều kiện hướng dẫn (omega, tmin, tmax) thành các token tiền tố."""
    def __init__(self, model_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(3, model_dim * num_tokens),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, guidance_cond: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(guidance_cond).view(guidance_cond.shape[0], self.num_tokens, -1)
        return self.norm(tokens)


# ============================================================
# ResBlock1D with AdaLN
# ============================================================
class ResBlock1D(nn.Module):
    """Khối ResBlock 1D với Điều chế AdaLN (AdaLN Modulation) (tỷ lệ + độ dời)."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.act2 = nn.SiLU()
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        scale_shift = self.adaLN(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


# ============================================================
# Main Model: IMFUNet1D v4.0 (Hybrid U-DiT)
# ============================================================
class IMFUNet1D(nn.Module):
    """
    Hybrid U-DiT dành cho Improved Mean Flow (iMF) — v4.0.
    
    3 cải tiến chính:
    1. Nút thắt cổ chai Hybrid U-DiT (Hybrid U-DiT Bottleneck): 4 lớp SelfAttention + RoPE (thay vì 1 lớp)
    2. Điều kiện hóa trong ngữ cảnh (In-context Conditioning): ngữ cảnh → các token tiền tố, được xử lý qua SelfAttention
       (theo bài báo iMF §4.3: 8 class tokens + 4 time tokens)
    3. 3D-RoPE: Nhúng vị trí xoay (Rotary Position Embedding) trên tọa độ 3D trong SelfAttention
    4. Các token điều kiện hóa hướng dẫn (Guidance-conditioning tokens): omega/tmin/tmax như là các điều kiện tường minh
    
    Bộ mã hóa/giải mã Conv1d (Conv1d encoder/decoder) được giữ nguyên → tiết kiệm VRAM.
    Nút thắt cổ chai Transformer (Transformer bottleneck) hoạt động trên 512 token (sau 3 lần giảm lấy mẫu) → rất nhẹ.
    
    Đầu vào:
      - x_t: [B, L, D] các token Slat nhiễu
      - t:   [B] các dấu thời gian (timesteps) ∈ [0, 1]  
      - context: [B, C] vector danh tính+biểu cảm (identity+expression)
    """
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: List[int] = [128, 256, 512],
        context_dim: int = 512,
        slat_length: int = 4096,
        num_bottleneck_layers: int = 4,     # Cải tiến 1
        num_context_tokens: int = 8,         # Cải tiến 2
        num_time_tokens: int = 4,            # Cải tiến 2
        num_r_tokens: int = 4,               # iMF r conditioning
        num_guidance_tokens: int = 4,        # iMF flexible guidance conditioning
        use_hilbert_ordering: bool = True,     # Hilbert SFC cho spatial locality
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.bottleneck_dim = hidden_dims[-1]
        self.hidden_dim = hidden_dims[0]
        self.num_bottleneck_layers = num_bottleneck_layers
        self.num_context_tokens = num_context_tokens
        self.num_time_tokens = num_time_tokens
        self.num_r_tokens = num_r_tokens
        self.num_guidance_tokens = num_guidance_tokens
        
        # Timestep embedding
        self.time_dim = hidden_dims[0] * 4
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dims[0]),
            nn.Linear(hidden_dims[0], self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        self.r_time_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dims[0]),
            nn.Linear(hidden_dims[0], self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        # === Cải tiến 2: In-context Conditioning ===
        # Context → prefix tokens (thay CrossAttention)
        self.context_tokenizer = ContextTokenizer(context_dim, self.bottleneck_dim, num_context_tokens) if num_context_tokens > 0 else None
        self.time_tokenizer = TimeTokenizer(self.time_dim, self.bottleneck_dim, num_time_tokens) if num_time_tokens > 0 else None
        self.r_tokenizer = TimeTokenizer(self.time_dim, self.bottleneck_dim, num_r_tokens) if num_r_tokens > 0 else None
        self.guidance_tokenizer = GuidanceTokenizer(self.bottleneck_dim, num_guidance_tokens) if num_guidance_tokens > 0 else None
        
        # Init conv  
        self.init_conv = nn.Conv1d(input_dim, hidden_dims[0], 3, padding=1)
        
        # Bộ mã hóa với giảm lấy mẫu không mất mát kiểu OneNet (OneNet-style lossless downsampling) (update3.md §4.1)
        self.downs = nn.ModuleList()
        in_c = hidden_dims[0]
        for dim in hidden_dims:
            self.downs.append(nn.ModuleList([
                ResBlock1D(in_c, dim, self.time_dim),
                ResBlock1D(dim, dim, self.time_dim),
                nn.Sequential(PixelUnshuffle1D(2), nn.Conv1d(dim * 2, dim, 1)),
            ]))
            in_c = dim
        
        # === Cải tiến 1: Nút thắt cổ chai Hybrid U-DiT ===
        # Transformer nhiều lớp (Multi-layer Transformer) thay cho 1 lớp SelfAttn
        grid_size = round(slat_length ** (1/3))
        self.mid_block_in = ResBlock1D(self.bottleneck_dim, self.bottleneck_dim, self.time_dim)
        self.mid_transformer = nn.ModuleList([
            SelfAttention1D(self.bottleneck_dim, num_heads=8, use_rope=True, grid_size=grid_size)
            for _ in range(num_bottleneck_layers)
        ])
        self.mid_block_out = ResBlock1D(self.bottleneck_dim, self.bottleneck_dim, self.time_dim)
        
        # Bộ giải mã với PixelShuffle1D + kết nối phần dư dài dạng cộng (additive long skip) (update3.md §4.1, §4.2)
        self.ups = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # U-ViT long skip projections
        hidden_dims_rev = list(reversed(hidden_dims))
        in_c = self.bottleneck_dim
        for dim in hidden_dims_rev:
            self.ups.append(nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_c, dim * 2, 1), PixelShuffle1D(2)),
                ResBlock1D(dim, dim, self.time_dim),
                ResBlock1D(dim, dim, self.time_dim),
            ]))
            # Kết nối phần dư dạng cộng (Additive skip): ánh xạ các kênh của bộ mã hóa sang các kênh của bộ giải mã nếu cần thiết
            self.skip_projs.append(
                nn.Conv1d(dim, dim, 1) if in_c != dim else nn.Identity()
            )
            in_c = dim
        
        # Đầu ra cuối cùng (RMSNorm để đảm bảo tính nhất quán)
        self.final_norm = RMSNorm(hidden_dims[0])
        self.final_act = nn.SiLU()
        self.final_proj = nn.Conv1d(hidden_dims[0], input_dim, 3, padding=1
        )
        
        # Hilbert Space-Filling Curve ordering
        # Conv1d kernel=3 sẽ convolve qua các feature 3D lân cận nhờ Hilbert ordering
        self.use_hilbert_ordering = bool(use_hilbert_ordering)
        if self.use_hilbert_ordering:
            grid_size = round(slat_length ** (1/3))
            if grid_size ** 3 == slat_length and (grid_size & (grid_size - 1)) == 0:
                h2r, r2h = get_hilbert_permutation_tensors(grid_size)
                self.register_buffer('_hilbert_to_raster', h2r, persistent=False)
                self.register_buffer('_raster_to_hilbert', r2h, persistent=False)
            else:
                self.use_hilbert_ordering = False

    def _make_prefix_tokens(
        self,
        tokenizer: Optional[nn.Module],
        source: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        if tokenizer is None or num_tokens <= 0:
            model_dim = int(self.bottleneck_dim)
            return source.new_zeros((source.shape[0], 0, model_dim))
        return tokenizer(source)

    def _forward_core(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shared hidden-state path for forward() and get_hidden_state().
        Returns final decoder features in canonical [B, L, hidden_dim] layout.
        """
        # Định dạng đầu vào [B, L, D] → [B, D, L]
        if x_t.ndim == 3 and x_t.shape[-1] == self.input_dim:
            x_t = x_t.permute(0, 2, 1)

        # Hilbert reorder: raster → hilbert (cải thiện spatial locality cho Conv1d)
        if self.use_hilbert_ordering:
            x_t = x_t[:, :, self._hilbert_to_raster]

        if r is None:
            r = t

        def _ensure_batch_scalar(val, default: float) -> torch.Tensor:
            if val is None:
                return torch.full((t.shape[0],), default, device=t.device, dtype=t.dtype)
            if not torch.is_tensor(val):
                return torch.full((t.shape[0],), float(val), device=t.device, dtype=t.dtype)
            out = val.to(device=t.device, dtype=t.dtype)
            if out.ndim == 0:
                return out.expand(t.shape[0])
            if out.shape[0] != t.shape[0]:
                return out.reshape(-1)[0].expand(t.shape[0])
            return out

        omega_b = _ensure_batch_scalar(omega, 1.0)
        cfg_tmin_b = _ensure_batch_scalar(cfg_tmin, 0.0)
        cfg_tmax_b = _ensure_batch_scalar(cfg_tmax, 1.0)

        t_emb = self.time_mlp(t)
        r_emb = self.r_time_mlp(r)
        # Embedding cặp nhận biết thứ tự (Order-aware pair embedding) cho (r, t) trong khi vẫn giữ nguyên hành vi ở biên (boundary behavior) r=t.
        dt = (t - r).unsqueeze(-1)
        time_cond_emb = t_emb + dt * (r_emb - t_emb)
        guidance_cond = torch.stack([omega_b, cfg_tmin_b, cfg_tmax_b], dim=-1)

        x = self.init_conv(x_t)
        
        # === Bộ mã hóa (Encoder) ===
        skips = []
        for block1, block2, downsamp in self.downs:
            x = block1(x, time_cond_emb)
            x = block2(x, time_cond_emb)
            skips.append(x)
            x = downsamp(x)
        
        # === Nút thắt cổ chai Hybrid U-DiT ===
        x = self.mid_block_in(x, time_cond_emb)
        
        # Điều kiện hóa trong ngữ cảnh: nối các token tiền tố
        b, c, seq_len = x.shape
        # x: [B, C, L] → [B, L, C] → concat prefix → [B, L+K, C]
        x_seq = x.permute(0, 2, 1)  # [B, L, C]
        
        ctx_tokens = self._make_prefix_tokens(self.context_tokenizer, context, self.num_context_tokens)
        time_tokens = self._make_prefix_tokens(self.time_tokenizer, time_cond_emb, self.num_time_tokens)
        r_tokens = self._make_prefix_tokens(self.r_tokenizer, r_emb, self.num_r_tokens)
        guidance_tokens = self._make_prefix_tokens(self.guidance_tokenizer, guidance_cond, self.num_guidance_tokens)
        
        # Nối (Concat): [time_tokens | r_tokens | guidance_tokens | ctx_tokens | data_tokens]
        num_prefix = self.num_context_tokens + self.num_time_tokens + self.num_r_tokens + self.num_guidance_tokens
        x_with_prefix = torch.cat([time_tokens, r_tokens, guidance_tokens, ctx_tokens, x_seq], dim=1)  # [B, K+L, C]
        
        # Chuyển đổi ngược lại thành [B, C, K+L] cho SelfAttention1D (cần định dạng [B, C, L])
        x_with_prefix = x_with_prefix.permute(0, 2, 1)
        
        # SelfAttention nhiều lớp với 3D-RoPE
        for attn_layer in self.mid_transformer:
            x_with_prefix = attn_layer(x_with_prefix)
        
        # Loại bỏ các token tiền tố → trở về dạng [B, C, L]
        x = x_with_prefix[:, :, num_prefix:]
        
        x = self.mid_block_out(x, time_cond_emb)
        
        # === Bộ giải mã với các kết nối phần dư dài dạng cộng (kiểu U-ViT) ===
        for (up, block1, block2), skip_proj in zip(self.ups, self.skip_projs):
            x = up(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            # Kết nối phần dư dạng cộng (update3.md §4.2): không nối (concat), chỉ cộng kết nối phần dư đã được ánh xạ
            x = x + skip_proj(skip)
            x = block1(x, time_cond_emb)
            x = block2(x, time_cond_emb)
        
        # Đầu ra cuối cùng với RMSNorm
        hidden = self.final_norm(x.permute(0, 2, 1))
        if self.use_hilbert_ordering:
            hidden = hidden[:, self._raster_to_hilbert, :]
        return hidden

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Lan truyền xuôi dự đoán vận tốc.
        return_hidden=True trả về (velocity, hidden_state) để tái sử dụng cho v-head.
        """
        orig_shape = bool(x_t.ndim == 3 and x_t.shape[-1] == self.input_dim)
        hidden = self._forward_core(
            x_t,
            t,
            context,
            r=r,
            omega=omega,
            cfg_tmin=cfg_tmin,
            cfg_tmax=cfg_tmax,
        )
        out = self.final_act(hidden)
        out = self.final_proj(out.permute(0, 2, 1)).permute(0, 2, 1)

        if not orig_shape:
            out = out.permute(0, 2, 1)

        if return_hidden:
            return out, hidden
        return out

    def get_hidden_state(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Trả về hidden state cuối decoder dưới dạng [B, L, hidden_dim]."""
        return self._forward_core(
            x_t,
            t,
            context,
            r=r,
            omega=omega,
            cfg_tmin=cfg_tmin,
            cfg_tmax=cfg_tmax,
        )
