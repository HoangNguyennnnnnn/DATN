"""
Voxel Mamba Backbone cho FaceDiff v5.0
======================================
Thay thế IMFUNet1D bằng Mô hình Trạng thái Không gian (State Space Model) có độ phức tạp O(N).

Kiến trúc:
- Các khối Mamba hai chiều (Bidirectional Mamba blocks) (quét xuôi + quét ngược)
- Điều kiện hóa trong ngữ cảnh (In-context conditioning) (ngữ cảnh + thời gian + các token điều hướng)
- Độ phức tạp tuyến tính O(N) thay vì O(N²)

Tài liệu tham khảo (References):
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection
"""

import os
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import RMSNorm
from src.hilbert import get_hilbert_permutation_tensors


# RMSNorm hiện được import từ src.utils


Mamba = None  # type: ignore[assignment]
_MAMBA_IMPORT_ATTEMPTED = False
_MAMBA_IMPORT_ERROR: Optional[Exception] = None


def _ensure_mamba_import() -> bool:
    """Thử import mamba-ssm đúng một lần và lưu kết quả vào bộ nhớ tạm (cache)."""
    global Mamba, _MAMBA_IMPORT_ATTEMPTED, _MAMBA_IMPORT_ERROR

    if _MAMBA_IMPORT_ATTEMPTED:
        return Mamba is not None

    _MAMBA_IMPORT_ATTEMPTED = True
    try:
        from mamba_ssm import Mamba as ImportedMamba
        Mamba = ImportedMamba  # type: ignore[assignment]
        _MAMBA_IMPORT_ERROR = None
    except Exception as exc:
        Mamba = None  # type: ignore[assignment]
        _MAMBA_IMPORT_ERROR = exc

    return Mamba is not None


def _resolve_backend() -> str:
    """Xác định backend một lần duy nhất, cùng với một lối thoát (escape hatch) rõ ràng dành cho gỡ lỗi (debugging)."""
    requested = os.environ.get("FACEDIFF_VOXEL_MAMBA_BACKEND", "auto")
    return _resolve_requested_backend(requested)


def _resolve_requested_backend(requested: Optional[str]) -> str:
    """Xác định backend từ biến môi trường/cấu hình đồng thời duy trì một tùy chọn dự phòng (fallback) an toàn."""
    normalized = "auto" if requested is None else str(requested).strip().lower()

    if normalized in {"gru", "fallback"}:
        return "gru"

    if normalized in {"mamba", "mamba-ssm"}:
        return "mamba" if _ensure_mamba_import() else "gru"

    return "mamba" if _ensure_mamba_import() else "gru"


MAMBA_BACKEND = _resolve_backend()
MAMBA_AVAILABLE = MAMBA_BACKEND == "mamba"

if MAMBA_AVAILABLE:
    print("[VoxelMamba] Using mamba-ssm CUDA implementation")
else:
    if _MAMBA_IMPORT_ATTEMPTED and _MAMBA_IMPORT_ERROR is not None:
        print(f"[VoxelMamba] mamba-ssm import failed ({_MAMBA_IMPORT_ERROR!r}); using fallback bidirectional GRU")
    else:
        print("[VoxelMamba] Backend forced to bidirectional GRU")


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


class BidirectionalMambaBlock(nn.Module):
    """
    Khối SSM Hai chiều (Bidirectional SSM Block) với kết nối phần dư (residual connection).
    
    Sử dụng Mamba cho quét xuôi (forward scan) + quét ngược (backward scan),
    sau đó kết hợp với phần dư.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1, backend: str = "auto"):
        super().__init__()
        self.dim = dim
        self.backend = _resolve_requested_backend(backend)
        self.use_mamba = self.backend == "mamba"
        
        if self.use_mamba:
            # Mamba quét xuôi và ngược
            self.forward_mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.backward_mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Dự phòng (Fallback): GRU Hai chiều
            self.gru = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        
        self.norm = RMSNorm(dim)  # RMSNorm: cheaper, no mean-centering (update3.md §4.5)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [B, L, dim]
        Trả về: [B, L, dim]
        """
        residual = x
        x = self.norm(x)
        
        if self.use_mamba:
            # Quét xuôi (Forward scan)
            fwd = self.forward_mamba(x)
            # Quét ngược (Backward scan)
            bwd_input = torch.flip(x, dims=[1])
            bwd = self.backward_mamba(bwd_input)
            bwd = torch.flip(bwd, dims=[1])
            # Kết hợp (Combine)
            out = fwd + bwd
        else:
            # GRU dự phòng (Fallback GRU)
            out, _ = self.gru(x)
        
        out = self.dropout(out)
        return out + residual


class VoxelMamba(nn.Module):
    """
    Mạng cơ sở Voxel Mamba (Voxel Mamba Backbone) dành cho việc sinh token Slat.
    
    Thay thế mạng IMFUNet1D với độ phức tạp O(N).
    
    Kiến trúc:
    1. Nhúng đầu vào (Input embedding): [B, L, input_dim] -> [B, L, hidden_dim]
    2. Các token tiền tố (Prefix tokens): [ngữ cảnh(8) + thời_gian_t(4) + thời_gian_r(4) + interval(4) + điều_hướng(4)] = 24 tokens
    3. Ngăn xếp các khối BidirectionalMambaBlocks
    4. Phép chiếu đầu ra (Output projection): [B, L, hidden_dim] -> [B, L, input_dim]
    
    iMF yêu cầu u_θ(z_t, r, t) phải được điều kiện hóa trên CẢ HAI dấu thời gian r (bắt đầu) và t (kết thúc).
    Nếu không có điều kiện r, mô hình sẽ suy thoái thành khớp luồng đơn giản (simple flow matching)
    và hàm hợp JVP sẽ trở nên vô nghĩa.
    
    Tham số:
        input_dim: Số chiều của token Slat (mặc định: 32, khớp với latent_dim của SC-VAE)
        hidden_dim: Số chiều ẩn cho Mamba (mặc định: 512)
        num_layers: Số lượng khối BidirectionalMambaBlocks (mặc định: 12)
        slat_length: Chiều dài chuỗi (mặc định: 4096)
        context_dim: Số chiều ngữ cảnh lai (mặc định: 946)
        num_context_tokens: Số lượng token tiền tố cho ngữ cảnh (mặc định: 8)
        num_time_tokens: Số lượng token thời gian cho t (mặc định: 4)
        num_r_tokens: Số lượng token thời gian cho r (mặc định: 4)
        num_interval_tokens: Số lượng token (t-r) interval (mặc định: 4, theo iMF paper Tab. 4)
        num_guidance_tokens: Số lượng token tiền tố cho điều hướng (mặc định: 4)
        dropout: Tỉ lệ bỏ qua (Dropout rate) (mặc định: 0.1)
    """
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 12,
        slat_length: int = 4096,
        context_dim: int = 946,
        backend: str = "auto",
        strict: bool = False,
        num_context_tokens: int = 8,
        num_time_tokens: int = 4,
        num_r_tokens: int = 4,
        num_interval_tokens: int = 4,
        num_guidance_tokens: int = 4,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_hilbert_ordering: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.slat_length = slat_length
        self.num_context_tokens = num_context_tokens
        self.num_time_tokens = num_time_tokens
        self.num_r_tokens = num_r_tokens
        self.num_interval_tokens = num_interval_tokens
        self.num_guidance_tokens = num_guidance_tokens
        self.total_prefix_tokens = num_context_tokens + num_time_tokens + num_r_tokens + num_interval_tokens + num_guidance_tokens
        self.requested_backend = str(backend).strip().lower()
        self.backend = _resolve_requested_backend(backend)
        if strict and self.requested_backend in {"mamba", "mamba-ssm"} and self.backend != "mamba":
            raise RuntimeError(
                "VoxelMamba strict mode requested mamba backend, but mamba-ssm is not available in this environment"
            )
        self.use_mamba = self.backend == "mamba"
        
        # Nhúng đầu vào (Input embedding)
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # Nhúng dấu thời gian cho t (thời gian kết thúc)
        self.time_embed_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Nhúng dấu thời gian cho r (thời gian bắt đầu) — bài báo iMF yêu cầu điều kiện hóa r riêng biệt
        # u_θ(z_t, r, t): mạng nơ-ron phải phân biệt được vận tốc trung bình qua các khoảng [r, t] khác nhau
        self.r_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Các bộ tạo token điều kiện hóa trong ngữ cảnh (In-context conditioning tokenizers)
        self.context_tokenizer = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * num_context_tokens),
            nn.SiLU(),
        )
        
        self.time_tokenizer = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim * num_time_tokens),
            nn.SiLU(),
        )
        
        self.r_tokenizer = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim * num_r_tokens),
            nn.SiLU(),
        )
        
        # Explicit (t-r) interval conditioning — iMF paper Tab. 4:
        # "(t,r) cond: t−r, t, r" — network conditions on the averaging
        # interval length (t-r) in addition to t and r individually.
        # This gives the SSM direct access to interval magnitude without
        # needing to learn the subtraction implicitly.
        self.interval_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.interval_tokenizer = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim * num_interval_tokens),
            nn.SiLU(),
        )
        
        self.guidance_tokenizer = nn.Sequential(
            nn.Linear(3, hidden_dim * num_guidance_tokens),  # [omega, t_min, t_max]
            nn.SiLU(),
        )
        
        # Ngăn xếp các khối Mamba hai chiều
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                dim=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                backend=self.backend,
            )
            for _ in range(num_layers)
        ])
        
        # Chuẩn hóa (norm) và phép chiếu (projection) cuối cùng
        self.output_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Khởi tạo ma trận trọng số phép chiếu đầu ra về 0 để thiết lập ánh xạ đồng nhất (identity mapping) ở bước đầu
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Hilbert Space-Filling Curve ordering
        # Bảo toàn spatial locality: token 3D gần nhau → liền kề trong chuỗi 1D
        # GRU/Mamba xử lý tuần tự → hidden state mang context lân cận hiệu quả hơn
        # VRAM: chỉ 2 tensor int64 × 4096 = 64KB
        self.use_hilbert_ordering = bool(use_hilbert_ordering)
        if self.use_hilbert_ordering:
            grid_size = round(slat_length ** (1/3))
            if grid_size ** 3 == slat_length and (grid_size & (grid_size - 1)) == 0:
                h2r, r2h = get_hilbert_permutation_tensors(grid_size)
                self.register_buffer('_hilbert_to_raster', h2r, persistent=False)
                self.register_buffer('_raster_to_hilbert', r2h, persistent=False)
                print(f"[VoxelMamba] Hilbert ordering enabled (grid {grid_size}³ = {slat_length} tokens)")
            else:
                print(f"[VoxelMamba] Hilbert ordering disabled: slat_length={slat_length} is not a perfect cube of power-of-2")
                self.use_hilbert_ordering = False
        
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
        """Phần lõi chung cho cả forward() và get_hidden_state(), trả về hidden state đã chuẩn hóa."""
        B, L, D = x_t.shape
        device = x_t.device
        
        if r is None:
            r = t
        if omega is None:
            omega = torch.ones(B, device=device)
        if cfg_tmin is None:
            cfg_tmin = torch.zeros(B, device=device)
        if cfg_tmax is None:
            cfg_tmax = torch.ones(B, device=device)
        
        h = self.input_embed(x_t)
        
        # Hilbert reorder: raster → hilbert (bảo toàn spatial locality cho GRU/Mamba)
        if self.use_hilbert_ordering:
            h = h[:, self._hilbert_to_raster, :]
        
        ctx_emb = self.context_tokenizer(context)
        ctx_tokens = ctx_emb.view(B, self.num_context_tokens, self.hidden_dim)
        
        t_emb = self.time_mlp(t)
        t_emb_expanded = self.time_tokenizer(t_emb)
        time_tokens = t_emb_expanded.view(B, self.num_time_tokens, self.hidden_dim)
        
        r_emb = self.r_mlp(r)
        r_emb_expanded = self.r_tokenizer(r_emb)
        r_tokens = r_emb_expanded.view(B, self.num_r_tokens, self.hidden_dim)
        
        # (t-r) interval tokens — iMF paper Tab. 4 ("(t,r) cond: t-r"). Signed.
        # _sample_t_r() đảm bảo r ≤ t nên (t-r) ≥ 0; vẫn giữ signed để model học
        # đúng đặc trưng "interval magnitude" mà sin/cos timestep embedding xử lý
        # native cho đầu vào không âm.
        interval = (t - r)
        interval_emb = self.interval_mlp(interval)
        interval_emb_expanded = self.interval_tokenizer(interval_emb)
        interval_tokens = interval_emb_expanded.view(B, self.num_interval_tokens, self.hidden_dim)
        
        guidance_input = torch.stack([omega, cfg_tmin, cfg_tmax], dim=-1)
        g_emb = self.guidance_tokenizer(guidance_input)
        guidance_tokens = g_emb.view(B, self.num_guidance_tokens, self.hidden_dim)
        
        prefix = torch.cat([ctx_tokens, time_tokens, r_tokens, interval_tokens, guidance_tokens], dim=1)
        h = torch.cat([prefix, h], dim=1)
        
        for layer in self.layers:
            h = layer(h)
        
        h = h[:, self.total_prefix_tokens:, :]
        
        # Hilbert inverse: hilbert → raster (trả output về thứ tự gốc)
        if self.use_hilbert_ordering:
            h = h[:, self._raster_to_hilbert, :]
        
        return self.output_norm(h)

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
        
        Tham số:
            x_t: Các token tiềm ẩn mang nhiễu [B, L, input_dim]
            t: Dấu thời gian kết thúc [B]
            context: Vector điều kiện hóa [B, context_dim]
            r: Dấu thời gian bắt đầu [B], mặc định None -> r=t (điều kiện biên)
            omega: Thang đo điều hướng [B], mặc định 1.0
            cfg_tmin: Bắt đầu khoảng CFG [B], mặc định 0.0
            cfg_tmax: Kết thúc khoảng CFG [B], mặc định 1.0
            return_hidden: Nếu True, trả về tuple (velocity, hidden_state) để tái sử dụng
                           cho v-head phụ trợ mà không cần forward pass thêm.
            
        Trả về:
            Vận tốc được dự đoán [B, L, input_dim], hoặc tuple (velocity, hidden) nếu return_hidden=True
        """
        h = self._forward_core(x_t, t, context, r=r, omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax)
        velocity = self.output_proj(h)
        
        if return_hidden:
            return velocity, h
        return velocity
    
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
        """
        Lấy trạng thái ẩn trước khi thực hiện phép chiếu đầu ra (dành cho v-head phụ trợ).
        Sử dụng chung _forward_core() với hàm forward() nhằm duy trì tính nhất quán phân phối.
        
        Lưu ý: Nếu bạn đã gọi forward(return_hidden=True), hãy tái sử dụng hidden state
        từ đó thay vì gọi lại hàm này để tránh forward pass thừa.
        
        Tham số:
            x_t: [B, L, input_dim]
            t: [B]
            context: [B, context_dim]
            r: [B] dấu thời gian bắt đầu, mặc định bằng t (điều kiện biên)
            omega: [B] thang đo điều hướng, mặc định là 1.0
            cfg_tmin: [B] điểm bắt đầu khoảng điều hướng, mặc định là 0.0
            cfg_tmax: [B] điểm kết thúc khoảng điều hướng, mặc định là 1.0
            
        Trả về:
            Trạng thái ẩn [B, L, hidden_dim]
        """
        return self._forward_core(x_t, t, context, r=r, omega=omega, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax)


# ============================================================
# Hàm Kiểm thử (Test function)
# ============================================================
def test_voxel_mamba():
    """Kiểm thử quá trình lan truyền xuôi của mạng VoxelMamba."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[VoxelMamba Test] Device: {device}")
    
    # Cấu hình
    batch_size = 2
    slat_length = 4096
    input_dim = 32
    context_dim = 946
    
    # Khởi tạo mô hình
    model = VoxelMamba(
        input_dim=input_dim,
        hidden_dim=512,
        num_layers=6,  # Kích thước nhỏ hơn để kiểm thử nhanh
        slat_length=slat_length,
        context_dim=context_dim,
    ).to(device)
    
    # Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[VoxelMamba] Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Tạo dữ liệu giả lập (dummy inputs)
    x_t = torch.randn(batch_size, slat_length, input_dim, device=device)
    t = torch.rand(batch_size, device=device)
    context = torch.randn(batch_size, context_dim, device=device)
    omega = torch.ones(batch_size, device=device) * 2.0
    
    # Lan truyền xuôi (Forward pass)
    print("[VoxelMamba] Testing forward pass...")
    with torch.no_grad():
        output = model(x_t, t, context, omega=omega)
    
    print(f"[VoxelMamba] Input shape: {x_t.shape}")
    print(f"[VoxelMamba] Output shape: {output.shape}")
    print(f"[VoxelMamba] Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Kiểm thử trích xuất trạng thái ẩn
    print("[VoxelMamba] Testing hidden state extraction...")
    with torch.no_grad():
        hidden = model.get_hidden_state(x_t, t, context)
    print(f"[VoxelMamba] Hidden state shape: {hidden.shape}")
    
    # Kiểm tra xem đang sử dụng Mamba hay GRU
    if MAMBA_AVAILABLE:
        print("[VoxelMamba] ✓ Using CUDA-accelerated Mamba")
    else:
        print("[VoxelMamba] ⚠ Using fallback GRU (install mamba-ssm for speedup)")
    
    print("[VoxelMamba] Test passed!\n")
    return model


if __name__ == "__main__":
    test_voxel_mamba()
