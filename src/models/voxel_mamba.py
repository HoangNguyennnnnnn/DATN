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
from typing import Optional, List, Tuple

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
        half_dim = max(self.dim // 2, 1)
        if half_dim == 1:
            frequencies = torch.ones(1, device=device)
        else:
            scale = math.log(10000) / (half_dim - 1)
            frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = t[:, None] * frequencies[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if embeddings.shape[-1] < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))
        return embeddings[:, :self.dim]


class FeedForward(nn.Module):
    """Per-token nonlinear transformation (MLP) — CRITICAL missing component.

    2026-05-21 v3: Added to fix memorization failure. Without FFN, Mamba can
    only do linear sequential mixing (recurrence). FFN provides:
    - Feature expansion: dim → ffn_dim (4x) → dim
    - Nonlinearity: GELU activation
    - Per-token independence: each token transformed separately

    This is the standard design from DiT, DiM-3D, and all Transformer-based
    diffusion models. Mamba handles inter-token mixing, FFN handles
    intra-token feature transformation. Both are essential.
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        ffn_dim = dim * expand
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        # Zero-init last linear → FFN starts as identity (safe for residual)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextCrossAttention(nn.Module):
    """Cross-attn: slat tokens (Q) attend ArcFace context tokens (K/V). ~1M params/layer."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(ctx_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return x + self.proj(out)


class BidirectionalMambaBlock(nn.Module):
    """
    Khối SSM Hai chiều + FFN + Dual AdaLN Conditioning.

    2026-05-21 v4: FFN + Context AdaLN (replaces cross-attention).
    - Mamba: inter-token sequential mixing (spatial)
    - FFN: intra-token nonlinear feature transformation (CRITICAL)
    - Time AdaLN: modulate both sub-blocks with (t, r) info
    - Context AdaLN: modulate Mamba output with identity/expression info

    Architecture per block:
        # Sub-block 1: Mamba + Context modulation
        x_mod = norm(x) * (1 + scale_t) + shift_t    # Time AdaLN
        out = bimamba(x_mod)                           # Spatial mixing
        out = out * (1 + scale_c) + shift_c            # Context AdaLN
        x = x + gate_t * out                           # Gated residual

        # Sub-block 2: FFN
        x_ffn = norm_ffn(x) * (1 + scale_f) + shift_f # Time AdaLN
        x = x + gate_ffn * ffn(x_ffn)                 # Gated residual
    """
    def __init__(
        self,
        dim,
        time_dim,
        ctx_dim,
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_expand=4,
        dropout=0.0,
        use_mamba=True,
        use_ffn=True,
        backend: str = "auto",
        context_cond_mode: str = "cross_attn",
        cross_attn_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.context_cond_mode = str(context_cond_mode).strip().lower()
        self.backend = _resolve_requested_backend(backend)
        self.use_mamba = use_mamba
        self.use_ffn = use_ffn

        if self.use_mamba:
            self.forward_mamba = Mamba(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand,
            )
            self.backward_mamba = Mamba(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand,
            )
        else:
            self.gru = nn.GRU(
                input_size=dim, hidden_size=dim // 2,
                num_layers=1, bidirectional=True, batch_first=True,
            )

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Time-only AdaLN: time_cond [B, time_dim] → (scale, shift, gate) [B, dim] × 3
        # Time signal is monotonic and high-variance → suitable for AdaLN modulation
        self.adaLN_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 3 * dim, bias=True),
        )
        # Init: scale small non-zero, shift zero, gate bias = 1.0 (NOT zero!)
        # Gate bias = 1.0 → Mamba receives full gradient from step 1 → breaks gate=0 trap
        last_linear = self.adaLN_time[-1]
        W = last_linear.weight
        nn.init.normal_(W[:dim], mean=0.0, std=0.02)      # scale weights
        nn.init.zeros_(W[dim : 2 * dim])                   # shift weights
        nn.init.zeros_(W[2 * dim :])                       # gate weights
        nn.init.zeros_(last_linear.bias[:dim])             # scale bias
        nn.init.zeros_(last_linear.bias[dim : 2 * dim])   # shift bias
        nn.init.constant_(last_linear.bias[2 * dim :], 1.0)  # gate bias = 1.0

        if self.context_cond_mode == "cross_attn":
            self.cross_attn = ContextCrossAttention(
                dim, num_heads=cross_attn_heads, dropout=dropout,
            )
            self.adaLN_ctx = None
        else:
            self.cross_attn = None
            self.adaLN_ctx = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ctx_dim, 2 * dim, bias=True),
            )
            ctx_linear = self.adaLN_ctx[-1]
            nn.init.normal_(ctx_linear.weight[:dim], mean=0.0, std=0.1)
            nn.init.zeros_(ctx_linear.weight[dim:])
            nn.init.zeros_(ctx_linear.bias)

        # ============================================================
        # SUB-BLOCK 2: FFN (Per-token nonlinear transform)
        # 2026-05-21 v3: The CRITICAL missing component.
        # Without FFN, model cannot learn per-token features independently.
        # ============================================================
        self.norm_ffn = RMSNorm(dim)
        self.ffn = FeedForward(dim=dim, expand=ffn_expand, dropout=dropout)

        # FFN also gets Time AdaLN (same as DiT paper — same conditioning for both sub-blocks)
        self.adaLN_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 3 * dim, bias=True),
        )
        # Same init pattern: scale small, shift zero, gate bias = 1.0
        last_ffn = self.adaLN_ffn[-1]
        W_ffn = last_ffn.weight
        nn.init.normal_(W_ffn[:dim], mean=0.0, std=0.02)
        nn.init.zeros_(W_ffn[dim : 2 * dim])
        nn.init.zeros_(W_ffn[2 * dim :])
        nn.init.zeros_(last_ffn.bias[:dim])
        nn.init.zeros_(last_ffn.bias[dim : 2 * dim])
        nn.init.constant_(last_ffn.bias[2 * dim :], 1.0)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        ctx_cond: Optional[torch.Tensor] = None,
        ctx_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, L, dim] hidden states
        time_cond: [B, time_dim] time conditioning (t, r, interval, guidance)
        ctx_cond: [B, ctx_dim] — AdaLN path only
        ctx_tokens: [B, K, dim] — cross-attn path only
        """
        # ── Sub-block 1: Mamba + Context Cross-Attention ──
        scale_t, shift_t, gate_t = self.adaLN_time(time_cond).chunk(3, dim=-1)
        scale_t = scale_t.unsqueeze(1)  # [B, 1, dim]
        shift_t = shift_t.unsqueeze(1)
        gate_t = gate_t.unsqueeze(1)

        residual = x
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale_t) + shift_t

        if self.use_mamba:
            fwd = self.forward_mamba(x_mod)
            bwd_input = torch.flip(x_mod, dims=[1])
            bwd = self.backward_mamba(bwd_input)
            bwd = torch.flip(bwd, dims=[1])
            out = fwd + bwd
        else:
            out, _ = self.gru(x_mod)

        out = self.dropout(out)

        if self.cross_attn is not None:
            if ctx_tokens is None:
                raise ValueError("cross_attn block requires ctx_tokens [B, K, dim]")
            out = self.cross_attn(out, ctx_tokens)
        else:
            if ctx_cond is None:
                raise ValueError("adaln block requires ctx_cond [B, ctx_dim]")
            scale_c, shift_c = self.adaLN_ctx(ctx_cond).chunk(2, dim=-1)
            scale_c = scale_c.unsqueeze(1)
            shift_c = shift_c.unsqueeze(1)
            out = out * (1 + scale_c) + shift_c

        x = residual + gate_t * out

        # ── Sub-block 2: FFN (per-token nonlinear transform) ──
        if self.use_ffn:
            scale_f, shift_f, gate_f = self.adaLN_ffn(time_cond).chunk(3, dim=-1)
            scale_f = scale_f.unsqueeze(1)
            shift_f = shift_f.unsqueeze(1)
            gate_f = gate_f.unsqueeze(1)
    
            x_ffn = self.norm_ffn(x) * (1 + scale_f) + shift_f
            x = x + gate_f * self.ffn(x_ffn)
            
        return x


def voxel_mamba_from_stage2_config(mcfg: dict, **overrides) -> "VoxelMamba":
    """Build VoxelMamba from checkpoint ``stage2_model_config`` (+ optional overrides)."""
    ctx_mode = mcfg.get("context_cond_mode", "adaln")
    arc_only = bool(mcfg.get("context_use_arcface_only", ctx_mode == "cross_attn"))
    seg_w = mcfg.get("context_segment_weights")
    if seg_w is not None and len(seg_w) == 3:
        seg_w = tuple(float(x) for x in seg_w)
    elif arc_only:
        seg_w = None
    defaults = dict(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        slat_length=mcfg["slat_length"],
        context_dim=mcfg["context_dim"],
        backend=mcfg.get("backend", "auto"),
        strict=False,
        num_context_tokens=mcfg.get("num_context_tokens", 0 if ctx_mode == "cross_attn" else 8),
        num_time_tokens=mcfg.get("num_time_tokens", 4),
        num_r_tokens=mcfg.get("num_r_tokens", 4),
        num_interval_tokens=mcfg.get("num_interval_tokens", 4),
        num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
        use_per_layer_context=bool(mcfg.get("use_per_layer_context", False)),
        d_state=mcfg.get("d_state", 16),
        d_conv=mcfg.get("d_conv", 4),
        expand=mcfg.get("expand", 2),
        ffn_expand=int(mcfg.get("ffn_expand", 4)),
        dropout=mcfg.get("dropout", 0.0),
        context_segment_weights=seg_w,
        context_cond_mode=ctx_mode,
        context_use_arcface_only=arc_only,
        num_context_kv_tokens=int(mcfg.get("num_context_kv_tokens", 8)),
        context_cross_attn_heads=int(mcfg.get("context_cross_attn_heads", 8)),
    )
    defaults.update(overrides)
    return VoxelMamba(**defaults)


class VoxelMamba(nn.Module):
    """
    Mạng cơ sở Voxel Mamba (Voxel Mamba Backbone) dành cho việc sinh token Slat.
    
    Thay thế mạng IMFUNet1D với độ phức tạp O(N).
    
    Kiến trúc (v8 — ArcFace cross-attn, mặc định):
    1. Nhúng đầu vào + Hilbert reorder
    2. Prefix time/r/interval/guidance (không prefix context — tránh trùng cross-attn)
    3. ArcFace [512] → K context tokens → cross-attn sau Mamba mỗi layer
    4. Time AdaLN trước Mamba; CFG học nhánh null context
    5. output_proj → velocity [B, 4096, input_dim]
    
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
        use_per_layer_context: bool = False,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        use_hilbert_ordering: bool = True,
        use_ffn: bool = True,
        context_segment_weights: Optional[Tuple[float, float, float]] = None,
        context_arc_dim: int = 512,
        context_flame_dim: int = 50,
        context_cond_mode: str = "cross_attn",
        context_use_arcface_only: bool = True,
        num_context_kv_tokens: int = 8,
        context_cross_attn_heads: int = 8,
    ):
        super().__init__()
        self.context_cond_mode = str(context_cond_mode).strip().lower()
        self.context_use_arcface_only = bool(context_use_arcface_only)
        self.num_context_kv_tokens = int(num_context_kv_tokens)
        self.context_cross_attn_heads = int(context_cross_attn_heads)
        self.context_arc_dim = int(context_arc_dim)
        self.context_flame_dim = int(context_flame_dim)
        self._effective_arc_dim = (
            min(self.context_arc_dim, int(context_dim))
            if self.context_use_arcface_only
            else int(context_dim)
        )
        if self.context_cond_mode == "cross_attn":
            num_context_tokens = 0
            use_per_layer_context = False
        if context_segment_weights is not None:
            w = torch.tensor(context_segment_weights, dtype=torch.float32)
            self.register_buffer("_context_segment_weights", w, persistent=True)
        else:
            self._context_segment_weights = None
        
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
        # 2026-05-19: Refactored từ 1-layer Linear+SiLU → 2-layer MLP với residual-style
        # expansion. Diagnostic ep59 phát hiện 1-layer Linear không đủ capacity preserve
        # 946-dim context diversity → output tokens cos_sim 0.94 across IDs (vs input 0.87).
        # 2-layer MLP cho phép non-linear features, RMSNorm + larger intermediate stabilize.
        self.context_tokenizer = (
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim * 2),                  # 946 → 1024
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * num_context_tokens),  # 1024 → 4096
            )
            if num_context_tokens > 0 else None
        )
        
        self.time_tokenizer = (
            nn.Sequential(
                nn.Linear(self.time_embed_dim, hidden_dim * num_time_tokens),
                nn.SiLU(),
            )
            if num_time_tokens > 0 else None
        )
        
        self.r_tokenizer = (
            nn.Sequential(
                nn.Linear(self.time_embed_dim, hidden_dim * num_r_tokens),
                nn.SiLU(),
            )
            if num_r_tokens > 0 else None
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
        self.interval_tokenizer = (
            nn.Sequential(
                nn.Linear(self.time_embed_dim, hidden_dim * num_interval_tokens),
                nn.SiLU(),
            )
            if num_interval_tokens > 0 else None
        )
        
        self.guidance_tokenizer = (
            nn.Sequential(
                nn.Linear(3, hidden_dim * num_guidance_tokens),  # [omega, t_min, t_max]
                nn.SiLU(),
            )
            if num_guidance_tokens > 0 else None
        )

        if self.context_cond_mode == "cross_attn":
            self.context_cond_mlp = None
            self.arcface_tokenizer = nn.Sequential(
                nn.Linear(self._effective_arc_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * self.num_context_kv_tokens),
            )
            self.null_ctx_tokens = nn.Parameter(
                torch.zeros(1, self.num_context_kv_tokens, hidden_dim),
            )
            nn.init.normal_(self.null_ctx_tokens, mean=0.0, std=0.02)
        else:
            self.arcface_tokenizer = None
            self.null_ctx_tokens = None
            self.context_cond_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self._time_guidance_in_dim = 3 * self.time_embed_dim + 3
        self.time_guidance_mlp = nn.Sequential(
            nn.Linear(self._time_guidance_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._time_dim = hidden_dim
        self._ctx_dim = hidden_dim
        self.use_per_layer_context = bool(use_per_layer_context)
        if self.use_per_layer_context:
            self.ctx_layer_projs = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                )
                for _ in range(num_layers)
            ])
            for proj in self.ctx_layer_projs:
                last = proj[-1]
                nn.init.normal_(last.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last.bias)
        else:
            self.ctx_layer_projs = None

        # Ngăn xếp các khối Mamba hai chiều + FFN với Dual AdaLN Conditioning
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                dim=hidden_dim,
                time_dim=self._time_dim,
                ctx_dim=self._ctx_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                ffn_expand=ffn_expand,
                use_mamba=self.use_mamba,
                use_ffn=use_ffn,
                context_cond_mode=self.context_cond_mode,
                cross_attn_heads=self.context_cross_attn_heads,
            )
            for _ in range(num_layers)
        ])
        mode_label = (
            f"cross_attn K={self.num_context_kv_tokens} arc_only"
            if self.context_cond_mode == "cross_attn"
            else "adaln"
        )
        print(f"[VoxelMamba] context_cond_mode={mode_label}")
        
        # Chuẩn hóa (norm) và phép chiếu (projection) cuối cùng
        self.output_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # 2026-05-21 CRITICAL FIX: output_proj MUST NOT be zero-initialized!
        # Zero W_out → gradient = W_out^T @ grad = 0 → backbone starved.
        # 2026-05-22 Paper alignment: iMF Appendix A uses N(0, σ²) with σ²=0.1/fan_in
        # for all linear layers (except zero-init last residual layer).
        # Old: xavier_uniform_(gain=0.02) → σ≈0.0009 (too small, 16x below paper).
        # New: N(0, sqrt(0.1/fan_in)) → σ≈0.014 for fan_in=512.
        nn.init.normal_(self.output_proj.weight, std=math.sqrt(0.1 / self.output_proj.in_features))
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
        
    def _make_prefix_tokens(
        self,
        tokenizer: Optional[nn.Module],
        source: torch.Tensor,
        num_tokens: int,
        batch_size: int,
    ) -> torch.Tensor:
        if tokenizer is None or num_tokens <= 0:
            return source.new_zeros((batch_size, 0, self.hidden_dim))
        return tokenizer(source).view(batch_size, num_tokens, self.hidden_dim)

    def _build_cond_emb(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
        context: torch.Tensor,
        omega: torch.Tensor,
        cfg_tmin: torch.Tensor,
        cfg_tmax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build SEPARATE time and context conditioning embeddings.
        
        2026-05-21 v2: Returns tuple (ctx_cond, time_cond) instead of additive fusion.
        Bug 1 fix: additive fusion caused time_cond (high variance from t∈[0,1]) to
        drown ctx_cond (low variance, static per identity). Separate paths ensure
        each signal reaches its target mechanism with full magnitude.
        
        Returns:
            ctx_cond: [B, ctx_dim] — for cross-attention POST-Mamba
            time_cond: [B, time_dim] — for AdaLN (scale, shift, gate) PRE-Mamba
        """
        t_emb = self.time_mlp(t)
        r_emb = self.r_mlp(r)
        interval_emb = self.interval_mlp(t - r)
        guidance_feat = torch.stack([omega, cfg_tmin, cfg_tmax], dim=-1)
        time_feat = torch.cat([t_emb, r_emb, interval_emb, guidance_feat], dim=-1)
        time_cond = self.time_guidance_mlp(time_feat)
        if self.context_cond_mlp is None:
            return None, time_cond
        ctx_in = self._extract_context_input(context)
        ctx_cond = self.context_cond_mlp(ctx_in)
        return ctx_cond, time_cond

    def _extract_context_input(self, context: torch.Tensor) -> torch.Tensor:
        if self.context_use_arcface_only:
            return context[..., : self._effective_arc_dim]
        return self._scale_context_segments(context)

    def _build_ctx_tokens(self, context: torch.Tensor) -> torch.Tensor:
        arc = self._extract_context_input(context)
        b = arc.shape[0]
        flat = self.arcface_tokenizer(arc)
        tokens = flat.view(b, self.num_context_kv_tokens, self.hidden_dim)
        if self.null_ctx_tokens is not None:
            is_null = arc.abs().sum(dim=-1) < 1e-4
            if is_null.any():
                null = self.null_ctx_tokens.expand(b, -1, -1).to(
                    device=tokens.device, dtype=tokens.dtype,
                )
                mask = is_null.view(b, 1, 1)
                tokens = torch.where(mask, null, tokens)
        return tokens

    def null_context_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.null_ctx_tokens is None:
            raise RuntimeError("null_context_tokens only available in cross_attn mode")
        return self.null_ctx_tokens.expand(batch_size, -1, -1).to(device)

    def _scale_context_segments(self, context: torch.Tensor) -> torch.Tensor:
        """Nhân Arc/FLAME/DINO trước MLP — legacy adaln path."""
        w = self._context_segment_weights
        if w is None:
            return context
        arc, fl = self.context_arc_dim, self.context_flame_dim
        w = w.to(device=context.device, dtype=context.dtype)
        parts = [
            context[..., :arc] * w[0],
            context[..., arc : arc + fl] * w[1],
            context[..., arc + fl :] * w[2],
        ]
        return torch.cat(parts, dim=-1)

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

        if self.use_hilbert_ordering:
            h = h[:, self._hilbert_to_raster, :]

        # v7 (2026-05-20): Hybrid End-prefix strategy.
        # Place prefix tokens at END of sequence (positions slat_length .. slat_length+24).
        # - Forward Mamba scan: data tokens first (Hilbert spatial locality intact),
        #   prefix processed LAST → state ends knowing context
        # - Backward Mamba scan: reversed sequence puts prefix FIRST → backward state
        #   initialized with context info → propagates to ALL data positions
        # → Combined fwd+bwd: data tokens absorb context via backward state.
        if self.total_prefix_tokens > 0:
            t_emb = self.time_mlp(t)
            r_emb = self.r_mlp(r)
            interval_emb = self.interval_mlp(t - r)
            guidance_input = torch.stack([omega, cfg_tmin, cfg_tmax], dim=-1)
            if self.context_tokenizer is not None and self.num_context_tokens > 0:
                ctx_tokens = self._make_prefix_tokens(
                    self.context_tokenizer, context, self.num_context_tokens, B,
                )
            else:
                ctx_tokens = h.new_zeros((B, 0, self.hidden_dim))
            time_tokens = self._make_prefix_tokens(self.time_tokenizer, t_emb, self.num_time_tokens, B)
            r_tokens = self._make_prefix_tokens(self.r_tokenizer, r_emb, self.num_r_tokens, B)
            interval_tokens = self._make_prefix_tokens(self.interval_tokenizer, interval_emb, self.num_interval_tokens, B)
            guidance_tokens = self._make_prefix_tokens(self.guidance_tokenizer, guidance_input, self.num_guidance_tokens, B)
            prefix = torch.cat([ctx_tokens, time_tokens, r_tokens, interval_tokens, guidance_tokens], dim=1)
            # PREFIX AT END (not start) → preserves Hilbert order for data tokens in forward scan
            h = torch.cat([h, prefix], dim=1)

        ctx_cond, time_cond = self._build_cond_emb(t, r, context, omega, cfg_tmin, cfg_tmax)
        cross_ctx = None
        if self.context_cond_mode == "cross_attn":
            cross_ctx = self._build_ctx_tokens(context)

        for i, layer in enumerate(self.layers):
            if cross_ctx is not None:
                h = layer(h, time_cond, ctx_tokens=cross_ctx)
            else:
                ctx_l = self.ctx_layer_projs[i](ctx_cond) if self.ctx_layer_projs is not None else ctx_cond
                h = layer(h, time_cond, ctx_cond=ctx_l)

        if self.total_prefix_tokens > 0:
            # Strip prefix tokens FROM END (not start)
            h = h[:, :self.slat_length, :]
        
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
