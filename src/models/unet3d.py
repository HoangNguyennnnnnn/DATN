"""3D UNet backbone cho Stage 2 (flow matching trên Slat latents).

Slat [B, 4096, C] = grid 16³×C (raster) → conv3d.
Conditioning: time qua FiLM (global) + context qua CROSS-ATTENTION (per-position,
ở 8³ và bottleneck 4³). Cross-attn fix "mean face trap" của FiLM-only ở scale.
Interface khớp ImprovedMeanFlow.compute_loss: forward(z_t, t, context, r, omega, cfg_*).
r/omega/cfg nhận-và-bỏ-qua (standard FM v-pred); giữ để tương thích sampler.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class _ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        s, b = self.film(cond)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + s) + b
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class _CrossAttn3D(nn.Module):
    """Spatial voxels (query) attend to context tokens (key/value) — context per-position."""

    def __init__(self, ch: int, ctx_token_dim: int, heads: int = 8, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.q = nn.Linear(ch, ch)
        self.kv = nn.Linear(ctx_token_dim, ch * 2)
        self.proj = nn.Linear(ch, ch)
        # Small NON-zero init: zero-init làm gate cross-attn mở quá chậm → conditioning collapse.
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W]  ctx_tokens: [B,K,ctx_token_dim]
        B, C, D, H, W = x.shape
        h = self.norm(x).reshape(B, C, D * H * W).transpose(1, 2)  # [B,N,C]
        q = self.q(h)
        k, v = self.kv(ctx_tokens).chunk(2, dim=-1)
        nh = self.heads
        q = q.reshape(B, -1, nh, C // nh).transpose(1, 2)
        k = k.reshape(B, -1, nh, C // nh).transpose(1, 2)
        v = v.reshape(B, -1, nh, C // nh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, D * H * W, C)
        o = self.proj(o).transpose(1, 2).reshape(B, C, D, H, W)
        return x + o


class VoxelUNet3D(nn.Module):
    arch = "unet3d"

    def __init__(
        self,
        input_dim: int = 32,
        context_dim: int = 946,
        base: int = 128,
        cond_dim: int = 512,
        grid_size: int = 16,
        context_use_arcface_only: bool = False,
        num_ctx_tokens: int = 16,
        attn_heads: int = 8,
        context_whiten_path: Optional[str] = None,
        **_ignored,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.grid_size = int(grid_size)
        self.base = int(base)
        self.cond_dim = int(cond_dim)
        self.context_use_arcface_only = bool(context_use_arcface_only)
        self.num_ctx_tokens = int(num_ctx_tokens)

        # Context whitening (Bước 2): PCA-whiten context để mọi chiều unit variance,
        # loại FLAME-constant + giảm DINOv2 lấn át → off-diag cos 0.72→0.13 (phân biệt identity rõ).
        self.context_whiten_path = context_whiten_path
        self._has_whiten = False
        if context_whiten_path:
            wd = torch.load(context_whiten_path, map_location="cpu", weights_only=False)
            self.register_buffer("_whiten_mean", wd["mean"].float(), persistent=True)
            self.register_buffer("_whiten_W", wd["W"].float(), persistent=True)  # [out_dim, 946]
            self._has_whiten = True
            ctx_in = int(wd["out_dim"])
        elif self.context_use_arcface_only:
            ctx_in = 512
        else:
            ctx_in = self.context_dim
        self._ctx_in = ctx_in

        self.t_mlp = nn.Sequential(
            nn.Linear(256, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.ctx_tokenizer = nn.Sequential(
            nn.Linear(ctx_in, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, self.num_ctx_tokens * cond_dim),
        )

        c1, c2, c3 = base, base * 2, base * 4
        C = self.input_dim
        self.in_conv = nn.Conv3d(C, c1, 3, padding=1)
        self.d1 = _ResBlock3D(c1, c1, cond_dim)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)   # 16->8
        self.d2 = _ResBlock3D(c2, c2, cond_dim)
        self.attn_d2 = _CrossAttn3D(c2, cond_dim, heads=attn_heads)   # 8³
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)   # 8->4
        self.mid1 = _ResBlock3D(c3, c3, cond_dim)
        self.attn_mid = _CrossAttn3D(c3, cond_dim, heads=attn_heads)  # 4³
        self.mid2 = _ResBlock3D(c3, c3, cond_dim)
        self.up2 = _ResBlock3D(c3 + c2, c2, cond_dim)
        self.attn_up2 = _CrossAttn3D(c2, cond_dim, heads=attn_heads)  # 8³ decoder
        self.up1 = _ResBlock3D(c2 + c1, c1, cond_dim)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv3d(c1, C, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _to_grid(self, s: torch.Tensor) -> torch.Tensor:
        B = s.shape[0]
        g = self.grid_size
        return s.transpose(1, 2).reshape(B, self.input_dim, g, g, g)

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.reshape(B, self.input_dim, self.grid_size ** 3).transpose(1, 2)

    def _prep_context(self, context: torch.Tensor) -> torch.Tensor:
        context = context.float()
        if self._has_whiten:
            # CFG dropout zero-out context GỐC → giữ "null" = 0 sau whiten (không để bias mean trôi).
            is_null = (context.abs().sum(dim=-1, keepdim=True) < 1e-6)
            cw = (context - self._whiten_mean) @ self._whiten_W.t()  # [B, out_dim]
            cw = torch.where(is_null, torch.zeros_like(cw), cw)
            return cw
        if self.context_use_arcface_only and context.shape[-1] > 512:
            context = context[..., :512]
        if context.shape[-1] != self._ctx_in:
            if context.shape[-1] > self._ctx_in:
                context = context[..., : self._ctx_in]
            else:
                context = F.pad(context, (0, self._ctx_in - context.shape[-1]))
        return context

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        ctx = self._prep_context(context)
        time_cond = self.t_mlp(_sinusoidal(t, 256))
        ctx_tokens = self.ctx_tokenizer(ctx).reshape(ctx.shape[0], self.num_ctx_tokens, self.cond_dim)
        x = self._to_grid(z_t)
        h0 = self.d1(self.in_conv(x), time_cond)
        h1 = self.d2(self.down1(h0), time_cond)
        h1 = self.attn_d2(h1, ctx_tokens)
        h2 = self.mid1(self.down2(h1), time_cond)
        h2 = self.attn_mid(h2, ctx_tokens)
        h2 = self.mid2(h2, time_cond)
        u2 = F.interpolate(h2, scale_factor=2, mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h1], dim=1), time_cond)
        u2 = self.attn_up2(u2, ctx_tokens)
        u1 = F.interpolate(u2, scale_factor=2, mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h0], dim=1), time_cond)
        out = self.out_conv(F.silu(self.out_norm(u1)))
        out = self._to_seq(out)
        if return_hidden:
            return out, None
        return out


def voxel_unet3d_from_stage2_config(mcfg: dict, **overrides) -> "VoxelUNet3D":
    kw = dict(
        input_dim=mcfg.get("input_dim", 32),
        context_dim=mcfg.get("context_dim", 946),
        base=mcfg.get("base", 128),
        cond_dim=mcfg.get("cond_dim", 512),
        grid_size=mcfg.get("grid_size", 16),
        context_use_arcface_only=mcfg.get("context_use_arcface_only", False),
        num_ctx_tokens=mcfg.get("num_ctx_tokens", 16),
        context_whiten_path=mcfg.get("context_whiten_path", None),
    )
    kw.update(overrides)
    return VoxelUNet3D(**kw)
