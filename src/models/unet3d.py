"""3D UNet backbone cho Stage 2 (iMF / flow matching trên Slat latents).

Thay VoxelMamba: slat [B, 4096, C] = grid 16³×C (raster) → conv3d.
Validate trên overfit harness (29/05): UNet sinh được identity-specific (cos>0.9),
VoxelMamba thì không (cos 0.03) — conv có inductive bias không gian + data-efficient.

Conditioning: time + hybrid context (946) qua FiLM (kiểu DDPM class-cond).
Interface khớp `ImprovedMeanFlow.compute_loss`: forward(z_t, t, context, r, omega, cfg_*).
r/omega/cfg nhận-và-bỏ-qua (standard flow matching v-pred); giữ để tương thích sampler.
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


class VoxelUNet3D(nn.Module):
    """3D UNet trên grid 16³. Output velocity (v-pred) cùng shape với input slat.

    Args:
        input_dim: số kênh slat (C). grid = grid_size³ tokens.
        context_dim: chiều context (946 hybrid / 512 arcface).
        base: số kênh gốc (16³ rất nhỏ nên rẻ); mults nội bộ [1,2,4].
        grid_size: 16 (16³ = 4096).
    """

    arch = "unet3d"

    def __init__(
        self,
        input_dim: int = 32,
        context_dim: int = 946,
        base: int = 128,
        cond_dim: int = 512,
        grid_size: int = 16,
        context_use_arcface_only: bool = False,
        **_ignored,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.grid_size = int(grid_size)
        self.base = int(base)
        self.cond_dim = int(cond_dim)
        self.context_use_arcface_only = bool(context_use_arcface_only)
        ctx_in = 512 if self.context_use_arcface_only else self.context_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(256, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(ctx_in, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self._ctx_in = ctx_in

        c1, c2, c3 = base, base * 2, base * 4
        C = self.input_dim
        self.in_conv = nn.Conv3d(C, c1, 3, padding=1)
        self.d1 = _ResBlock3D(c1, c1, cond_dim)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)   # 16->8
        self.d2 = _ResBlock3D(c2, c2, cond_dim)
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)   # 8->4
        self.mid1 = _ResBlock3D(c3, c3, cond_dim)
        self.mid2 = _ResBlock3D(c3, c3, cond_dim)
        self.up2 = _ResBlock3D(c3 + c2, c2, cond_dim)
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

    def _cond(self, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.context_use_arcface_only and context.shape[-1] > 512:
            context = context[..., :512]
        if context.shape[-1] != self._ctx_in:
            # an toàn: pad/cắt để khớp (vd context truyền vào lệch nhẹ)
            if context.shape[-1] > self._ctx_in:
                context = context[..., : self._ctx_in]
            else:
                context = F.pad(context, (0, self._ctx_in - context.shape[-1]))
        return self.t_mlp(_sinusoidal(t, 256)) + self.c_mlp(context.float())

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
        # r/omega/cfg bỏ qua (standard FM v-pred). return_hidden không hỗ trợ (no v-head path).
        cond = self._cond(t, context)
        x = self._to_grid(z_t)
        h0 = self.d1(self.in_conv(x), cond)
        h1 = self.d2(self.down1(h0), cond)
        h2 = self.mid2(self.mid1(self.down2(h1), cond), cond)
        u2 = F.interpolate(h2, scale_factor=2, mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h1], dim=1), cond)
        u1 = F.interpolate(u2, scale_factor=2, mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h0], dim=1), cond)
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
    )
    kw.update(overrides)
    return VoxelUNet3D(**kw)
