"""8-layer auxiliary v-head theo iMF paper (arXiv:2512.02012, Table 4: aux-head depth = 8).

V-head dự đoán marginal velocity v(z_t) từ hidden state của backbone, được dùng:
  - JVP tangent vector cho compound function V = u + (t-r)·du/dt
  - Auxiliary loss ||v_θ − (e−x)||² (iMF Appendix A)

Kiến trúc: 8 layer × pre-norm MLP block (RMSNorm → Linear → SiLU → Linear → residual).
Output: zero-init projection để initial v-head output = 0 (boundary identity).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms * self.weight


class VHeadBlock(nn.Module):
    """Pre-norm MLP block: x + fc2(SiLU(fc1(norm(x))))"""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class VHead(nn.Module):
    """8-layer auxiliary v-head theo iMF paper Table 4.

    Args:
        hidden_dim: chiều ẩn của backbone (e.g., cond_dim 512 cho UNet3D)
        out_dim:    chiều output (e.g., 32 = slat latent dim)
        depth:      số block, paper dùng 8
        mlp_ratio:  expansion ratio trong MLP block
    """

    def __init__(self, hidden_dim: int, out_dim: int, depth: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [VHeadBlock(hidden_dim, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm_out = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm_out(x))
