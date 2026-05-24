"""Utilities for stabilizing VoxelMamba context cross-attention."""
from __future__ import annotations

import torch.nn as nn


def cross_attn_proj_stats(model: nn.Module) -> dict[str, float]:
    norms = []
    for name, p in model.named_parameters():
        if "cross_attn.proj.weight" in name:
            norms.append(float(p.detach().float().norm().item()))
    if not norms:
        return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": len(norms),
        "mean": sum(norms) / len(norms),
        "max": max(norms),
        "min": min(norms),
    }


def renormalize_cross_attn_proj_weights(model: nn.Module, max_norm: float = 1.0) -> int:
    """Scale down exploded cross_attn.proj weights (e.g. after LR boost). Returns #tensors clipped."""
    if max_norm <= 0:
        return 0
    n_fixed = 0
    for name, p in model.named_parameters():
        if "cross_attn.proj.weight" not in name:
            continue
        n = p.data.norm()
        if n > max_norm:
            p.data.mul_(max_norm / (n + 1e-8))
            n_fixed += 1
    return n_fixed
