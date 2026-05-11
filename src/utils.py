"""
FaceDiff — Shared Utilities
============================
Common helper functions used by both SC-VAE and iMF training scripts.
Eliminates code duplication across train_sc_vae.py and train_imf.py.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Set


def load_identity_set(ids_file: str) -> Optional[Set[str]]:
    """Load identity IDs from text file (one ID per line).
    
    Returns None if file path is empty or not found.
    Used for train/test split filtering in both SC-VAE and iMF training.
    """
    if not ids_file:
        return None
    if not os.path.isfile(ids_file):
        print(f"[Split] IDs file not found, skip filter: {ids_file}")
        return None
    ids = set()
    with open(ids_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                ids.add(token)
    print(f"[Split] Loaded {len(ids)} IDs from {ids_file}")
    return ids


def extract_identity_from_obj_path(
    obj_path: str, data_root: str, dataset_name: str
) -> str:
    """Infer identity token from mesh path for split filtering.
    
    FaceVerse: basename split on '_' → first token.
    FaceScape: finds the numeric directory component in the path.
    
    Returns: identity string (e.g. '64', '125').
    """
    rel_path = os.path.relpath(obj_path, data_root)
    basename = os.path.splitext(os.path.basename(obj_path))[0]
    rel_parts = rel_path.split(os.path.sep)

    if dataset_name == "faceverse":
        # FaceVerse format: ID_Expression/basename.obj OR ID_Expression_...
        # We try to find the first numeric part in rel_parts or basename
        token = basename.split("_", 1)[0]
        for part in rel_parts:
            p = part.split("_")[0]
            if p.isdigit():
                token = p
                break
    else:
        # FaceScape nested format: <id>/models_reg/*.obj OR <trainset>/<id>/models_reg/*.obj
        # Default fallback to expression number (not ideal but kept for compatibility)
        token = basename.split("_", 1)[0]
        # Robust search for numeric ID in path parts (FaceScape IDs are directory names)
        for part in rel_parts:
            if part.isdigit():
                token = part
                break

    token = token.strip()
    if token.isdigit():
        token = str(int(token))
    return token


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (update3.md §4.5).
    
    Faster than LayerNorm because it doesn't calculate the mean (no mean-centering).
    Effective for stabilizing attention in long sequences or high-dimensional latent tokens.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight
