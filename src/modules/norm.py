"""
LayerNorm32: LayerNorm that computes in FP32 for numerical stability
Adapted from TRELLIS.2
"""
import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x):
        # x can be SparseTensor or Tensor
        if hasattr(x, 'features'):
            orig_dtype = x.features.dtype
            # Always compute in FP32 for stability
            feats_fp32 = x.features.float()
            normed = super().forward(feats_fp32)
            return x.replace(normed.to(orig_dtype))
        else:
            return super().forward(x)