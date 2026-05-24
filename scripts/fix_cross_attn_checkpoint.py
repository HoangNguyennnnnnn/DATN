#!/usr/bin/env python3
"""Renormalize cross_attn.proj in checkpoint + add context_gate keys for new arch."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.voxel_mamba import voxel_mamba_from_stage2_config
from src.cross_attn_utils import cross_attn_proj_stats, renormalize_cross_attn_proj_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="Checkpoint paths to patch in-place")
    ap.add_argument("--max-norm", type=float, default=1.0)
    args = ap.parse_args()

    for path in args.ckpt:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        mcfg = ckpt.get("stage2_model_config", {})
        model = voxel_mamba_from_stage2_config(mcfg, dropout=0.0)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n{path}")
        print(f"  before: {cross_attn_proj_stats(model)}")
        n = renormalize_cross_attn_proj_weights(model, max_norm=args.max_norm)
        print(f"  clipped: {n}  after: {cross_attn_proj_stats(model)}")
        ckpt["model_state_dict"] = model.state_dict()
        if "ema_state_dict" in ckpt:
            ema_sd = ckpt["ema_state_dict"]
            for k in list(ema_sd.keys()):
                if "cross_attn.proj.weight" in k:
                    w = ema_sd[k]
                    norm = w.float().norm()
                    if norm > args.max_norm:
                        ema_sd[k] = w * (args.max_norm / (norm + 1e-8))
            ckpt["ema_state_dict"] = ema_sd
        torch.save(ckpt, path)
        print(f"  saved.")


if __name__ == "__main__":
    main()
