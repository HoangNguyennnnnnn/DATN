#!/usr/bin/env python3
"""Phân tích luồng gradient + độ nhạy context trên checkpoint iMF (VoxelMamba + v-head)."""
from __future__ import annotations

import argparse
import io
import os
import sys
from collections import defaultdict

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TrainConfig
from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.v_head import VHead
from src.models.voxel_mamba import voxel_mamba_from_stage2_config
def _group_name(param_name: str) -> str:
    n = param_name
    if n.startswith("layers."):
        parts = n.split(".")
        layer_i = parts[1]
        rest = ".".join(parts[2:])
        if "cross_attn" in rest:
            sub = rest.split("cross_attn.", 1)[-1].split(".")[0]
            return f"layers[{layer_i}].cross_attn.{sub}"
        if "forward_mamba" in rest or "backward_mamba" in rest:
            return f"layers[{layer_i}].mamba"
        if "ffn" in rest:
            return f"layers[{layer_i}].ffn"
        if "ada" in rest.lower() or "time" in rest or "gate" in rest:
            return f"layers[{layer_i}].adaln"
        return f"layers[{layer_i}].other"
    if "arcface_tokenizer" in n or "null_ctx" in n:
        return "context_tokenizer"
    if "output_proj" in n:
        return "output_proj"
    if "output_norm" in n:
        return "output_norm"
    if "input_embed" in n:
        return "input_embed"
    if "time_mlp" in n or "r_mlp" in n or "interval_mlp" in n or "time_tokenizer" in n:
        return "time_cond"
    if "v_head" in n or n.startswith("blocks."):
        return "v_head"
    return n.split(".")[0]


def _collect_grad_norms(model: torch.nn.Module, v_head: torch.nn.Module | None) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for mod, name in [(model, "model"), (v_head, "v_head")]:
        if mod is None:
            continue
        for pname, p in mod.named_parameters():
            if p.grad is None:
                continue
            gnorm = float(p.grad.detach().float().norm().item())
            key = _group_name(pname) if name == "model" else f"v_head.{pname.split('.')[0]}"
            totals[key] += gnorm
            counts[key] += 1
    return {k: totals[k] / max(counts[k], 1) for k in totals}


def _load_batch_lmdb(lmdb_path: str, batch_size: int, device: torch.device):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=64)
    keys = []
    with env.begin() as txn:
        cur = txn.cursor()
        for k, _ in cur:
            if k != b"__meta__":
                keys.append(k)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(keys), size=min(batch_size, len(keys)), replace=False)
    slats, ctxs = [], []
    with env.begin() as txn:
        for i in picks:
            raw = txn.get(keys[i])
            d = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            slats.append(d["slat"].float())
            ctxs.append(d["context"].float())
    slat = torch.stack(slats).to(device)
    ctx = torch.stack(ctxs).to(device)
    return slat, ctx


@torch.no_grad()
def _pred_stats(model, slat, ctx, imf: ImprovedMeanFlow, device):
    b = slat.shape[0]
    e = torch.randn_like(slat)
    t = torch.full((b,), 0.5, device=device)
    z_t = imf._interpolate(slat, e, t)
    u = model(z_t, t, ctx, r=t)
    u_shuffle = model(z_t, t, ctx[torch.randperm(b)], r=t)
    cos = F.cosine_similarity(
        u.reshape(b, -1), u_shuffle.reshape(b, -1), dim=-1
    ).mean().item()
    return {
        "u_norm": float(u.norm().item() / b),
        "u_std": float(u.std().item()),
        "cos_u_ctx_vs_shuffle": cos,
    }


def _one_backward(
    model,
    v_head,
    imf,
    slat,
    ctx,
    *,
    ratio_r_neq_t: float,
    cfg_enabled: bool,
    slat_mean,
    slat_std,
    force_jvp: bool = False,
) -> tuple[dict, dict]:
    device = slat.device
    model.zero_grad(set_to_none=True)
    if v_head is not None:
        v_head.zero_grad(set_to_none=True)

    occ = (slat.norm(dim=-1) > 1e-6).float()
    slat_n = (slat - slat_mean) / slat_std if slat_mean is not None else slat

    old_ratio = imf.ratio_r_neq_t
    if force_jvp:
        imf.ratio_r_neq_t = 1.0  # maximize JVP branch
    else:
        imf.ratio_r_neq_t = ratio_r_neq_t

    out = imf.compute_loss(
        model,
        slat_n,
        ctx,
        v_head=v_head,
        cfg_conditioning=cfg_enabled,
        cfg_context_dropout=0.0,
        return_components=True,
    )
    loss = out["loss"]
    loss.backward()
    grads = _collect_grad_norms(model, v_head)
    imf.ratio_r_neq_t = old_ratio
    if device.type == "cuda":
        torch.cuda.empty_cache()
    comps = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in out.items()}
    return grads, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_v8_lite/latest_step.pt")
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--compare-ckpt", default="", help="Optional earlier ckpt for weight delta")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  GRADIENT FLOW AUDIT — iMF VoxelMamba")
    print("=" * 72)
    print(f"  ckpt={args.ckpt}  device={device}  batch={args.batch_size}\n")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("stage2_model_config", {})
    print(f"  epoch={ckpt.get('epoch')}  step={ckpt.get('global_step')}  loss={ckpt.get('loss', 0):.4f}")

    model = voxel_mamba_from_stage2_config(mcfg, backend=mcfg.get("backend", "auto"), dropout=0.0).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.train()

    hidden = int(mcfg.get("hidden_dim", 512))
    out_dim = int(mcfg.get("input_dim", 32))
    v_head = VHead(hidden, out_dim, depth=8, mlp_ratio=4).to(device)
    if "v_head_state_dict" in ckpt:
        v_head.load_state_dict(ckpt["v_head_state_dict"], strict=True)

    imf = ImprovedMeanFlow(ratio_r_neq_t=0.5)
    stats_path = "data/slat_stats.pt"
    slat_mean = slat_std = None
    if os.path.isfile(stats_path):
        st = torch.load(stats_path, map_location=device, weights_only=False)
        slat_mean = st["mean"].to(device)
        slat_std = st["std"].to(device)

    slat, ctx = _load_batch_lmdb(args.lmdb, args.batch_size, device)

    with torch.no_grad():
        pred = _pred_stats(model, slat, ctx, imf, device)
    print("  [Forward sensitivity @ t=0.5, r=t]")
    print(f"    ||u||/B ≈ {pred['u_norm']:.4f}   std(u) ≈ {pred['u_std']:.4f}")
    print(f"    cos(u correct ctx, u shuffle ctx) = {pred['cos_u_ctx_vs_shuffle']:.4f}")
    print("    (thấp = u phụ thuộc context; ~1 = context bị bỏ qua)\n")

    scenarios = [
        ("boundary_r_eq_t", dict(ratio_r_neq_t=0.0, cfg_enabled=False, force_jvp=False)),
        ("jvp_r_neq_t", dict(ratio_r_neq_t=1.0, cfg_enabled=False, force_jvp=True)),
        ("ctx_zero", dict(ratio_r_neq_t=0.5, cfg_enabled=False, force_jvp=False)),
    ]

    results = {}
    for name, kw in scenarios:
        model.zero_grad(set_to_none=True)
        if v_head is not None:
            v_head.zero_grad(set_to_none=True)
        c = ctx.clone()
        if name == "ctx_zero":
            c = torch.zeros_like(c)
        g, comps = _one_backward(
            model, v_head, imf, slat, c,
            slat_mean=slat_mean, slat_std=slat_std, **kw,
        )
        results[name] = (g, comps)
        print(f"  --- {name} --- loss={comps.get('loss', 0):.4f}  bnd={comps.get('loss_boundary', 0):.4f}  jvp={comps.get('loss_jvp', 0):.4f}")
        top = sorted(g.items(), key=lambda x: -x[1])[:12]
        for k, v in top:
            print(f"      {k:40s} grad_norm={v:.2e}")
        cross_layers = sorted(
            [(k, v) for k, v in g.items() if "cross_attn" in k],
            key=lambda x: -x[1],
        )
        if cross_layers:
            print("      --- cross_attn ---")
            for k, v in cross_layers[:8]:
                print(f"      {k:40s} grad_norm={v:.2e}")
        print()

    # Cross-attn proj vs mamba ratio
    def _sum_prefix(grads, prefix):
        return sum(v for k, v in grads.items() if prefix in k)

    g_bnd = results["boundary_r_eq_t"][0]
    cross = _sum_prefix(g_bnd, "cross_attn.proj")
    mamba = _sum_prefix(g_bnd, "mamba")
    out_p = _sum_prefix(g_bnd, "output_proj")
    vh = _sum_prefix(g_bnd, "v_head")
    print("  [Gradient balance — boundary batch]")
    print(f"    Σ cross_attn.proj  : {cross:.2e}")
    print(f"    Σ mamba            : {mamba:.2e}")
    print(f"    Σ output_proj      : {out_p:.2e}")
    print(f"    Σ v_head           : {vh:.2e}")
    if mamba > 0:
        print(f"    cross/mamba        : {cross / mamba:.4f}")
    if out_p > 0:
        print(f"    cross/output_proj  : {cross / out_p:.4f}")

    g_ctx = results["boundary_r_eq_t"][0]
    g_zero = results["ctx_zero"][0]
    cross_real = _sum_prefix(g_ctx, "cross_attn")
    cross_zero = _sum_prefix(g_zero, "cross_attn")
    print("\n  [Context path — boundary]")
    print(f"    Σ grad cross_attn (real ctx) : {cross_real:.2e}")
    print(f"    Σ grad cross_attn (zero ctx)  : {cross_zero:.2e}")

    # Dead params?
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if dead:
        print(f"\n  ⚠️  {len(dead)} params without grad in last backward (sample):")
        for n in dead[:8]:
            print(f"      {n}")
    else:
        print("\n  ✓ All model params received grad in boundary backward")

    # Weight norms snapshot
    def _wnorm(mod, pattern):
        s = 0.0
        for n, p in mod.named_parameters():
            if pattern in n:
                s += float(p.detach().float().norm().item() ** 2)
        return s ** 0.5

    print("\n  [Weight L2 norms]")
    print(f"    output_proj     : {_wnorm(model, 'output_proj'):.4f}")
    print(f"    cross_attn.proj : {_wnorm(model, 'cross_attn.proj'):.4f}")
    print(f"    arcface_tok     : {_wnorm(model, 'arcface_tokenizer'):.4f}")
    print(f"    v_head.proj     : {_wnorm(v_head, 'proj'):.4f}")

    if args.compare_ckpt and os.path.isfile(args.compare_ckpt):
        ck0 = torch.load(args.compare_ckpt, map_location="cpu", weights_only=False)
        s0 = ck0["model_state_dict"]
        delta_cross = 0.0
        n = 0
        for k, p in model.state_dict().items():
            if "cross_attn.proj" in k and k in s0:
                d = (p.cpu().float() - s0[k].float()).norm().item()
                delta_cross += d ** 2
                n += 1
        print(f"\n  [Δweight vs {args.compare_ckpt}]")
        print(f"    cross_attn.proj L2 delta ≈ {(delta_cross ** 0.5):.6f}  ({n} tensors)")

    print("\n" + "=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    cos = pred["cos_u_ctx_vs_shuffle"]
    if cross_real < 1e-8:
        print("  ⚠️  cross_attn gần như không nhận gradient → context path có thể chết.")
    elif cos > 0.95:
        print("  ⚠️  u gần như không đổi khi shuffle context → học identity yếu.")
    elif cross / max(mamba, 1e-8) < 0.01:
        print("  ⚠️  gradient context << mamba → backbone chủ yếu học dynamics, ít cond.")
    else:
        print("  ✓ Có gradient qua cross-attn và u phân biệt context (đang học).")
    print("=" * 72)


if __name__ == "__main__":
    main()
