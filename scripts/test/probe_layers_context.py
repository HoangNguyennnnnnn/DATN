#!/usr/bin/env python3
"""
Mở từng layer VoxelMamba: cos(h|ctx_A, h|ctx_B) cùng z_t — xem context chết ở đâu.
"""
from __future__ import annotations

import io
import os
import sys

import lmdb
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.voxel_mamba import VoxelMamba


def cos_pooled(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine giữa hai tensor [B,L,D] — pool mean theo L rồi cos theo D."""
    pa = a.float().mean(dim=1)
    pb = b.float().mean(dim=1)
    if pa.shape[0] == 1:
        return F.cosine_similarity(pa.flatten(), pb.flatten(), dim=0).item()
    return F.cosine_similarity(pa[0], pb[0], dim=0).item()


def cos_token_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@torch.no_grad()
def probe_block(block, x, time_cond, ctx_a, ctx_b, label: str, rows: list):
    """Chạy block 2 lần (ctx_a, ctx_b), ghi cos từng bước trong block."""
    B, L, D = x.shape

    def one(ctx_cond):
        scale_t, shift_t, gate_t = block.adaLN_time(time_cond).chunk(3, dim=-1)
        scale_t = scale_t.unsqueeze(1)
        shift_t = shift_t.unsqueeze(1)
        gate_t = gate_t.unsqueeze(1)
        residual = x
        x_norm = block.norm(x)
        x_mod = x_norm * (1 + scale_t) + shift_t
        if block.use_mamba:
            fwd = block.forward_mamba(x_mod)
            bwd = torch.flip(block.backward_mamba(torch.flip(x_mod, dims=[1])), dims=[1])
            m_out = fwd + bwd
        else:
            m_out, _ = block.gru(x_mod)
        m_out = block.dropout(m_out)
        scale_c, shift_c = block.adaLN_ctx(ctx_cond).chunk(2, dim=-1)
        scale_c = scale_c.unsqueeze(1)
        shift_c = shift_c.unsqueeze(1)
        c_out = m_out * (1 + scale_c) + shift_c
        h = residual + gate_t * c_out
        if block.use_ffn:
            scale_f, shift_f, gate_f = block.adaLN_ffn(time_cond).chunk(3, dim=-1)
            scale_f = scale_f.unsqueeze(1)
            shift_f = shift_f.unsqueeze(1)
            gate_f = gate_f.unsqueeze(1)
            x_ffn = block.norm_ffn(h) * (1 + scale_f) + shift_f
            h = h + gate_f * block.ffn(x_ffn)
        return dict(
            x_mod=x_mod,
            m_out=m_out,
            c_out=c_out,
            h=h,
            scale_c=scale_c,
            shift_c=shift_c,
            gate_t=gate_t,
        )

    oa = one(ctx_a)
    ob = one(ctx_b)
    rows.append((f"{label} in→out", cos_pooled(oa["h"], ob["h"])))
    rows.append((f"  └ after mamba", cos_pooled(oa["m_out"], ob["m_out"])))
    rows.append((f"  └ after ctx AdaLN", cos_pooled(oa["c_out"], ob["c_out"])))
    rows.append((f"  └ |Δscale_c|", (oa["scale_c"] - ob["scale_c"]).abs().mean().item()))
    rows.append((f"  └ |Δshift_c|", (oa["shift_c"] - ob["shift_c"]).abs().mean().item()))
    return oa["h"], ob["h"]


@torch.no_grad()
def forward_probe(model, x_t, t, r, ctx_a, ctx_b, omega, cfg_tmin, cfg_tmax, rows: list):
    B = x_t.shape[0]
    h = model.input_embed(x_t)
    rows.append(("input_embed", cos_pooled(h, h)))  # same x → 1.0 baseline

    if model.use_hilbert_ordering:
        h = h[:, model._hilbert_to_raster, :]

    ha, hb = h.clone(), h.clone()
    if model.total_prefix_tokens > 0:
        t_emb = model.time_mlp(t)
        r_emb = model.r_mlp(r)
        interval_emb = model.interval_mlp(t - r)
        guidance_input = torch.stack([omega, cfg_tmin, cfg_tmax], dim=-1)

        def prefix_for(ctx):
            ctx_tok = model._make_prefix_tokens(model.context_tokenizer, ctx, model.num_context_tokens, B)
            time_tok = model._make_prefix_tokens(model.time_tokenizer, t_emb, model.num_time_tokens, B)
            r_tok = model._make_prefix_tokens(model.r_tokenizer, r_emb, model.num_r_tokens, B)
            int_tok = model._make_prefix_tokens(model.interval_tokenizer, interval_emb, model.num_interval_tokens, B)
            g_tok = model._make_prefix_tokens(model.guidance_tokenizer, guidance_input, model.num_guidance_tokens, B)
            return torch.cat([ctx_tok, time_tok, r_tok, int_tok, g_tok], dim=1)

        pa = torch.cat([ha, prefix_for(ctx_a)], dim=1)
        pb = torch.cat([hb, prefix_for(ctx_b)], dim=1)
        slat = model.slat_length
        rows.append(("slat tokens only (pre-cat)", cos_pooled(ha, hb)))
        rows.append(("+prefix ALL tokens", cos_pooled(pa[:, :slat], pb[:, :slat])))
        rows.append(("prefix region only", cos_pooled(pa[:, slat:], pb[:, slat:])))
        ha, hb = pa, pb

    ctx_cond_a, time_cond = model._build_cond_emb(t, r, ctx_a, omega, cfg_tmin, cfg_tmax)
    ctx_cond_b, _ = model._build_cond_emb(t, r, ctx_b, omega, cfg_tmin, cfg_tmax)
    rows.append(("ctx_cond_mlp vec", F.cosine_similarity(ctx_cond_a, ctx_cond_b, dim=-1).mean().item()))
    if model.ctx_layer_projs is not None:
        rows.append(("ctx_layer_proj[0]", F.cosine_similarity(
            model.ctx_layer_projs[0](ctx_cond_a), model.ctx_layer_projs[0](ctx_cond_b), dim=-1
        ).mean().item()))

    slat_len = model.slat_length
    for i, layer in enumerate(model.layers):
        ctx_la = model.ctx_layer_projs[i](ctx_cond_a) if model.ctx_layer_projs else ctx_cond_a
        ctx_lb = model.ctx_layer_projs[i](ctx_cond_b) if model.ctx_layer_projs else ctx_cond_b
        ha_s, hb_s = ha[:, :slat_len], hb[:, :slat_len]
        ha_s, hb_s = probe_block(layer, ha_s, time_cond, ctx_la, ctx_lb, f"block[{i:02d}]", rows)
        if ha.shape[1] > slat_len:
            ha = torch.cat([ha_s, ha[:, slat_len:]], dim=1)
            hb = torch.cat([hb_s, hb[:, slat_len:]], dim=1)
        else:
            ha, hb = ha_s, hb_s
        rows.append((f"block[{i:02d}] slat hidden", cos_pooled(ha_s, hb_s)))

    if model.total_prefix_tokens > 0:
        ha, hb = ha[:, :slat_len], hb[:, :slat_len]
    if model.use_hilbert_ordering:
        ha = ha[:, model._raster_to_hilbert, :]
        hb = hb[:, model._raster_to_hilbert, :]
    ha = model.output_norm(ha)
    hb = model.output_norm(hb)
    rows.append(("output_norm hidden", cos_pooled(ha, hb)))
    ua = model.output_proj(ha)
    ub = model.output_proj(hb)
    rows.append(("output_proj u (velocity)", cos_pooled(ua, ub)))
    return ua, ub


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_v7_phaseB/epoch_40.pt")
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--backend", default="gru", choices=["gru", "auto", "mamba"])
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    from src.models.voxel_mamba import voxel_mamba_from_stage2_config

    device = torch.device("cpu")
    model = voxel_mamba_from_stage2_config(mcfg, backend=args.backend, dropout=0.0).to(device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
        b0 = torch.load(io.BytesIO(txn.get(keys[0])), map_location="cpu", weights_only=False)
        b1 = torch.load(io.BytesIO(txn.get(keys[50])), map_location="cpu", weights_only=False)
    env.close()

    stats = torch.load("data/slat_stats.pt", weights_only=False)
    mean = stats["mean"].view(1, 1, -1)
    std = stats["std"].view(1, 1, -1)
    x = (b0["slat"].float().unsqueeze(0) - mean) / std
    ctx_a = b0["context"].float().unsqueeze(0)
    ctx_b = b1["context"].float().unsqueeze(0)
    ctx_z = torch.zeros_like(ctx_a)
    noise = torch.randn_like(ctx_a)

    t = torch.tensor([0.5])
    r = torch.zeros(1)
    o = torch.ones(1)
    z0 = torch.zeros(1)
    z = 0.5 * x + 0.5 * torch.randn_like(x)

    print("=" * 72)
    print(f"  LAYER PROBE  ckpt={args.ckpt}  backend={args.backend}")
    print(f"  ctx_A vs ctx_B (người khác) | cùng z_t, t=0.5")
    print("=" * 72)

    for name, ctx in [("A vs B (identity)", ctx_b), ("A vs ZERO", ctx_z), ("A vs NOISE", noise)]:
        rows = []
        ua, ub = forward_probe(model, z, t, r, ctx_a, ctx, o, z0, o, rows)
        print(f"\n--- {name} ---")
        print(f"  {'stage':<32} {'cos':>8}  (1=giống hệt, 0=khác hẳn)")
        for stage, c in rows:
            if stage.startswith("  └"):
                print(f"  {stage:<32} {c:>8.4f}")
            else:
                print(f"  {stage:<32} {c:>8.4f}")

    print("\n" + "=" * 72)
    print("  Giải thích nhanh:")
    print("  - ctx_cond_mlp thấp + block[00] lên ~1.0 → Mamba+gate triệt tiêu ngay layer 1")
    print("  - |Δscale_c| nhỏ → AdaLN context gần không đổi giữa A/B")
    print("  - prefix region cos thấp nhưng slat region cos→1 → prefix không lan vào 4096 token")
    print("=" * 72)


if __name__ == "__main__":
    main()
