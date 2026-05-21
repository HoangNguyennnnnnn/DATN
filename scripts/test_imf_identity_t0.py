"""
Test 3: Identity at t=0 — Model should predict v ≈ 0
=====================================================
At t=0, z_t = x (no noise). The flow velocity target v = ε - x.
Since z_0 = x, the model sees clean data and should predict
a coherent velocity field (not random noise).

We also check t=1 (pure noise) where model output should have
larger magnitude since it needs to denoise fully.

This tests whether the time conditioning is working correctly.
"""
import os
import sys
import io
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import TrainConfig
from src.models.voxel_mamba import VoxelMamba
from src.models.imf_diffusion import ImprovedMeanFlow


def main():
    import argparse
    ap = argparse.ArgumentParser("iMF Identity Test at t=0")
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/latest_step.pt")
    ap.add_argument("--lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    print("=" * 60)
    print("  TEST 3: IDENTITY AT t=0 & BOUNDARY CONDITIONS")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("stage2_model_config", {})
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 0):.4f}")

    model = VoxelMamba(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        slat_length=mcfg["slat_length"],
        context_dim=mcfg["context_dim"],
        backend=mcfg.get("backend", "auto"),
        strict=False,
        num_context_tokens=mcfg.get("num_context_tokens", 8),
        num_time_tokens=mcfg.get("num_time_tokens", 4),
        num_r_tokens=mcfg.get("num_r_tokens", 4),
        num_interval_tokens=mcfg.get("num_interval_tokens", 4),
        num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
        d_state=mcfg.get("d_state", 16),
        d_conv=mcfg.get("d_conv", 4),
        expand=mcfg.get("expand", 2),
        dropout=0.0,
    ).to(device)

    state = ckpt["ema_state_dict"] if (args.use_ema and "ema_state_dict" in ckpt) else ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Model loaded. Backend={getattr(model, 'backend', '?')}")

    # Load normalization stats
    slat_norm_mean = slat_norm_std = None
    stats_path = "data/slat_stats.pt"
    if os.path.exists(stats_path):
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1)
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1)
        print(f"  [Slat Norm] Loaded {stats_path}")

    # Load samples from LMDB
    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = []
        cur = txn.cursor()
        for k, _ in cur:
            if k == b"__meta__":
                continue
            keys.append(k)
            if len(keys) >= args.num_samples * 5:
                break
        rng = np.random.default_rng(42)
        picks = rng.choice(len(keys), size=args.num_samples, replace=False)
        keys = [keys[i] for i in picks]

        contexts, slats = [], []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            ctx = blob["context"]
            slt = blob["slat"]
            if isinstance(ctx, np.ndarray):
                ctx = torch.from_numpy(ctx)
            if isinstance(slt, np.ndarray):
                slt = torch.from_numpy(slt)
            contexts.append(ctx.float())
            slats.append(slt.float())
    env.close()

    contexts = torch.stack(contexts).to(device)
    slats_raw = torch.stack(slats).to(device)
    B = slats_raw.shape[0]
    print(f"  Loaded {B} samples. slat shape={slats_raw.shape}")

    # Normalize slats (same as training)
    if slat_norm_mean is not None:
        slats_norm = (slats_raw - slat_norm_mean) / slat_norm_std
    else:
        slats_norm = slats_raw

    print(f"\n{'='*60}")
    print("  PART A: Model output at different t values")
    print(f"{'='*60}")
    print(f"  {'t':>5} {'||v_pred||':>12} {'||x||':>10} {'ratio':>8} {'v_mean':>8} {'v_std':>8}")
    print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    t_values = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    results = {}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        noise = torch.randn_like(slats_norm)  # Fixed noise for all t

        for t_val in t_values:
            t = torch.full((B,), t_val, device=device)
            r = torch.zeros(B, device=device)  # boundary case r=0

            # Construct z_t = (1-t)*x + t*noise  (flow matching interpolation)
            z_t = (1.0 - t_val) * slats_norm + t_val * noise

            # Model prediction
            v_pred = model(z_t, t, contexts, r=r,
                           omega=torch.ones(B, device=device),
                           cfg_tmin=torch.zeros(B, device=device),
                           cfg_tmax=torch.ones(B, device=device))

            v_pred_f = v_pred.float()
            v_norm = v_pred_f.norm(dim=-1).mean().item()
            x_norm = slats_norm.float().norm(dim=-1).mean().item()
            ratio = v_norm / max(x_norm, 1e-8)
            v_mean = v_pred_f.mean().item()
            v_std = v_pred_f.std().item()

            results[t_val] = {
                "v_norm": v_norm,
                "x_norm": x_norm,
                "ratio": ratio,
                "v_mean": v_mean,
                "v_std": v_std,
            }
            print(f"  {t_val:>5.2f} {v_norm:>12.4f} {x_norm:>10.4f} "
                  f"{ratio:>8.4f} {v_mean:>8.4f} {v_std:>8.4f}")

    print(f"\n{'='*60}")
    print("  PART B: Does model output CHANGE with t?")
    print(f"{'='*60}")

    # Key question: is v_pred at t=0 different from v_pred at t=1?
    v_range = results[1.0]["v_norm"] - results[0.0]["v_norm"]
    v_ratio_range = results[1.0]["ratio"] - results[0.0]["ratio"]
    std_range = results[1.0]["v_std"] - results[0.0]["v_std"]

    print(f"  ||v(t=1)|| - ||v(t=0)|| = {v_range:.4f}")
    print(f"  ratio(t=1) - ratio(t=0) = {v_ratio_range:.4f}")
    print(f"  std(t=1) - std(t=0) = {std_range:.4f}")

    # Check if outputs are actually different for different t
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = torch.zeros(B, device=device)
        t1 = torch.ones(B, device=device)
        r0 = torch.zeros(B, device=device)

        v_at_0 = model(slats_norm, t0, contexts, r=r0,
                        omega=torch.ones(B, device=device),
                        cfg_tmin=torch.zeros(B, device=device),
                        cfg_tmax=torch.ones(B, device=device)).float()
        v_at_1 = model(noise, t1, contexts, r=r0,
                        omega=torch.ones(B, device=device),
                        cfg_tmin=torch.zeros(B, device=device),
                        cfg_tmax=torch.ones(B, device=device)).float()

    cos_between = F.cosine_similarity(v_at_0.flatten(), v_at_1.flatten(), dim=0).item()
    l2_diff = (v_at_0 - v_at_1).norm().item()
    print(f"\n  cos_sim(v(t=0), v(t=1)) = {cos_between:.5f}")
    print(f"  ||v(t=0) - v(t=1)||₂ = {l2_diff:.4f}")

    print(f"\n{'='*60}")
    print("  PART C: Does model output CHANGE with context?")
    print(f"{'='*60}")

    # Use same z_t but shuffle contexts → should get different v_pred
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_mid = torch.full((B,), 0.5, device=device)
        z_mid = 0.5 * slats_norm + 0.5 * noise

        v_real_ctx = model(z_mid, t_mid, contexts, r=torch.zeros(B, device=device),
                           omega=torch.ones(B, device=device),
                           cfg_tmin=torch.zeros(B, device=device),
                           cfg_tmax=torch.ones(B, device=device)).float()

        # Shuffle contexts (rotate by 1)
        ctx_shuffled = torch.roll(contexts, shifts=1, dims=0)
        v_wrong_ctx = model(z_mid, t_mid, ctx_shuffled, r=torch.zeros(B, device=device),
                            omega=torch.ones(B, device=device),
                            cfg_tmin=torch.zeros(B, device=device),
                            cfg_tmax=torch.ones(B, device=device)).float()

    # Per-sample similarity between real-ctx and wrong-ctx predictions
    cos_sims = []
    for i in range(B):
        cs = F.cosine_similarity(v_real_ctx[i].flatten(),
                                  v_wrong_ctx[i].flatten(), dim=0).item()
        cos_sims.append(cs)

    avg_ctx_cos = np.mean(cos_sims)
    print(f"  AVG cos_sim(v_real_ctx, v_wrong_ctx) = {avg_ctx_cos:.5f}")
    print(f"  Per-sample: {[f'{c:.4f}' for c in cos_sims]}")

    if avg_ctx_cos > 0.99:
        print(f"\n  ⚠️  WARNING: Model outputs nearly IDENTICAL regardless of context!")
        print(f"  → Context conditioning is NOT working (model ignores context)")
    elif avg_ctx_cos > 0.9:
        print(f"\n  ⚠️  Model is weakly sensitive to context (cos > 0.9)")
    else:
        print(f"\n  ✅ Model produces different outputs for different contexts")

    # Final verdict
    print(f"\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")

    issues = []
    if abs(v_ratio_range) < 0.05:
        issues.append("Model output magnitude does NOT change with t → time conditioning broken")
    if abs(cos_between) > 0.95:
        issues.append("v(t=0) ≈ v(t=1) → model ignores time entirely")
    if avg_ctx_cos > 0.95:
        issues.append("Model ignores context → context conditioning broken")
    if results[0.0]["ratio"] > 2.0:
        issues.append(f"||v(t=0)|| / ||x|| = {results[0.0]['ratio']:.2f} >> expected ~1.0 → model outputs excessive velocity at clean input")

    if not issues:
        print("  ✅ PASS: Time and context conditioning appear functional.")
        print("  Model responds to both t and context changes.")
    else:
        print("  ❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")


if __name__ == "__main__":
    main()
