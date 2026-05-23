"""
Test 2: iMF Memorization Sanity Check
=====================================
Train iMF model on a SINGLE (slat, context) pair for N epochs.
If model can't memorize 1 sample → architecture fundamentally broken.

Pass criteria: 1-step cos_sim > 0.9 by epoch 300.
"""
import os
import sys
import io
import time
import copy
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TrainConfig
from src.models.voxel_mamba import VoxelMamba
from src.models.imf_diffusion import ImprovedMeanFlow


def load_single_sample(lmdb_path: str, seed: int = 42) -> tuple:
    """Load 1 random (slat, context) from LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = []
        cur = txn.cursor()
        for k, _ in cur:
            if k == b"__meta__":
                continue
            keys.append(k)
            if len(keys) >= 100:
                break
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(keys))
        blob = torch.load(io.BytesIO(txn.get(keys[pick])),
                          map_location="cpu", weights_only=False)
    env.close()

    ctx = blob["context"]
    slt = blob["slat"]
    if isinstance(ctx, np.ndarray):
        ctx = torch.from_numpy(ctx)
    if isinstance(slt, np.ndarray):
        slt = torch.from_numpy(slt)
    return slt.float(), ctx.float(), keys[pick].decode()


@torch.no_grad()
def eval_1step(model: VoxelMamba, imf: ImprovedMeanFlow,
               slat_gt: torch.Tensor, context: torch.Tensor,
               slat_norm_mean: torch.Tensor | None,
               slat_norm_std: torch.Tensor | None) -> dict:
    """Evaluate 1-step generation quality."""
    model.eval()
    device = next(model.parameters()).device
    B = 1
    slat_gt_dev = slat_gt.unsqueeze(0).to(device)
    ctx_dev = context.unsqueeze(0).to(device)

    # Normalize GT for comparison in normalized space
    slat_gt_norm = slat_gt_dev
    if slat_norm_mean is not None:
        slat_gt_norm = (slat_gt_dev - slat_norm_mean) / slat_norm_std

    with torch.autocast("cuda", dtype=torch.bfloat16):
        slat_pred = imf.sample_1_step(
            model,
            context=ctx_dev,
            shape=slat_gt_dev.shape,
            omega=torch.ones(B, device=device),
        )

    # Compare in normalized space (what model outputs)
    pred_f = slat_pred.float().squeeze(0)
    gt_f = slat_gt_norm.float().squeeze(0)
    cos_sim = F.cosine_similarity(pred_f.flatten(), gt_f.flatten(), dim=0).item()
    mse = F.mse_loss(pred_f, gt_f).item()

    # Also compare after unnormalize (raw space)
    if slat_norm_mean is not None:
        pred_raw = pred_f * slat_norm_std.squeeze(0) + slat_norm_mean.squeeze(0)
    else:
        pred_raw = pred_f
    gt_raw = slat_gt_dev.float().squeeze(0)
    cos_raw = F.cosine_similarity(pred_raw.flatten(), gt_raw.flatten(), dim=0).item()
    mse_raw = F.mse_loss(pred_raw, gt_raw).item()

    model.train()
    return {
        "cos_sim_norm": cos_sim,
        "mse_norm": mse,
        "cos_sim_raw": cos_raw,
        "mse_raw": mse_raw,
        "pred_std": pred_f.std().item(),
        "gt_std": gt_f.std().item(),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser("iMF Memorization Test")
    ap.add_argument("--lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/latest_step.pt",
                    help="Starting checkpoint (or 'scratch' for random init)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda")
    print("=" * 60)
    print("  TEST 2: iMF MEMORIZATION (Single Sample)")
    print("=" * 60)

    # Load single sample
    slat_gt, context, sample_key = load_single_sample(args.lmdb)
    print(f"  Sample: {sample_key}")
    print(f"  slat shape: {slat_gt.shape}, context shape: {context.shape}")
    print(f"  slat stats: mean={slat_gt.mean():.4f}, std={slat_gt.std():.4f}")

    # Build model from checkpoint config
    if args.ckpt != "scratch" and os.path.exists(args.ckpt):
        print(f"\n  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mcfg = ckpt.get("stage2_model_config", {})
        print(f"  Original epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 0):.4f}")
    else:
        print("\n  Training from SCRATCH (random init)")
        ckpt = None
        cfg = TrainConfig()
        mcfg = {
            "input_dim": cfg.imf.input_dim,
            "hidden_dim": cfg.imf.mamba_hidden_dim,
            "num_layers": cfg.imf.mamba_num_layers,
            "slat_length": cfg.imf.slat_length,
            "context_dim": cfg.imf.context_dim,
            "backend": "auto",
            "num_context_tokens": 8,
            "num_time_tokens": 4,
            "num_r_tokens": 4,
            "num_interval_tokens": 4,
            "num_guidance_tokens": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        }

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

    if ckpt is not None:
        state = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print("  Loaded model weights.")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Load normalization stats
    slat_norm_mean = None
    slat_norm_std = None
    stats_path = "data/slat_stats.pt"
    if os.path.exists(stats_path):
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1)
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1)
        print(f"  [Slat Norm] Loaded from {stats_path}")

    # Prepare training data (single sample, batch=1)
    x_raw = slat_gt.unsqueeze(0).to(device)  # [1, 4096, 32]
    ctx = context.unsqueeze(0).to(device)     # [1, ctx_dim]

    # Normalize slat for training (same as train_imf.py)
    if slat_norm_mean is not None:
        x = (x_raw - slat_norm_mean) / slat_norm_std
    else:
        x = x_raw

    # iMF framework
    imf = ImprovedMeanFlow(
        sigma_min=1e-4,
        ratio_r_neq_t=0.5,
        t_sampler="uniform",
        adaptive_loss_weighting=False,  # No adaptive for single sample
    )

    # Optimizer - simple AdamW, no scheduler needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # Pre-training eval
    print(f"\n{'='*60}")
    print(f"  Pre-training evaluation:")
    metrics = eval_1step(model, imf, slat_gt, context, slat_norm_mean, slat_norm_std)
    print(f"  cos_sim(norm)={metrics['cos_sim_norm']:.5f}  "
          f"cos_sim(raw)={metrics['cos_sim_raw']:.5f}  "
          f"MSE(norm)={metrics['mse_norm']:.5f}  "
          f"pred_std={metrics['pred_std']:.4f}  gt_std={metrics['gt_std']:.4f}")

    # Training loop
    print(f"\n  Training for {args.epochs} epochs on 1 sample...")
    print(f"  LR={args.lr}, eval every {args.eval_every} epochs")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6} {'Loss':>10} {'cos_norm':>10} {'cos_raw':>10} "
          f"{'MSE_norm':>10} {'pred_std':>9} {'time':>6}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*6}")

    best_cos = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time()

        # Train multiple iterations per epoch (iMF samples random t each time)
        epoch_loss = 0.0
        n_iters = 50  # 50 random t samples per epoch
        for _ in range(n_iters):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_out = imf.compute_loss(
                    model, x, ctx,
                    v_head=None,
                    return_components=True,
                )
                loss = loss_out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_iters
        elapsed = time.time() - t_start

        if epoch % args.eval_every == 0 or epoch == 1:
            metrics = eval_1step(model, imf, slat_gt, context,
                                 slat_norm_mean, slat_norm_std)
            cos_n = metrics["cos_sim_norm"]
            cos_r = metrics["cos_sim_raw"]
            mse_n = metrics["mse_norm"]
            p_std = metrics["pred_std"]
            best_cos = max(best_cos, cos_r)
            print(f"  {epoch:>6} {avg_loss:>10.5f} {cos_n:>10.5f} {cos_r:>10.5f} "
                  f"{mse_n:>10.5f} {p_std:>9.4f} {elapsed:>5.1f}s")

            # Early success
            if cos_r > 0.95:
                print(f"\n  ✅ PASS: cos_sim(raw) = {cos_r:.4f} > 0.95 at epoch {epoch}")
                print(f"  Model can memorize a single sample → architecture has capacity.")
                return

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"  Final evaluation after {args.epochs} epochs:")
    metrics = eval_1step(model, imf, slat_gt, context, slat_norm_mean, slat_norm_std)
    print(f"  cos_sim(norm)={metrics['cos_sim_norm']:.5f}  "
          f"cos_sim(raw)={metrics['cos_sim_raw']:.5f}  "
          f"MSE(norm)={metrics['mse_norm']:.5f}")
    print(f"  Best cos_sim(raw) achieved: {best_cos:.5f}")

    if best_cos > 0.9:
        print(f"\n  ✅ PASS: best cos_sim(raw) = {best_cos:.4f} > 0.9")
        print(f"  Architecture has capacity. Problem is sample efficiency.")
    elif best_cos > 0.5:
        print(f"\n  ⚠️ PARTIAL: best cos_sim(raw) = {best_cos:.4f}")
        print(f"  Model learning slowly. May need more epochs or architecture tuning.")
    else:
        print(f"\n  ❌ FAIL: best cos_sim(raw) = {best_cos:.4f} < 0.5")
        print(f"  Architecture cannot memorize even 1 sample → fundamentally broken.")


if __name__ == "__main__":
    main()
