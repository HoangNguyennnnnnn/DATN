"""
Test sampling từ iMF VoxelMamba checkpoint.
So sánh slat sampled vs slat ground truth qua MSE / cosine similarity.
"""
import argparse
import os
import sys
import io
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.config import TrainConfig
from src.models.voxel_mamba import VoxelMamba
from src.models.imf_diffusion import ImprovedMeanFlow


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/best.pt")
    ap.add_argument("--lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--omega", type=float, default=4.0, help="CFG guidance scale")
    ap.add_argument("--use-ema", action="store_true", help="Use EMA weights")
    args = ap.parse_args()

    device = "cuda"
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')} loss={ckpt.get('loss'):.4f}")

    mcfg = ckpt.get("stage2_model_config", {})
    print(f"  arch={mcfg.get('arch')} hidden={mcfg.get('hidden_dim')} layers={mcfg.get('num_layers')}")

    cfg = TrainConfig()
    imf_cfg = cfg.imf

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
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()

    diffusion = ImprovedMeanFlow()

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, max_readers=64)
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

        contexts = []
        slats_gt = []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            ctx = blob["context"]
            slt = blob["slat"]
            if isinstance(ctx, np.ndarray):
                ctx = torch.from_numpy(ctx)
            if isinstance(slt, np.ndarray):
                slt = torch.from_numpy(slt)
            contexts.append(ctx.float())
            slats_gt.append(slt.float())

    contexts = torch.stack(contexts).to(device)
    slats_gt = torch.stack(slats_gt).to(device)
    print(f"\nLoaded {args.num_samples} samples")
    print(f"  context shape: {contexts.shape}")
    print(f"  slat_gt (raw) shape: {slats_gt.shape}, mean={slats_gt.mean().item():.4f}, std={slats_gt.std().item():.4f}")

    # Load per-channel normalization stats (from checkpoint config or default path)
    slat_norm_mean = None
    slat_norm_std = None
    stats_path = mcfg.get("slat_stats_path") or "data/slat_stats.pt"
    if stats_path and os.path.exists(stats_path):
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1)
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1)
        print(f"  [Slat Norm] Using {stats_path}: mean range [{slat_norm_mean.min():.4f}, {slat_norm_mean.max():.4f}], "
              f"std range [{slat_norm_std.min():.4f}, {slat_norm_std.max():.4f}]")
    else:
        print(f"  [Slat Norm] No stats file at {stats_path} — comparing in raw space")

    print(f"\n[1/2] Sampling 1-step with omega={args.omega} ...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        slats_pred = diffusion.sample_1_step(
            model,
            context=contexts,
            shape=tuple(slats_gt.shape),
            omega=torch.tensor([args.omega] * args.num_samples, device=device),
        )

    slats_pred_f = slats_pred.float()
    print(f"  slat_pred (normalized) shape: {slats_pred_f.shape}, mean={slats_pred_f.mean().item():.4f}, std={slats_pred_f.std().item():.4f}")

    # Reverse normalization to compare in raw slat space
    if slat_norm_mean is not None and slat_norm_std is not None:
        slats_pred_f = slats_pred_f * slat_norm_std + slat_norm_mean
        print(f"  slat_pred (raw, post-unnormalize) mean={slats_pred_f.mean().item():.4f}, std={slats_pred_f.std().item():.4f}")

    # Per-sample metrics
    print("\n[2/2] Metrics (per sample):")
    print(f"  {'idx':>4} {'MSE':>10} {'L1':>10} {'cos_sim':>10} {'gt_std':>8} {'pred_std':>8}")

    mses, l1s, cos_sims = [], [], []
    for i in range(args.num_samples):
        gt = slats_gt[i]
        pr = slats_pred_f[i]
        mse = F.mse_loss(pr, gt).item()
        l1 = F.l1_loss(pr, gt).item()
        cos = F.cosine_similarity(pr.flatten(), gt.flatten(), dim=0).item()
        mses.append(mse)
        l1s.append(l1)
        cos_sims.append(cos)
        print(f"  {i:>4} {mse:>10.5f} {l1:>10.5f} {cos:>10.5f} {gt.std().item():>8.4f} {pr.std().item():>8.4f}")

    print(f"\n  AVG MSE={np.mean(mses):.5f}  AVG L1={np.mean(l1s):.5f}  AVG cos_sim={np.mean(cos_sims):.5f}")
    print(f"\nReference:")
    print(f"  Random guess: MSE ~ slat_var, cos_sim ~ 0")
    print(f"  Perfect: MSE = 0, cos_sim = 1.0")
    print(f"  GT slat variance: {slats_gt.var().item():.4f}")
    print(f"  Naive zero pred would give MSE = {(slats_gt**2).mean().item():.4f}")

    # Also do many-step sampling (5 steps) for comparison
    print(f"\n[Extra] Many-step sampling (5 steps Euler) ...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b = contexts.shape[0]
        z = torch.randn_like(slats_gt)
        num_steps = 5
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = ts[i].expand(b)
            t_nxt = ts[i + 1].expand(b)
            v = model(z, t_cur, contexts, r=t_nxt,
                      omega=torch.tensor([args.omega] * b, device=device),
                      cfg_tmin=torch.tensor([0.2] * b, device=device),
                      cfg_tmax=torch.tensor([0.8] * b, device=device))
            z = z - (t_cur - t_nxt).view(-1, 1, 1) * v
        slats_multi = z.float()

    # Reverse normalization for 5-step too
    if slat_norm_mean is not None and slat_norm_std is not None:
        slats_multi = slats_multi * slat_norm_std + slat_norm_mean

    cos_sims_multi = []
    for i in range(args.num_samples):
        cos = F.cosine_similarity(slats_multi[i].flatten(), slats_gt[i].flatten(), dim=0).item()
        cos_sims_multi.append(cos)
    print(f"  5-step AVG cos_sim={np.mean(cos_sims_multi):.5f} (vs 1-step={np.mean(cos_sims):.5f})")


if __name__ == "__main__":
    main()
