"""
Test xem model có học được velocity field ở phân phối t TRAINING không.
Nếu loss train = 0.21 ổn, model phải dự đoán đúng (e-x) ở boundary t=r.
"""
import argparse
import os
import sys
import io
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import TrainConfig
from src.models.voxel_mamba import VoxelMamba


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/imf_unet/best.pt")
    ap.add_argument("--lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt["stage2_model_config"]
    print(f"Checkpoint: epoch={ckpt['epoch']} loss={ckpt['loss']:.4f}")

    model = VoxelMamba(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        slat_length=mcfg["slat_length"],
        context_dim=mcfg["context_dim"],
        backend=mcfg.get("backend", "auto"),
        num_context_tokens=mcfg.get("num_context_tokens", 8),
        num_time_tokens=mcfg.get("num_time_tokens", 4),
        num_r_tokens=mcfg.get("num_r_tokens", 4),
        num_interval_tokens=mcfg.get("num_interval_tokens", 4),
        num_guidance_tokens=mcfg.get("num_guidance_tokens", 4),
        d_state=mcfg.get("d_state", 16),
        d_conv=mcfg.get("d_conv", 4),
        expand=mcfg.get("expand", 2),
    ).to(device)
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, max_readers=64)
    with env.begin() as txn:
        keys = []
        for k, _ in txn.cursor():
            if k == b"__meta__":
                continue
            keys.append(k)
            if len(keys) >= args.num_samples * 5:
                break
        rng = np.random.default_rng(42)
        keys = [keys[i] for i in rng.choice(len(keys), size=args.num_samples, replace=False)]
        contexts, slats = [], []
        for k in keys:
            blob = torch.load(io.BytesIO(txn.get(k)), map_location="cpu", weights_only=False)
            contexts.append(torch.as_tensor(blob["context"]).float())
            slats.append(torch.as_tensor(blob["slat"]).float())
    contexts = torch.stack(contexts).to(device)
    x_raw = torch.stack(slats).to(device)
    b = x_raw.shape[0]

    # Apply same per-channel normalization as training
    stats_path = mcfg.get("slat_stats_path") or "data/slat_stats.pt"
    if stats_path and os.path.exists(stats_path):
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_mean = _stats["mean"].to(device).view(1, 1, -1)
        slat_std = _stats["std"].to(device).view(1, 1, -1)
        x = (x_raw - slat_mean) / slat_std
        print(f"\n[Slat Norm] Applied: x_raw std={x_raw.std():.4f} → x_norm std={x.std():.4f}")
    else:
        x = x_raw
        print(f"\n[WARNING] No slat stats found, using raw slat")
    print(f"Slat (x_normalized) std={x.std().item():.4f}, mean={x.mean().item():.4f}")

    # Test 1: Boundary loss at training t distribution (t=r, logit_normal centered ~0.4)
    print("\n=== Test 1: Velocity prediction at boundary (r=t, t~training dist) ===")
    print(f"  {'t':>6} {'gt_v_std':>10} {'pred_v_std':>10} {'cos_sim':>10} {'mse':>10}")
    for t_val in [0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 0.99]:
        t = torch.full((b,), t_val, device=device)
        e = torch.randn_like(x)
        z_t = (1 - t).view(-1, 1, 1) * x + t.view(-1, 1, 1) * e  # paper: x=data, e=noise
        v_gt = e - x  # boundary velocity at any t for linear flow
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(z_t, t, contexts, r=t,
                           omega=torch.ones(b, device=device),
                           cfg_tmin=torch.full((b,), 0.2, device=device),
                           cfg_tmax=torch.full((b,), 0.8, device=device))
        v_pred = v_pred.float()
        mse = F.mse_loss(v_pred, v_gt).item()
        cos = F.cosine_similarity(v_pred.flatten(), v_gt.flatten(), dim=0).item()
        print(f"  {t_val:>6.2f} {v_gt.std().item():>10.4f} {v_pred.std().item():>10.4f} {cos:>10.4f} {mse:>10.4f}")

    # Test 2: Mean flow u(z, r=0, t=1) — this IS the 1-step sampling case
    print("\n=== Test 2: Mean velocity u(z_1, r=0, t=1) — used for 1-step sampling ===")
    e = torch.randn_like(x)  # pure noise = z_1
    t = torch.ones(b, device=device)
    r = torch.zeros(b, device=device)
    # Theoretical: u(z_1, r=0, t=1) = (z_1 - z_0) / (1-0) = e - x
    v_target = e - x
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u_pred = model(e, t, contexts, r=r,
                       omega=torch.ones(b, device=device),
                       cfg_tmin=torch.full((b,), 0.2, device=device),
                       cfg_tmax=torch.full((b,), 0.8, device=device))
    u_pred = u_pred.float()
    print(f"  target std={v_target.std().item():.4f}, pred std={u_pred.std().item():.4f}")
    print(f"  cos_sim={F.cosine_similarity(u_pred.flatten(), v_target.flatten(), dim=0).item():.4f}")
    print(f"  MSE={F.mse_loss(u_pred, v_target).item():.4f}")

    # Test 3: pure noise prediction — does model just predict the noise input back?
    print("\n=== Test 3: Does u_pred ≈ z_1? (model regurgitating input) ===")
    cos_to_input = F.cosine_similarity(u_pred.flatten(), e.flatten(), dim=0).item()
    print(f"  cos_sim(u_pred, z_1)={cos_to_input:.4f}")
    print(f"  If close to 1.0 → model just outputs input → z_0 = z_1 - z_1 = 0")
    print(f"  z_0 = z_1 - u_pred: std={(e - u_pred).std().item():.4f} (gt slat std={x.std().item():.4f})")


if __name__ == "__main__":
    main()
