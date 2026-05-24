"""
Test: time conditioning + context conditioning (iMF / VoxelMamba).

PART A: ||u_pred|| theo t (không kỳ vọng u→0 tại t=0; target flow là ε−x).
PART B: u(t=0) vs u(t=1) — cos thấp = time OK.
PART C: cùng z_t, đổi context — cos thấp = context OK.

Dùng đúng LMDB + context_segment_weights khớp train (balanced: slat_context_balanced.lmdb).
"""
import os
import sys
import io
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.voxel_mamba import VoxelMamba, voxel_mamba_from_stage2_config


def _load_voxel_mamba_from_ckpt(
    ckpt: dict, device: torch.device, *, use_ema: bool = True
) -> VoxelMamba:
    mcfg = ckpt.get("stage2_model_config", {})
    seg_w = mcfg.get("context_segment_weights")
    if seg_w is not None and len(seg_w) == 3:
        seg_w = tuple(float(x) for x in seg_w)
    model = voxel_mamba_from_stage2_config(
        mcfg, backend=mcfg.get("backend", "auto"), dropout=0.0,
    ).to(device)
    state_key = "ema_state_dict" if (use_ema and "ema_state_dict" in ckpt) else "model_state_dict"
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt[state_key].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if seg_w is not None:
        print(f"  [context] segment weights = {seg_w}")
    elif getattr(model, "_context_segment_weights", None) is not None:
        print(f"  [context] segment weights from ckpt buffer = {model._context_segment_weights.tolist()}")
    else:
        print("  [warn] context_segment_weights=None — khác train balanced (3,2,0.5)?")
    model.eval()
    return model, mcfg


def main():
    import argparse
    ap = argparse.ArgumentParser("iMF identity / conditioning test")
    ap.add_argument(
        "--ckpt",
        "--checkpoint",
        default="checkpoints/imf_v8_lite/latest_step.pt",
        help="Path to iMF checkpoint (.pt)",
    )
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--no-ema", action="store_true", help="Dùng model_state_dict thay EMA")
    ap.add_argument(
        "--context-segment-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("ARC", "FLAME", "DINO"),
        help="Ghi đè weights nếu ckpt cũ chưa lưu (mặc định balanced: 3 2 0.5)",
    )
    args = ap.parse_args()
    use_ema = not args.no_ema

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  iMF CONDITIONING TEST (time + context)")
    print("=" * 60)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 0):.4f}")

    mcfg = ckpt.get("stage2_model_config", {})
    if args.context_segment_weights is not None:
        mcfg = dict(mcfg)
        mcfg["context_segment_weights"] = tuple(args.context_segment_weights)
    elif mcfg.get("context_segment_weights") is None and not mcfg.get(
        "context_use_arcface_only", True
    ):
        mcfg = dict(mcfg)
        mcfg["context_segment_weights"] = (3.0, 2.0, 0.5)
        print("  [context] fallback segment weights (3, 2, 0.5) — ckpt chưa ghi config")
    elif mcfg.get("context_use_arcface_only", True):
        print("  [context] arcface_only=True (v8 lite) — không dùng segment weights")

    ckpt_for_load = dict(ckpt)
    ckpt_for_load["stage2_model_config"] = mcfg
    model, _ = _load_voxel_mamba_from_ckpt(ckpt_for_load, device, use_ema=use_ema)
    print(f"  Backend={getattr(model, 'backend', '?')}")

    slat_norm_mean = slat_norm_std = None
    stats_path = mcfg.get("slat_stats_path") or "data/slat_stats.pt"
    if stats_path and os.path.exists(stats_path):
        _stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1)
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1)
        print(f"  [Slat Norm] {stats_path}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
        rng = np.random.default_rng(42)
        picks = rng.choice(len(keys), size=min(args.num_samples, len(keys)), replace=False)
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
    print(f"  LMDB: {args.lmdb}")
    print(f"  Loaded {B} samples. slat shape={slats_raw.shape}")

    if slat_norm_mean is not None:
        slats_norm = (slats_raw - slat_norm_mean) / slat_norm_std
    else:
        slats_norm = slats_raw

    omega = torch.ones(B, device=device)
    zcfg = torch.zeros(B, device=device)
    ocfg = torch.ones(B, device=device)

    print(f"\n{'='*60}")
    print("  PART A: ||u_pred|| vs t (magnitude)")
    print(f"{'='*60}")
    print(f"  {'t':>5} {'||u||':>12} {'||x||':>10} {'ratio':>8}")
    print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*8}")

    t_values = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    results = {}
    noise = torch.randn_like(slats_norm)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for t_val in t_values:
            t = torch.full((B,), t_val, device=device)
            r = torch.zeros(B, device=device)
            z_t = (1.0 - t_val) * slats_norm + t_val * noise
            u_pred = model(z_t, t, contexts, r=r, omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
            u_norm = u_pred.norm(dim=-1).mean().item()
            x_norm = slats_norm.float().norm(dim=-1).mean().item()
            results[t_val] = {"u_norm": u_norm, "ratio": u_norm / max(x_norm, 1e-8)}
            print(f"  {t_val:>5.2f} {u_norm:>12.4f} {x_norm:>10.4f} {results[t_val]['ratio']:>8.4f}")

    print(f"\n{'='*60}")
    print("  PART B: Time conditioning (direction)")
    print(f"{'='*60}")
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        t0 = torch.zeros(B, device=device)
        t1 = torch.ones(B, device=device)
        r0 = torch.zeros(B, device=device)
        u_at_0 = model(slats_norm, t0, contexts, r=r0, omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
        u_at_1 = model(noise, t1, contexts, r=r0, omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()

    cos_t = F.cosine_similarity(u_at_0.flatten(), u_at_1.flatten(), dim=0).item()
    l2_t = (u_at_0 - u_at_1).norm().item()
    mag_delta = results[1.0]["u_norm"] - results[0.0]["u_norm"]
    print(f"  cos_sim(u@t=0, u@t=1) = {cos_t:.5f}  (thấp ≈ tốt, ví dụ <0.3)")
    print(f"  ||u@t0 - u@t1||₂ = {l2_t:.2f}")
    print(f"  ||u|| delta t1-t0 = {mag_delta:.4f}  (có thể nhỏ; dùng cos là chính)")

    print(f"\n{'='*60}")
    print("  PART C: Context conditioning (shuffle ctx)")
    print(f"{'='*60}")
    perm = torch.randperm(B, device=device)
    ctx_perm = contexts[perm]
    # Tránh trùng hoàn toàn
    if (perm == torch.arange(B, device=device)).all() and B > 1:
        ctx_perm = torch.roll(contexts, shifts=1, dims=0)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        t_mid = torch.full((B,), 0.5, device=device)
        z_mid = 0.5 * slats_norm + 0.5 * noise
        u_real = model(z_mid, t_mid, contexts, r=torch.zeros(B, device=device),
                       omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
        u_wrong = model(z_mid, t_mid, ctx_perm, r=torch.zeros(B, device=device),
                        omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()

    cos_sims = [
        F.cosine_similarity(u_real[i].flatten(), u_wrong[i].flatten(), dim=0).item()
        for i in range(B)
    ]
    avg_ctx_cos = float(np.mean(cos_sims))
    print(f"  AVG cos_sim(u correct_ctx, u wrong_ctx) = {avg_ctx_cos:.5f}")
    print(f"  Per-sample: {[f'{c:.4f}' for c in cos_sims]}")

    # Chỉ shuffle khối ArcFace (512) — nhạy identity hơn roll full ctx
    ctx_arc_perm = contexts.clone()
    ctx_arc_perm[:, :512] = contexts[perm, :512]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        u_wrong_arc = model(z_mid, t_mid, ctx_arc_perm, r=torch.zeros(B, device=device),
                            omega=omega, cfg_tmin=zcfg, cfg_tmax=ocfg).float()
    cos_arc = [
        F.cosine_similarity(u_real[i].flatten(), u_wrong_arc[i].flatten(), dim=0).item()
        for i in range(B)
    ]
    print(f"  AVG cos (chỉ đổi ArcFace 512-d) = {float(np.mean(cos_arc)):.5f}")

    print(f"\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")
    issues = []
    if cos_t > 0.85:
        issues.append(f"u@t=0 ≈ u@t=1 (cos={cos_t:.3f}) → time conditioning yếu/hỏng")
    if avg_ctx_cos > 0.95:
        issues.append(f"Context shuffle gần như không đổi output (cos={avg_ctx_cos:.4f})")
    if float(np.mean(cos_arc)) > 0.95:
        issues.append(f"Đổi riêng ArcFace vẫn cos≈1 (cos={np.mean(cos_arc):.4f}) → identity chưa vào model")

    if not issues:
        print("  PASS: Time và context đều ảnh hưởng output.")
    else:
        print("  ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        if ckpt.get("epoch", 0) < 10:
            print("  Gợi ý: epoch còn sớm — chạy lại sau ~10–20 epoch trước khi kết luận train hỏng.")


if __name__ == "__main__":
    main()
