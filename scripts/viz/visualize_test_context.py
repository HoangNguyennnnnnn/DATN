"""
Visualize hybrid context (946-dim) cho test samples:
  - ArcFace: 512 dims (identity)
  - FLAME: 50 dims (expression)
  - DINOv2: 384 dims (back-of-head shape)

Đối chiếu context với:
  - GT mesh file (.obj) để xem context tương ứng với mặt nào
  - Render image (nếu có)

Outputs: outputs_context_viz/<key>_context.png + <key>_stats.txt
"""
import argparse
import io
import os
import sys

import lmdb
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-lmdb", default="data/hybrid_context.lmdb")
    ap.add_argument("--slat-lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--dataset-filter", default="faceverse")
    ap.add_argument("--out-dir", default="outputs_context_viz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Tìm test keys (context only, no slat)
    ctx_env = lmdb.open(args.context_lmdb, readonly=True, lock=False)
    slat_env = lmdb.open(args.slat_lmdb, readonly=True, lock=False)
    test_keys = []
    with ctx_env.begin() as ctx_txn, slat_env.begin() as slat_txn:
        for k, _ in ctx_txn.cursor():
            if k == b"__meta__":
                continue
            ks = k.decode()
            if args.dataset_filter != "both" and not ks.startswith(args.dataset_filter):
                continue
            if slat_txn.get(k) is None:
                test_keys.append(k)
                if len(test_keys) >= args.n_samples * 10:
                    break
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(test_keys), size=min(args.n_samples, len(test_keys)), replace=False)
    test_keys = [test_keys[i] for i in picks]
    print(f"Visualizing {len(test_keys)} test contexts:")

    # Cũng load 1 train context để so sánh
    train_keys = []
    with slat_env.begin() as slat_txn:
        for k, _ in slat_txn.cursor():
            if k == b"__meta__":
                continue
            ks = k.decode()
            if args.dataset_filter != "both" and not ks.startswith(args.dataset_filter):
                continue
            train_keys.append(k)
            if len(train_keys) >= 5:
                break
    train_key = train_keys[0] if train_keys else None

    # Load contexts
    contexts, names = [], []
    with ctx_env.begin() as txn:
        for k in test_keys:
            raw = txn.get(k)
            ctx = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False).float()
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            contexts.append(ctx[0].numpy())
            names.append(k.decode())
        # Train sample for comparison
        train_ctx = None
        if train_key:
            raw = txn.get(train_key)
            tc = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False).float()
            if tc.ndim == 1:
                tc = tc.unsqueeze(0)
            train_ctx = tc[0].numpy()
            train_name = train_key.decode()

    # Split components
    def split(v):
        return v[:512], v[512:562], v[562:946]

    # === 1. Stats summary ===
    stats_lines = []
    stats_lines.append(f"{'Sample':<50} {'ArcFace(512)':>30} {'FLAME(50)':>25} {'DINOv2(384)':>30}")
    stats_lines.append("-" * 135)
    stats_lines.append(f"{'  (norm | mean | std)':<50}")
    for ctx_v, name in zip(contexts, names):
        arc, flame, dino = split(ctx_v)
        s = (f"{name:<50} "
             f"{np.linalg.norm(arc):.3f}|{arc.mean():+.3f}|{arc.std():.3f}".ljust(33) +
             f"  {np.linalg.norm(flame):.3f}|{flame.mean():+.3f}|{flame.std():.3f}".ljust(28) +
             f"  {np.linalg.norm(dino):.3f}|{dino.mean():+.3f}|{dino.std():.3f}")
        stats_lines.append(s)
    if train_ctx is not None:
        arc, flame, dino = split(train_ctx)
        stats_lines.append("")
        stats_lines.append(f"[TRAIN baseline] {train_name}")
        stats_lines.append(
            f"{'  ':<50} "
            f"{np.linalg.norm(arc):.3f}|{arc.mean():+.3f}|{arc.std():.3f}".ljust(83) +
            f"  {np.linalg.norm(flame):.3f}|{flame.mean():+.3f}|{flame.std():.3f}".ljust(28) +
            f"  {np.linalg.norm(dino):.3f}|{dino.mean():+.3f}|{dino.std():.3f}"
        )
    print("\n".join(stats_lines))
    with open(os.path.join(args.out_dir, "stats.txt"), "w") as f:
        f.write("\n".join(stats_lines))
    print(f"\n✓ Stats: {args.out_dir}/stats.txt")

    # === 2. Visualize each context: 3 subplots (ArcFace | FLAME | DINOv2) ===
    for ctx_v, name in zip(contexts, names):
        arc, flame, dino = split(ctx_v)
        safe_name = name.replace("/", "_").replace(".obj", "")

        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        fig.suptitle(f"Hybrid Context (946-dim): {name}", fontsize=13)

        # ArcFace 512
        axes[0].bar(np.arange(512), arc, color="#1f77b4", width=1.0)
        axes[0].set_title(f"ArcFace identity [512] — ‖v‖={np.linalg.norm(arc):.3f}, "
                          f"μ={arc.mean():+.3f}, σ={arc.std():.3f}")
        axes[0].set_xlabel("dim")
        axes[0].axhline(0, color="k", linewidth=0.5)
        axes[0].grid(alpha=0.3)

        # FLAME 50
        axes[1].bar(np.arange(50), flame, color="#ff7f0e", width=0.8)
        axes[1].set_title(f"FLAME expression [50] — ‖v‖={np.linalg.norm(flame):.3f}, "
                          f"μ={flame.mean():+.3f}, σ={flame.std():.3f}")
        axes[1].set_xlabel("dim")
        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].grid(alpha=0.3)

        # DINOv2 384
        axes[2].bar(np.arange(384), dino, color="#2ca02c", width=1.0)
        axes[2].set_title(f"DINOv2 back-of-head [384] — ‖v‖={np.linalg.norm(dino):.3f}, "
                          f"μ={dino.mean():+.3f}, σ={dino.std():.3f}")
        axes[2].set_xlabel("dim")
        axes[2].axhline(0, color="k", linewidth=0.5)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"{safe_name}_context.png")
        plt.savefig(out_path, dpi=80)
        plt.close()
        print(f"  ✓ {out_path}")

    # === 3. Heatmap so sánh all samples (1 row per sample) ===
    fig, ax = plt.subplots(figsize=(16, max(3, len(contexts) * 0.6 + 1)))
    all_ctx = np.stack(contexts)
    if train_ctx is not None:
        all_ctx = np.vstack([all_ctx, train_ctx[None]])
        labels = names + [f"[TRAIN] {train_name}"]
    else:
        labels = names

    # Normalize per-component for visualization (otherwise ArcFace dominates)
    vis = np.zeros_like(all_ctx)
    for i in range(len(all_ctx)):
        vis[i, :512] = all_ctx[i, :512] / (np.abs(all_ctx[i, :512]).max() + 1e-6)
        vis[i, 512:562] = all_ctx[i, 512:562] / (np.abs(all_ctx[i, 512:562]).max() + 1e-6)
        vis[i, 562:946] = all_ctx[i, 562:946] / (np.abs(all_ctx[i, 562:946]).max() + 1e-6)

    im = ax.imshow(vis, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axvline(511.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.axvline(561.5, color="black", linewidth=2, linestyle="--", alpha=0.5)
    ax.text(256, -0.5, "ArcFace (512)", ha="center", fontsize=11, fontweight="bold")
    ax.text(537, -0.5, "FLAME (50)", ha="center", fontsize=11, fontweight="bold")
    ax.text(754, -0.5, "DINOv2 (384)", ha="center", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Feature dimension")
    ax.set_title("Hybrid Context Heatmap (per-component normalized to ±1 for vis)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    heatmap_path = os.path.join(args.out_dir, "_heatmap_all.png")
    plt.savefig(heatmap_path, dpi=80)
    plt.close()
    print(f"\n✓ Heatmap so sánh: {heatmap_path}")

    # === 4. Note about corresponding GT mesh ===
    print(f"\nGT meshes tương ứng (xem trong dataset gốc):")
    for name in names:
        if name.startswith("faceverse"):
            print(f"  - /mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/{name.split('/', 1)[1]}")
        elif name.startswith("facescape"):
            print(f"  - /mnt/16TData/Datasets/FaceScape/{name.split('/', 1)[1]}")


if __name__ == "__main__":
    main()
