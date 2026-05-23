"""
Tính per-channel mean và std của slat tokens từ toàn bộ data trong slat_context.lmdb.

Output: data/slat_stats.pt
{
    'mean': torch.Tensor[32],
    'std':  torch.Tensor[32],
    'n_samples': int,
    'n_tokens_per_sample': int,
    'computed_from': str,
    'sc_vae_ckpt': str (optional),
}

Sử dụng 2-pass streaming (welford-like):
  Pass 1: sum, count → mean
  Pass 2: sum of squared deviations → std
Tiết kiệm memory: chỉ giữ accumulators float64 size [32], không load toàn bộ dataset.
"""
import argparse
import io
import os
import sys
import time

import lmdb
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context.lmdb")
    ap.add_argument("--out", default="data/slat_stats.pt")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_500.pt")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Giới hạn số samples (debug); default = toàn bộ")
    args = ap.parse_args()

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, max_readers=64)

    # === Pass 1: count, sum ===
    print(f"[Pass 1/2] Computing sum and count from {args.lmdb}...")
    t0 = time.time()
    n_total_tokens = 0
    sum_per_ch = None  # float64 [32]
    n_samples = 0
    sample_shape = None

    with env.begin() as txn:
        cur = txn.cursor()
        for k, v in cur:
            if k == b"__meta__":
                continue
            blob = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
            slat = blob["slat"]
            if not torch.is_tensor(slat):
                slat = torch.as_tensor(slat)
            slat = slat.to(torch.float64)  # avoid precision loss in sum
            if sum_per_ch is None:
                sample_shape = tuple(slat.shape)
                sum_per_ch = torch.zeros(slat.shape[-1], dtype=torch.float64)
            sum_per_ch += slat.sum(dim=tuple(range(slat.ndim - 1)))
            n_total_tokens += slat.numel() // slat.shape[-1]
            n_samples += 1
            if args.max_samples and n_samples >= args.max_samples:
                break
            if n_samples % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_samples * (20369 - n_samples) if n_samples > 0 else 0
                print(f"  [{n_samples}] sample shape={sample_shape}, elapsed={elapsed:.1f}s, ETA={eta:.0f}s")

    mean_per_ch = sum_per_ch / n_total_tokens
    print(f"[Pass 1] done in {time.time()-t0:.1f}s: n_samples={n_samples}, n_tokens={n_total_tokens}")
    print(f"  mean per-ch: min={mean_per_ch.min():.4f}, max={mean_per_ch.max():.4f}, mean={mean_per_ch.mean():.4f}")

    # === Pass 2: sum of squared deviations ===
    print(f"\n[Pass 2/2] Computing variance...")
    t0 = time.time()
    sumsq = torch.zeros_like(mean_per_ch)
    n_seen = 0
    mean_ref = mean_per_ch  # shape [32]
    with env.begin() as txn:
        cur = txn.cursor()
        for k, v in cur:
            if k == b"__meta__":
                continue
            blob = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
            slat = blob["slat"]
            if not torch.is_tensor(slat):
                slat = torch.as_tensor(slat)
            slat = slat.to(torch.float64)
            # Broadcast subtract: slat [L, 32], mean_ref [32]
            diff = slat - mean_ref
            sumsq += (diff * diff).sum(dim=tuple(range(slat.ndim - 1)))
            n_seen += 1
            if args.max_samples and n_seen >= args.max_samples:
                break
            if n_seen % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_seen * (n_samples - n_seen)
                print(f"  [{n_seen}/{n_samples}] elapsed={elapsed:.1f}s, ETA={eta:.0f}s")

    var_per_ch = sumsq / n_total_tokens
    std_per_ch = torch.sqrt(var_per_ch)
    print(f"[Pass 2] done in {time.time()-t0:.1f}s")
    print(f"  std per-ch: min={std_per_ch.min():.4f}, max={std_per_ch.max():.4f}, mean={std_per_ch.mean():.4f}")

    # === Save ===
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "mean": mean_per_ch.to(torch.float32),
        "std": std_per_ch.to(torch.float32),
        "n_samples": n_samples,
        "n_tokens_per_sample": int(sample_shape[0]) if sample_shape else None,
        "n_total_tokens": n_total_tokens,
        "latent_dim": int(sum_per_ch.shape[0]),
        "computed_from": os.path.abspath(args.lmdb),
        "sc_vae_ckpt": args.sc_vae_ckpt,
    }
    torch.save(out, args.out)
    print(f"\n✓ Saved to {args.out}")
    print(f"  mean[0:8] = {mean_per_ch[:8].tolist()}")
    print(f"  std[0:8]  = {std_per_ch[:8].tolist()}")


if __name__ == "__main__":
    main()
