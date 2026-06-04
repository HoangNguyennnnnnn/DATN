#!/usr/bin/env python3
"""Chẩn đoán lỗ thủng mesh→ovoxel→mesh. So sánh từng giai đoạn:
  A. GT ovoxel (raw từ LMDB) → DC mesh  [baseline, lỗ ở đây = bug convert/DC]
  B. GT ovoxel → SC-VAE encode→decode → DC mesh  [lỗ thêm = SC-VAE recon kém]
Đo: voxel count, delta-flag IoU (A vs B), để biết lỗ do SC-VAE hay DC."""
import io, os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from src.config import TrainConfig
from src.models.sc_vae import SC_VAE

VOXEL_MARGIN = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovoxel-lmdb", default="data/ovoxel_cache_lmdb")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_600.pt")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = TrainConfig()

    sc = SC_VAE(in_channels=int(cfg.sc_vae.in_channels), latent_dim=int(cfg.sc_vae.latent_dim), device=dev).to(dev)
    ck = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    sc.load_state_dict(ck.get("model_state_dict", ck), strict=False); sc.eval()
    print(f"SC-VAE epoch={ck.get('epoch')}")

    env = lmdb.open(args.ovoxel_lmdb, readonly=True, lock=False, readahead=False)
    items = []
    with env.begin() as t:
        cur = t.cursor(); cur.first()
        for k, v in cur:
            if k == b"__meta__": continue
            items.append((k.decode(), v));
            if len(items) >= args.n: break
    env.close()

    for name, v in items:
        p = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
        feats = torch.as_tensor(p["features"])[:, :10].float().to(dev)
        coords = torch.as_tensor(p["coords"]).int().to(dev)
        N = coords.shape[0]
        # GT delta flag (kênh 3:6 raw — đã activated trong cache hay logits?)
        gt_delta = feats[:, 3:6]
        gt_flag = (gt_delta > 0.5) if gt_delta.max() <= 1.01 else (gt_delta > 0)
        gt_flag_count = int(gt_flag.sum())

        # B. SC-VAE encode→decode
        bcol = torch.zeros((N, 1), dtype=torch.int32, device=dev)
        si = spconv.SparseConvTensor(feats, torch.cat([bcol, coords], 1), [256, 256, 256], 1)
        with torch.no_grad():
            mu, _, xi, xs = sc.encode(si, return_indices=True)
            vox_feats, _, _, out_idx = sc.decode(mu, original_indices=xi, batch_size=1, return_indices=True)
        rec_delta = vox_feats[:, 3:6]   # logits
        rec_flag = (rec_delta > 0)
        rec_flag_count = int(rec_flag.sum())

        # so voxel count + flag (chỉ so được nếu out_idx khớp coords — đếm thô)
        print(f"\n[{name[:40]}]")
        print(f"  GT: {N} voxels, delta-flag(surface edges)={gt_flag_count} "
              f"({gt_flag_count/(N*3)*100:.0f}% of {N*3} edges)")
        print(f"  SC-VAE recon: {vox_feats.shape[0]} voxels, delta-flag={rec_flag_count} "
              f"({rec_flag_count/(vox_feats.shape[0]*3)*100:.0f}%)")
        print(f"  voxel count ratio recon/GT: {vox_feats.shape[0]/N:.2f}")
        print(f"  → nếu recon flag% << GT flag% = SC-VAE mất bề mặt → lỗ. "
              f"nếu voxel ratio<1 = SC-VAE mất voxel.")


if __name__ == "__main__":
    main()
