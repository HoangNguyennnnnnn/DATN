#!/usr/bin/env python3
"""Decode GT slat (từ LMDB) → SC-VAE → DC mesh. Tách bạch pipeline mesh khỏi Stage 2 generation.
Nếu mesh ra mặt đúng từ GT slat → pipeline (SC-VAE decode + mask + DC) OK, vấn đề chỉ ở generation.
Dùng slat_to_mesh (mask raw norm>0.1 đã đúng) + config in_channels=10."""
import io, os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
from src.config import TrainConfig
from src.models.sc_vae import SC_VAE
from scripts.test.test_e2e_inference import slat_to_mesh, save_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/slat_context_faceverse_balanced.lmdb")
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_shape/epoch_600.pt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out-dir", default="outputs_gt_mesh")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    cfg = TrainConfig()

    print(f"[1] SC-VAE in_channels={cfg.sc_vae.in_channels} latent={cfg.sc_vae.latent_dim}")
    sc = SC_VAE(in_channels=int(cfg.sc_vae.in_channels), latent_dim=int(cfg.sc_vae.latent_dim), device=dev).to(dev)
    ck = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    sc.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    sc.eval()

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        cur = txn.cursor(); cur.first(); done = 0; keys = []
        for k, v in cur:
            if k == b"__meta__": continue
            keys.append((k, v)); done += 1
            if done >= args.n: break
    env.close()

    for i, (k, v) in enumerate(keys):
        b = torch.load(io.BytesIO(v), map_location="cpu", weights_only=False)
        slat = b["slat"].float().to(dev)  # [4096,32] RAW (chưa normalize)
        print(f"\n[{i}] key={k.decode()[:40]} slat norm range [{slat.norm(dim=-1).min():.2f}, {slat.norm(dim=-1).max():.2f}], "
              f"occupied(norm>0.1)={int((slat.norm(dim=-1)>0.1).sum())}/4096")
        slat_raw = slat.unsqueeze(0)  # [1,4096,32]
        try:
            verts, faces, colors, n_vox = slat_to_mesh(slat_raw, sc, dev)  # mask=None → tự mask raw>0.1
            out = os.path.join(args.out_dir, f"gt_{i}.ply")
            ok = save_ply(verts, faces, colors, out)
            print(f"    -> {n_vox} voxels, verts={len(verts) if verts is not None else 0}, saved={ok} {out}")
        except Exception as e:
            import traceback; print(f"    [ERROR] {e}"); print(traceback.format_exc()[-600:])


if __name__ == "__main__":
    main()
