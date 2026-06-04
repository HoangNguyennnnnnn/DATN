#!/usr/bin/env python3
"""Decode mesh FaceVerse + FaceScape qua SC-VAE both → so sánh round-trip 2 dataset.
encode ovoxel → SC-VAE → decode → DC mesh. Dùng sc_vae_both/latest_step."""
import io, os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lmdb, torch
import spconv.pytorch as spconv
from src.config import TrainConfig
from src.models.sc_vae import SC_VAE
from scripts.test.test_sc_vae_recon_v2 import extract_ovoxel_mesh


def save_ply(verts, faces, colors, path):
    import trimesh
    if verts is None or len(verts) == 0:
        print(f"  [skip] empty mesh {path}"); return False
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if colors is not None and len(colors) == len(verts):
        import numpy as np
        c = (colors * 255).clip(0, 255).astype('uint8') if colors.max() <= 1.01 else colors.astype('uint8')
        m.visual.vertex_colors = c
    m.export(path); return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-vae-ckpt", default="checkpoints/sc_vae_both/latest_step.pt")
    ap.add_argument("--ovoxel-lmdb", default="data/ovoxel_cache_lmdb")
    ap.add_argument("--out-dir", default="outputs_scvae_both")
    ap.add_argument("--n-each", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    cfg = TrainConfig()

    sc = SC_VAE(in_channels=int(cfg.sc_vae.in_channels), latent_dim=int(cfg.sc_vae.latent_dim), device=dev).to(dev)
    ck = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    sc.load_state_dict(ck.get("model_state_dict", ck), strict=False); sc.eval()
    print(f"SC-VAE: {args.sc_vae_ckpt} epoch={ck.get('epoch')}")

    # Lấy key TRỰC TIẾP từ manifest (O(1) txn.get, KHÔNG scan cursor — tránh nghẽn HDD)
    import json
    man = json.load(open("data/mesh_manifest.json"))
    def to_key(rel):  # 064_03/064_03.obj → 064_03_064_03.c10.shape_mat.mx350000.pt
        return rel.replace("/", "_").replace(".obj", ".c10.shape_mat.mx350000.pt")
    env = lmdb.open(args.ovoxel_lmdb, readonly=True, lock=False, readahead=False)
    fv, fs = [], []
    with env.begin() as t:
        for rel in man["faceverse"]:
            if len(fv) >= args.n_each: break
            vb = t.get(to_key(rel).encode())
            if vb: fv.append((to_key(rel), bytes(vb)))
        for rel in man["facescape"]:
            if len(fs) >= args.n_each: break
            vb = t.get(to_key(rel).encode())
            if vb: fs.append((to_key(rel), bytes(vb)))
    env.close()
    print(f"FaceVerse: {[k for k,_ in fv]}")
    print(f"FaceScape: {[k for k,_ in fs]}")

    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    for tag, items in [("fv", fv), ("fs", fs)]:
        for i, (kd, vb) in enumerate(items):
            p = torch.load(io.BytesIO(vb), map_location="cpu", weights_only=False)
            feats = torch.as_tensor(p["features"])[:, :10].float().to(dev)
            coords = torch.as_tensor(p["coords"]).int().to(dev)
            N = coords.shape[0]
            bcol = torch.zeros((N, 1), dtype=torch.int32, device=dev)
            si = spconv.SparseConvTensor(feats, torch.cat([bcol, coords], 1), [256, 256, 256], 1)
            with torch.no_grad():
                mu, _, xi, xs = sc.encode(si, return_indices=True)
                vox, _, _, oi = sc.decode(mu, original_indices=xi, batch_size=1, return_indices=True)
            cd = oi[:, 1:].int()
            try:
                verts, faces, colors = extract_ovoxel_mesh(coords=cd, feats=vox.float(), aabb=aabb,
                                                           res=256, is_logits=True, threshold=0.5, target_faces=0)
                out = os.path.join(args.out_dir, f"{tag}_{i}.ply")
                ok = save_ply(verts, faces, colors, out)
                print(f"  [{tag}_{i}] {kd[:40]}: in_vox={N} recon_vox={vox.shape[0]} "
                      f"verts={len(verts) if verts is not None else 0} saved={ok}")
            except Exception as e:
                import traceback; print(f"  [{tag}_{i}] ERR {e}"); print(traceback.format_exc()[-400:])


if __name__ == "__main__":
    main()
