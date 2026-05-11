#!/usr/bin/env python3
"""
Precompute Slat + hybrid context cache for iMF (Stage 2)
=========================================================
Mỗi mesh → một file ``.pt`` trong ``data/slat_cache`` / ``data/slat_cache_facescape``
chứa ``slat`` (SC-VAE encode) và ``context`` (946-D: ArcFace + FLAME + DINOv2-back),
đúng cùng ``cache_tag`` / ``cache_contract`` với ``SlatDataset`` trong ``train_imf.py``.

Sau khi chạy xong, huấn luyện iMF với ``--offline-data`` để **không** nạp SC-VAE,
MeshRenderer, ArcFace, FLAME, DINO trên GPU → tiết kiệm VRAM, có thể tăng batch size.

Mặc định bật đầy đủ extractors (khớp pipeline train). Chỉ dùng ``--use-random-context``
khi debug nhanh (context không phải hybrid thật).

Usage:
    conda activate facediff
    python scripts/precompute_slat_cache.py \\
        --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \\
        --dataset both --num-workers 0 --skip-existing

    python src/train_imf.py --offline-data --batch-size 64 ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _build_one_slat_dataset(
    *,
    cfg,
    imf_cfg,
    sc_vae,
    renderer,
    arcface,
    flame,
    dinov2,
    data_root: str,
    dataset_name: str,
    cache_dir: str,
    include_ids,
    exclude_ids,
    single_ckpt_path: str,
):
    from src.train_imf import SlatDataset

    return SlatDataset(
        data_root=data_root,
        sc_vae=sc_vae,
        dataset_name=dataset_name,
        mesh_renderer=renderer,
        arcface=arcface,
        flame=flame,
        dinov2=dinov2,
        slat_length=int(imf_cfg.slat_length),
        latent_dim=int(imf_cfg.input_dim),
        cache_dir=cache_dir,
        device=str(cfg.device),
        include_ids=include_ids,
        exclude_ids=exclude_ids,
        dual_branch=bool(imf_cfg.dual_branch),
        shape_sc_vae=None,
        material_sc_vae=None,
        shape_feature_mode=str(imf_cfg.shape_feature_mode),
        material_feature_mode=str(imf_cfg.material_feature_mode),
        shape_target_in_channels=int(imf_cfg.shape_target_in_channels),
        material_target_in_channels=int(imf_cfg.material_target_in_channels),
        context_dim=int(imf_cfg.context_dim),
        single_sc_vae_checkpoint=single_ckpt_path if not imf_cfg.dual_branch else None,
        shape_sc_vae_checkpoint=str(imf_cfg.shape_sc_vae_checkpoint or single_ckpt_path),
        material_sc_vae_checkpoint=str(imf_cfg.material_sc_vae_checkpoint or single_ckpt_path),
        ovoxel_resolution=int(getattr(cfg.sc_vae, "ovoxel_resolution", 256)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Slat + context .pt caches for iMF (--offline-data)"
    )
    parser.add_argument(
        "--sc-vae-ckpt",
        type=str,
        required=True,
        help="Checkpoint SC-VAE (cùng file dùng cho train iMF online)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["faceverse", "facescape", "both"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: TrainConfig.device)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua index đã có cache hợp lệ (cùng cache_tag)",
    )
    parser.add_argument(
        "--use-random-context",
        action="store_true",
        help="KHÔNG khởi tạo renderer/ArcFace/FLAME/DINO — context ngẫu nhiên (chỉ để test tốc độ)",
    )
    parser.add_argument(
        "--faceverse-train-ids",
        type=str,
        default="train_faceverse_ids.txt",
    )
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt")
    parser.add_argument("--disable-id-filters", action="store_true")
    parser.add_argument(
        "--fv-cache-dir",
        type=str,
        default="data/slat_cache",
        help="Thư mục cache FaceVerse (mặc định giống train_imf)",
    )
    parser.add_argument(
        "--fs-cache-dir",
        type=str,
        default="data/slat_cache_facescape",
        help="Thư mục cache FaceScape (mặc định giống train_imf)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sc_vae_ckpt):
        raise FileNotFoundError(args.sc_vae_ckpt)

    from src.config import TrainConfig
    from src.models.sc_vae import SC_VAE
    from src.data.mesh_renderer import MeshRenderer
    from src.data.arcface_extractor import ArcFaceExtractor
    from src.data.flame_adapter import FLAMEExpressionAdapter
    from src.data.feature_extractor import DinoV3Extractor
    from src.utils import load_identity_set

    cfg = TrainConfig()
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    imf_cfg = cfg.imf

    if imf_cfg.dual_branch:
        raise RuntimeError(
            "precompute_slat_cache.py hiện chỉ hỗ trợ single-branch. "
            "Tắt dual_branch trong config hoặc mở rộng script với hai SC-VAE."
        )

    print(f"[Precompute] device={device}")
    print(f"[Precompute] SC-VAE ckpt: {args.sc_vae_ckpt}")
    print(f"[Precompute] shape_feature_mode={imf_cfg.shape_feature_mode} "
          f"in_ch={imf_cfg.shape_target_in_channels} slat_len={imf_cfg.slat_length}")

    sc_vae = SC_VAE(
        in_channels=int(imf_cfg.shape_target_in_channels),
        latent_dim=int(imf_cfg.input_dim),
        device=str(device),
        rho_prune_threshold=float(getattr(cfg.sc_vae, "rho_prune_threshold", 0.5)),
        encoder_dims=list(getattr(cfg.sc_vae, "encoder_dims", [64, 128, 256, 512])),
        num_res_blocks=int(getattr(cfg.sc_vae, "num_res_blocks", 2)),
    )
    ckpt = torch.load(args.sc_vae_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    sc_vae.load_state_dict(state, strict=True)
    sc_vae = sc_vae.to(device).eval()

    if args.use_random_context:
        renderer = arcface = flame = dinov2 = None
        print("[Precompute] WARN: --use-random-context → context trong cache KHÔNG phải hybrid thật")
    else:
        print("[Precompute] Loading hybrid context stack (renderer + ArcFace + FLAME + DINOv2)...")
        renderer = MeshRenderer(device=str(device), image_size=512)
        arcface = ArcFaceExtractor(device=str(device))
        flame = FLAMEExpressionAdapter(expression_dim=50, device=str(device))
        dinov2 = DinoV3Extractor(model_name="facebook/dinov2-small", device=str(device))

    if args.disable_id_filters:
        fv_inc = fv_exc = fs_inc = fs_exc = None
    else:
        fv_inc = load_identity_set(args.faceverse_train_ids)
        fv_exc = load_identity_set(args.faceverse_test_ids)
        fs_inc = load_identity_set(args.facescape_train_ids)
        fs_exc = load_identity_set(args.facescape_test_ids)

    datasets: list[tuple[str, object]] = []
    if args.dataset in ("faceverse", "both") and os.path.isdir(cfg.data.faceverse_root):
        ds = _build_one_slat_dataset(
            cfg=cfg,
            imf_cfg=imf_cfg,
            sc_vae=sc_vae,
            renderer=renderer,
            arcface=arcface,
            flame=flame,
            dinov2=dinov2,
            data_root=cfg.data.faceverse_root,
            dataset_name="faceverse",
            cache_dir=args.fv_cache_dir,
            include_ids=fv_inc,
            exclude_ids=fv_exc,
            single_ckpt_path=os.path.abspath(args.sc_vae_ckpt),
        )
        if len(ds) > 0:
            datasets.append(("faceverse", ds))
    if args.dataset in ("facescape", "both") and os.path.isdir(cfg.data.facescape_root):
        ds = _build_one_slat_dataset(
            cfg=cfg,
            imf_cfg=imf_cfg,
            sc_vae=sc_vae,
            renderer=renderer,
            arcface=arcface,
            flame=flame,
            dinov2=dinov2,
            data_root=cfg.data.facescape_root,
            dataset_name="facescape",
            cache_dir=args.fs_cache_dir,
            include_ids=fs_inc,
            exclude_ids=fs_exc,
            single_ckpt_path=os.path.abspath(args.sc_vae_ckpt),
        )
        if len(ds) > 0:
            datasets.append(("facescape", ds))

    if not datasets:
        raise RuntimeError("Không có dataset nào khả dụng (kiểm tra data roots và --dataset).")

    total = sum(len(ds) for _, ds in datasets)
    print(f"[Precompute] cache_tag (first ds) = {datasets[0][1].cache_tag}")
    print(f"[Precompute] total samples: {total}")

    t0 = time.time()
    processed = skipped = errors = 0

    for name, ds in datasets:
        print(f"\n[Precompute] === {name} ({len(ds)} meshes) → {ds.cache_dir} ===")
        for i in range(len(ds)):
            try:
                if args.skip_existing and ds.has_valid_cache(i):
                    skipped += 1
                    processed += 1
                    continue
                _ = ds[i]
                processed += 1
            except Exception as exc:
                errors += 1
                processed += 1
                if errors <= 15:
                    print(f"  [ERROR] {name}[{i}]: {exc}")

            if processed % 200 == 0 or processed == total:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-8)
                eta_min = (total - processed) / max(rate, 1e-8) / 60.0
                print(
                    f"  [{processed}/{total}] ok_errors={errors} skipped_valid={skipped} "
                    f"rate={rate:.2f}/s ETA~{eta_min:.0f}min"
                )

    elapsed = time.time() - t0
    print(f"\n[Precompute] Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  touched={processed} skipped_existing={skipped} errors={errors}")
    print("\nNext: python src/train_imf.py --offline-data --batch-size <lớn hơn> ...")


if __name__ == "__main__":
    main()
