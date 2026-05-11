#!/usr/bin/env python
"""
Compare SC-VAE checkpoints on a fixed set of batches from the current LMDB-backed dataset.
"""

import argparse
import copy
import functools
import json
import os
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainConfig
from src.models.sc_vae import SC_VAE
from src.models.sc_vae_loss import SCVAELoss
from src.scvae_train.data import (
    VoxelDataset,
    build_sparse_batch,
    collate_voxels,
    is_packed_micro_batch,
)
from src.scvae_train.metrics import (
    compute_chamfer_distance,
    compute_normal_consistency,
    compute_voxel_iou,
)
from src.scvae_train.runtime import align_recon_target
from src.scvae_train.runtime import is_oom_error as _is_oom_error
from src.scvae_train.runtime import is_sparse_runtime_error as _is_sparse_runtime_error
from src.train_sc_vae import _build_data_signature, _build_resume_contract
from src.utils import load_identity_set


def _clone_payload(value: Any):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_payload(v) for v in value]
    return copy.deepcopy(value)


def _unwrap_eval_batches(batch_payload) -> List[Any]:
    if isinstance(batch_payload, dict) and batch_payload.get("is_pre_concatenated"):
        return [batch_payload]
    if is_packed_micro_batch(batch_payload):
        return [payload for payload in batch_payload if isinstance(payload, list) and len(payload) > 0]
    if isinstance(batch_payload, list) and len(batch_payload) > 0:
        return [batch_payload]
    return []


def _build_eval_dataset(cfg: TrainConfig, args) -> tuple:
    datasets_to_concat = []

    if args.disable_id_filters:
        faceverse_include_ids = None
        faceverse_exclude_ids = None
        facescape_include_ids = None
        facescape_exclude_ids = None
    else:
        faceverse_include_ids = load_identity_set(args.faceverse_train_ids)
        faceverse_exclude_ids = load_identity_set(args.faceverse_test_ids)
        facescape_include_ids = load_identity_set(args.facescape_train_ids)
        facescape_exclude_ids = load_identity_set(args.facescape_test_ids)

    if cfg.data.active_dataset in ["faceverse", "both"] and os.path.isdir(cfg.data.faceverse_root):
        fv_dataset = VoxelDataset(
            data_root=cfg.data.faceverse_root,
            dataset_name="faceverse",
            max_voxels=cfg.sc_vae.max_voxels_per_mesh,
            cache_dir=os.path.join("data", "ovoxel_cache_recached", "faceverse"),
            use_ovoxel_converter=False,
            ovoxel_resolution=cfg.sc_vae.ovoxel_resolution,
            require_ovoxel_converter=False,
            target_in_channels=cfg.sc_vae.in_channels,
            feature_mode=cfg.sc_vae.input_feature_mode,
            include_ids=faceverse_include_ids,
            exclude_ids=faceverse_exclude_ids,
            lmdb_dir=cfg.data.lmdb_dir,
            lmdb_readahead=cfg.data.lmdb_readahead,
            lmdb_only=bool(args.lmdb_only),
            device="cpu",
        )
        if len(fv_dataset) > 0:
            datasets_to_concat.append(fv_dataset)

    if cfg.data.active_dataset in ["facescape", "both"] and os.path.isdir(cfg.data.facescape_root):
        fs_dataset = VoxelDataset(
            data_root=cfg.data.facescape_root,
            dataset_name="facescape",
            max_voxels=cfg.sc_vae.max_voxels_per_mesh,
            cache_dir=os.path.join("data", "ovoxel_cache_recached", "facescape"),
            use_ovoxel_converter=False,
            ovoxel_resolution=cfg.sc_vae.ovoxel_resolution,
            require_ovoxel_converter=False,
            target_in_channels=cfg.sc_vae.in_channels,
            feature_mode=cfg.sc_vae.input_feature_mode,
            include_ids=facescape_include_ids,
            exclude_ids=facescape_exclude_ids,
            lmdb_dir=cfg.data.lmdb_dir,
            lmdb_readahead=cfg.data.lmdb_readahead,
            lmdb_only=bool(args.lmdb_only),
            device="cpu",
        )
        if len(fs_dataset) > 0:
            datasets_to_concat.append(fs_dataset)

    if not datasets_to_concat:
        raise RuntimeError("No dataset samples found for evaluation.")

    dataset = ConcatDataset(datasets_to_concat) if len(datasets_to_concat) > 1 else datasets_to_concat[0]

    val_count = int(len(dataset) * max(float(cfg.sc_vae.val_split), 0.0))
    val_count = min(max(val_count, 0), max(len(dataset) - 1, 0))
    if val_count > 0:
        train_count = len(dataset) - val_count
        split_gen = torch.Generator().manual_seed(cfg.seed)
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=split_gen)
        eval_dataset = val_dataset
    else:
        train_dataset = dataset
        val_dataset = None
        eval_dataset = dataset

    return dataset, train_dataset, val_dataset, eval_dataset


def _collect_payloads(dataloader: DataLoader, num_batches: int) -> List[Any]:
    payloads: List[Any] = []
    for batch_payload in dataloader:
        for payload in _unwrap_eval_batches(batch_payload):
            payloads.append(_clone_payload(payload))
            if len(payloads) >= num_batches:
                return payloads
    return payloads


def _evaluate_checkpoint(
    checkpoint_path: str,
    payloads: List[Any],
    cfg: TrainConfig,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = SC_VAE(
        in_channels=cfg.sc_vae.in_channels,
        latent_dim=cfg.sc_vae.latent_dim,
        device=str(device),
        rho_prune_threshold=float(getattr(cfg.sc_vae, "rho_prune_threshold", 0.5)),
        encoder_dims=getattr(cfg.sc_vae, "encoder_dims", [64, 128, 256, 512]),
        num_res_blocks=getattr(cfg.sc_vae, "num_res_blocks", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loss_fn = SCVAELoss(
        kl_weight=cfg.sc_vae.kl_weight,
        use_bce_for_geom=cfg.sc_vae.use_bce_for_geom,
        rho_loss_weight=cfg.sc_vae.rho_loss_weight,
    )

    amp_enabled = bool(cfg.sc_vae.use_amp) and device.type == "cuda"
    amp_dtype = torch.float16

    metrics = {
        "total_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "rho_loss": 0.0,
        "chamfer_distance": 0.0,
        "normal_consistency": 0.0,
        "voxel_iou": 0.0,
        "mismatch_batches": 0,
        "skipped_batches": 0,
        "batches": 0,
    }

    with torch.no_grad():
        for payload in payloads:
            try:
                sparse_input, target_x = build_sparse_batch(
                    _clone_payload(payload),
                    device=device,
                    spatial_size=cfg.data.voxel_resolution,
                )

                autocast_enabled = amp_enabled
                if device.type == "cuda":
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled)
                else:
                    autocast_ctx = torch.amp.autocast("cpu", enabled=False)

                with autocast_ctx:
                    recon_x, mu, logvar, rho_logits, rho_targets, out_indices = model(sparse_input)
                    recon_aligned, target_aligned, mismatch = align_recon_target(
                        recon_x,
                        out_indices,
                        target_x,
                        sparse_input.indices,
                    )
                    loss_dict = loss_fn(
                        recon_aligned,
                        target_aligned,
                        mu,
                        logvar,
                        feature_mode=cfg.sc_vae.input_feature_mode,
                        rho_logits_list=rho_logits,
                        rho_targets_list=rho_targets,
                    )
            except RuntimeError as exc:
                if not _is_oom_error(exc) and not _is_sparse_runtime_error(exc):
                    raise
                metrics["skipped_batches"] += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            total_loss = (
                loss_dict["recon_loss"]
                + float(cfg.sc_vae.kl_weight) * loss_dict["kl_loss"]
                + float(getattr(cfg.sc_vae, "rho_loss_weight", 0.0)) * loss_dict.get("rho_loss", target_x.new_zeros(()))
            )

            metrics["total_loss"] += float(total_loss.item())
            metrics["recon_loss"] += float(loss_dict["recon_loss"].item())
            metrics["kl_loss"] += float(loss_dict["kl_loss"].item())
            metrics["rho_loss"] += float(loss_dict.get("rho_loss", target_x.new_zeros(())).item())
            metrics["chamfer_distance"] += compute_chamfer_distance(
                recon_aligned[:, :3],
                target_aligned[:, :3],
            )
            if target_aligned.shape[1] >= 6:
                metrics["normal_consistency"] += compute_normal_consistency(
                    recon_aligned[:, 3:6],
                    target_aligned[:, 3:6],
                )
            if len(rho_logits) > 0 and len(rho_targets) > 0:
                metrics["voxel_iou"] += compute_voxel_iou(rho_logits[-1], rho_targets[-1])
            metrics["mismatch_batches"] += int(bool(mismatch))
            metrics["batches"] += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

    count = max(metrics["batches"], 1)
    for key in ("total_loss", "recon_loss", "kl_loss", "rho_loss", "chamfer_distance", "normal_consistency", "voxel_iou"):
        metrics[key] /= count

    if metrics["batches"] <= 0:
        raise RuntimeError("All evaluation payloads were skipped by sparse/OOM guards.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ckpt, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SC-VAE checkpoints on fixed LMDB batches")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="artifacts/scvae_eval_resume_candidates/summary.json")
    parser.add_argument("--device", type=str, default=None, help="cuda:0, cpu, or omit for auto")
    parser.add_argument("--dataset", type=str, choices=["facescape", "faceverse", "both"], default="both")
    parser.add_argument("--facescape-root", type=str, default=None)
    parser.add_argument("--faceverse-root", type=str, default=None)
    parser.add_argument("--feature-mode", type=str, default="shape_mat", choices=["geom6", "geom_mat12", "mat6", "rgb3", "shape_native", "shape_mat"])
    parser.add_argument("--in-channels", type=int, default=10)
    parser.add_argument("--lmdb-dir", type=str, default="data/ovoxel_cache_lmdb")
    parser.add_argument("--lmdb-only", action="store_true")
    parser.add_argument("--lmdb-readahead", action="store_true")
    parser.add_argument("--no-lmdb-readahead", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-voxels", type=int, default=350000)
    parser.add_argument("--max-points-per-batch", type=int, default=10000000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=33)
    parser.add_argument("--disable-id-filters", action="store_true")
    parser.add_argument("--faceverse-train-ids", type=str, default="train_faceverse_ids.txt")
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data.active_dataset = args.dataset
    cfg.sc_vae.input_feature_mode = args.feature_mode
    cfg.sc_vae.in_channels = max(1, int(args.in_channels))
    cfg.sc_vae.max_voxels_per_mesh = max(1, int(args.max_voxels))
    cfg.sc_vae.max_points_per_batch = max(0, int(args.max_points_per_batch))
    cfg.sc_vae.batch_size = max(1, int(args.batch_size))
    cfg.sc_vae.val_split = max(0.0, min(0.5, float(args.val_split)))
    cfg.data.num_workers = max(0, int(args.num_workers))
    cfg.data.lmdb_dir = args.lmdb_dir
    if args.facescape_root:
        cfg.data.facescape_root = args.facescape_root
    if args.faceverse_root:
        cfg.data.faceverse_root = args.faceverse_root
    if args.lmdb_readahead:
        cfg.data.lmdb_readahead = True
    if args.no_lmdb_readahead:
        cfg.data.lmdb_readahead = False

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[eval] device={device}", flush=True)

    print("[eval] building dataset...", flush=True)
    dataset, train_dataset, val_dataset, eval_dataset = _build_eval_dataset(cfg, args)
    print(
        f"[eval] dataset ready total={len(dataset)} eval={len(eval_dataset)}",
        flush=True,
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.sc_vae.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=functools.partial(
            collate_voxels,
            spatial_size=cfg.data.voxel_resolution,
            max_points_per_batch=int(cfg.sc_vae.max_points_per_batch),
        ),
    )
    print("[eval] collecting fixed payloads...", flush=True)
    payloads = _collect_payloads(dataloader, max(1, int(args.num_batches)))
    if len(payloads) == 0:
        raise RuntimeError("No evaluation payloads collected.")
    print(f"[eval] collected payloads={len(payloads)}", flush=True)

    steps_per_epoch_estimate = max(len(dataloader) // max(int(args.gradient_accumulation_steps), 1), 1)
    expected_data_signature = _build_data_signature(
        cfg,
        cfg.sc_vae,
        dataset,
        train_dataset,
        val_dataset,
        faceverse_train_ids_file=args.faceverse_train_ids,
        faceverse_test_ids_file=args.faceverse_test_ids,
        facescape_train_ids_file=args.facescape_train_ids,
        facescape_test_ids_file=args.facescape_test_ids,
        lmdb_only=bool(args.lmdb_only),
    )
    expected_resume_contract = _build_resume_contract(
        cfg.sc_vae,
        steps_per_epoch_estimate=steps_per_epoch_estimate,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        loader_split_points=0,
        lmdb_only=bool(args.lmdb_only),
    )

    results = []
    for checkpoint_path in args.checkpoints:
        print(f"[eval] evaluating {checkpoint_path}...", flush=True)
        ckpt, metrics = _evaluate_checkpoint(checkpoint_path, payloads, cfg, device)
        ckpt_data_sig = ckpt.get("data_signature", {})
        ckpt_resume_sig = ckpt.get("resume_contract", {})
        result = {
            "checkpoint": checkpoint_path,
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "stored_loss": ckpt.get("loss"),
            "checkpoint_batch_size": ckpt.get("batch_size"),
            "data_signature": ckpt_data_sig.get("digest"),
            "resume_contract": ckpt_resume_sig.get("digest"),
            "data_signature_matches_current": ckpt_data_sig.get("digest") == expected_data_signature.get("digest"),
            "resume_contract_matches_current": ckpt_resume_sig.get("digest") == expected_resume_contract.get("digest"),
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"{os.path.basename(checkpoint_path)} | "
            f"loss={metrics['total_loss']:.6f} recon={metrics['recon_loss']:.6f} "
            f"cd={metrics['chamfer_distance']:.6f} nc={metrics['normal_consistency']:.4f} "
            f"iou={metrics['voxel_iou']:.4f}",
            flush=True,
        )

    results.sort(key=lambda item: item["metrics"]["total_loss"])
    payload = {
        "device": str(device),
        "num_payloads": len(payloads),
        "dataset_len": len(dataset),
        "eval_dataset_len": len(eval_dataset),
        "expected_data_signature": expected_data_signature.get("digest"),
        "expected_resume_contract": expected_resume_contract.get("digest"),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {args.output}", flush=True)


if __name__ == "__main__":
    main()
