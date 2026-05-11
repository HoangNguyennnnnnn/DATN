"""
FaceDiff — Stage 1: SC-VAE Training
=============================================
Huấn luyện SC-VAE nén O-Voxel features → Slat tokens.
Phải chạy TRƯỚC Stage 2 (iMF).

Tính năng:
- Độ chính xác hỗn hợp (Mixed precision - bfloat16) để tối ưu hiệu quả VRAM
- KL annealing: tăng dần trọng số KL qua các epoch để tránh sự sụp đổ hậu nghiệm (posterior collapse)
- Lưu/Phục hồi checkpoint
- Ghi log bằng WandB
- Cắt gradient (Gradient clipping)

Cách sử dụng:
    python src/train_sc_vae.py
    python src/train_sc_vae.py --feature-mode shape_mat --in-channels 10
    python src/train_sc_vae.py --resume checkpoints/sc_vae_shape/latest_step.pt
"""

import sys
import os
import time
import traceback
import argparse
import random
import functools
import hashlib
import numpy as np
import collections
import math
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset, random_split
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainConfig
from src.models.sc_vae import SC_VAE
from src.models.sc_vae_loss import SCVAELoss
from src.utils import load_identity_set
from src.scvae_train.data import (
    VoxelDataset,
    collate_voxels,
    is_packed_micro_batch as _is_packed_micro_batch,
    materialize_batch_items,
    build_sparse_batch,
    cap_points_per_batch,
)
from src.scvae_train.render import compute_stage2_render_perceptual_loss
from src.scvae_train.runtime import (
    align_recon_target,
    is_oom_error as _is_oom_error,
    is_sparse_runtime_error as _is_sparse_runtime_error,
    get_lr_scheduler,
    get_resume_scheduler,
    make_signature_record,
    save_checkpoint,
    load_checkpoint,
)
from src.scvae_train.metrics import (
    compute_chamfer_distance,
    compute_voxel_iou,
    compute_normal_consistency,
)
from src.scvae_train.visualization import save_validation_samples

# ============================================================
# WandB (optional)
# ============================================================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Train] wandb not installed. pip install wandb")


def _write_crash_report(
    checkpoint_dir: str,
    exc: BaseException,
    current_epoch: int,
    global_step: int,
) -> str:
    """Lưu lại (persist) một traceback ngay cả khi cửa sổ tmux biến mất sau sự cố (crash)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(checkpoint_dir, f"crash_{stamp}.txt")
    latest_path = os.path.join(checkpoint_dir, "crash_latest.txt")
    header = [
        "FaceDiff SC-VAE crash report",
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"epoch: {current_epoch}",
        f"global_step: {global_step}",
        f"exception: {repr(exc)}",
        "",
        "traceback:",
    ]
    for path in (report_path, latest_path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header))
                f.write("\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
    return report_path


def _digest_text_file(path: str):
    if not path or not os.path.exists(path):
        return None
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest_string_iter(values):
    hasher = hashlib.sha1()
    count = 0
    first = None
    last = None
    for value in values:
        text = str(value)
        if first is None:
            first = text
        last = text
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
        count += 1
    return {
        "count": count,
        "sha1": hasher.hexdigest(),
        "first": first,
        "last": last,
    }


def _dataset_signature_fragment(dataset):
    if dataset is None:
        return None

    if isinstance(dataset, Subset):
        hasher = hashlib.sha1()
        for idx in dataset.indices:
            hasher.update(f"{int(idx)},".encode("ascii"))
        return {
            "type": "Subset",
            "len": len(dataset),
            "indices_sha1": hasher.hexdigest(),
            "base": _dataset_signature_fragment(dataset.dataset),
        }

    if isinstance(dataset, ConcatDataset):
        return {
            "type": "ConcatDataset",
            "len": len(dataset),
            "parts": [_dataset_signature_fragment(ds) for ds in dataset.datasets],
        }

    if isinstance(dataset, VoxelDataset):
        rel_samples = [
            os.path.relpath(sample_path, dataset.data_root)
            for sample_path in dataset.samples
        ]
        return {
            "type": "VoxelDataset",
            "dataset_name": dataset.dataset_name,
            "data_root": os.path.abspath(dataset.data_root),
            "len": len(dataset),
            "target_in_channels": int(dataset.target_in_channels),
            "feature_mode": str(dataset.feature_mode),
            "max_voxels": int(dataset.max_voxels),
            "lmdb_dir": os.path.abspath(dataset.lmdb_dir) if dataset.lmdb_dir else None,
            "samples": _digest_string_iter(rel_samples),
        }

    return {
        "type": type(dataset).__name__,
        "len": len(dataset) if hasattr(dataset, "__len__") else None,
    }


def _build_lmdb_signature(lmdb_dir: str):
    signature = {
        "path": os.path.abspath(lmdb_dir) if lmdb_dir else None,
    }
    if not lmdb_dir or not os.path.isdir(lmdb_dir):
        return signature

    for filename in ("data.mdb", "lock.mdb"):
        path = os.path.join(lmdb_dir, filename)
        if os.path.exists(path):
            stat = os.stat(path)
            signature[filename] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }

    try:
        import lmdb

        env = lmdb.open(
            lmdb_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
        )
        try:
            signature["entries"] = int(env.stat().get("entries", 0))
            signature["map_size"] = int(env.info().get("map_size", 0))
        finally:
            env.close()
    except Exception as exc:
        signature["inspect_error"] = f"{type(exc).__name__}: {exc}"

    return signature


def _build_data_signature(
    cfg: TrainConfig,
    vae_cfg,
    dataset,
    train_dataset,
    val_dataset,
    faceverse_train_ids_file: str,
    faceverse_test_ids_file: str,
    facescape_train_ids_file: str,
    facescape_test_ids_file: str,
    lmdb_only: bool,
):
    details = {
        "active_dataset": str(cfg.data.active_dataset),
        "feature_mode": str(vae_cfg.input_feature_mode),
        "in_channels": int(vae_cfg.in_channels),
        "max_voxels_per_mesh": int(vae_cfg.max_voxels_per_mesh),
        "val_split": float(vae_cfg.val_split),
        "seed": int(cfg.seed),
        "lmdb_only": bool(lmdb_only),
        "lmdb_readahead": bool(cfg.data.lmdb_readahead),
        "data_roots": {
            "faceverse_root": os.path.abspath(cfg.data.faceverse_root),
            "facescape_root": os.path.abspath(cfg.data.facescape_root),
        },
        "lmdb": _build_lmdb_signature(cfg.data.lmdb_dir),
        "id_files": {
            "faceverse_train_ids": {
                "path": os.path.abspath(faceverse_train_ids_file),
                "sha1": _digest_text_file(faceverse_train_ids_file),
            },
            "faceverse_test_ids": {
                "path": os.path.abspath(faceverse_test_ids_file),
                "sha1": _digest_text_file(faceverse_test_ids_file),
            },
            "facescape_train_ids": {
                "path": os.path.abspath(facescape_train_ids_file),
                "sha1": _digest_text_file(facescape_train_ids_file),
            },
            "facescape_test_ids": {
                "path": os.path.abspath(facescape_test_ids_file),
                "sha1": _digest_text_file(facescape_test_ids_file),
            },
        },
        "dataset": _dataset_signature_fragment(dataset),
        "train_dataset": _dataset_signature_fragment(train_dataset),
        "val_dataset": _dataset_signature_fragment(val_dataset),
    }
    return make_signature_record(details)


def _build_resume_contract(
    vae_cfg,
    steps_per_epoch_estimate: int,
    gradient_accumulation_steps: int,
    loader_split_points: int,
    lmdb_only: bool,
):
    details = {
        "feature_mode": str(vae_cfg.input_feature_mode),
        "in_channels": int(vae_cfg.in_channels),
        "batch_size": int(vae_cfg.batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "loader_split_points": int(loader_split_points),
        "steps_per_epoch": int(steps_per_epoch_estimate),
        "num_epochs": int(vae_cfg.num_epochs),
        "learning_rate": float(vae_cfg.learning_rate),
        "weight_decay": float(vae_cfg.weight_decay),
        "lr_scheduler": str(vae_cfg.lr_scheduler),
        "lr_warmup_steps": int(vae_cfg.lr_warmup_steps),
        "min_lr": float(getattr(vae_cfg, "min_lr", 0.0)),
        "max_voxels_per_mesh": int(vae_cfg.max_voxels_per_mesh),
        "max_points_per_batch": int(getattr(vae_cfg, "max_points_per_batch", 0)),
        "rho_loss_weight": float(getattr(vae_cfg, "rho_loss_weight", 0.0)),
        "rho_warmup_epochs": int(getattr(vae_cfg, "rho_warmup_epochs", 0)),
        "use_stage2_render_loss": bool(getattr(vae_cfg, "use_stage2_render_loss", False)),
        "stage2_render_start_epoch": int(getattr(vae_cfg, "stage2_render_start_epoch", 0)),
        "lmdb_only": bool(lmdb_only),
    }
    return make_signature_record(details)


# ============================================================
# Training Loop
# ============================================================
class ChunkedRandomSampler(Sampler):
    """
    Đọc nối tiếp theo cụm (Sequential Read) để cứu ổ HDD, 
    sau đó trộn ngẫu nhiên bên trong cụm (In-RAM Shuffle).
    Giúp tránh hiện tượng Random Seek Death trên ổ đĩa cơ 18TB.
    """
    def __init__(self, data_source, chunk_size=500):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_samples = len(data_source)

    def __iter__(self):
        # Chia dataset thành các chunk (ví dụ: mỗi 500 mẫu liên tiếp trên đĩa)
        indices = list(range(self.num_samples))
        chunks = [indices[i : i + self.chunk_size] for i in range(0, self.num_samples, self.chunk_size)]
        
        # 1. Trộn thứ tự các chunk (Trộn mức vĩ mô)
        random.shuffle(chunks)
        
        for chunk in chunks:
            # 2. Trộn thứ tự bên trong 1 chunk (Trộn mức vi mô trong RAM)
            random.shuffle(chunk)
            for idx in chunk:
                yield idx

    def __len__(self):
        return self.num_samples


def train_sc_vae(
    cfg: TrainConfig,
    faceverse_train_ids_file: str = "train_faceverse_ids.txt",
    faceverse_test_ids_file: str = "test_faceverse_ids.txt",
    facescape_train_ids_file: str = "train_facescape_ids.txt",
    facescape_test_ids_file: str = "test_facescape_ids.txt",
    gradient_accumulation_steps: int = 67,
    loader_split_points: int = 0,
    use_activation_checkpointing: bool = False,
    clear_cache_freq: int = 4,
    perf_log_every_steps: int = 0,
    disable_id_filters: bool = False,
    lmdb_only: bool = False,
    enable_torch_compile: bool = True,
    allow_unsafe_resume: bool = False,
):
    """Vòng lặp huấn luyện chính (Main training loop) cho SC-VAE."""
    
    device = torch.device(cfg.device)
    vae_cfg = cfg.sc_vae
    base_point_cap = int(getattr(vae_cfg, "max_points_per_batch", 0))
    stage2_point_cap = int(getattr(vae_cfg, "stage2_max_points_per_batch", 0))
    if stage2_point_cap <= 0 and base_point_cap > 0:
        # Dành ra khoảng trống (headroom) cho LPIPS/SSIM khi hàm loss kết xuất giai đoạn 2 (stage-2 render loss) được kích hoạt.
        stage2_point_cap = max(1, int(base_point_cap * 0.75))

    adaptive_point_cap = int(base_point_cap) if base_point_cap > 0 else 0
    oom_backoff_count = 0
    max_oom_backoffs = 6
    min_point_cap = int(getattr(vae_cfg, "min_points_per_batch", 100000))
    if min_point_cap <= 0:
        min_point_cap = 100000
    fallback_point_cap = int(max(vae_cfg.max_voxels_per_mesh, 1) * max(vae_cfg.batch_size, 1) * 0.6)
    if base_point_cap > 0:
        fallback_point_cap = min(base_point_cap, fallback_point_cap)
    fallback_point_cap = max(min_point_cap, fallback_point_cap)

    def _resolve_point_cap(stage2_active: bool) -> int:
        cap = adaptive_point_cap if adaptive_point_cap > 0 else 0
        if cap <= 0 and base_point_cap > 0:
            cap = base_point_cap
        if stage2_active and stage2_point_cap > 0:
            cap = stage2_point_cap if cap <= 0 else min(cap, stage2_point_cap)
        return int(max(0, cap))

    def _apply_oom_backoff(stage2_active: bool, observed_points: int, reason: str) -> int:
        nonlocal adaptive_point_cap, oom_backoff_count
        base = int(observed_points) if int(observed_points) > 0 else int(adaptive_point_cap or base_point_cap or 0)
        if base <= 0:
            base = fallback_point_cap
        new_cap = int(max(min_point_cap, int(base * 0.8)))
        if adaptive_point_cap > 0:
            new_cap = min(adaptive_point_cap, new_cap)
        if adaptive_point_cap > 0 and new_cap >= adaptive_point_cap:
            new_cap = int(max(min_point_cap, int(adaptive_point_cap * 0.8)))
        adaptive_point_cap = int(max(min_point_cap, new_cap))
        oom_backoff_count += 1
        print(
            f"  [OOM] Backoff {oom_backoff_count}/{max_oom_backoffs}: "
            f"point cap -> {adaptive_point_cap} (reason={reason})"
        )
        if oom_backoff_count >= max_oom_backoffs:
            raise torch.OutOfMemoryError(
                f"OOM backoff exhausted after {oom_backoff_count} reductions; last cap={adaptive_point_cap}"
            )
        return _resolve_point_cap(stage2_active)

    # Các nút điều chỉnh hiệu suất (Performance knobs) cho huấn luyện chia sẻ GPU.
    # TF32 tăng tốc matmul/conv trên các GPU Ada với tác động chất lượng không đáng kể đối với khối lượng công việc này.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    print("=" * 60)
    print("  FACEDIFF — STAGE 1: SC-VAE TRAINING")
    print("=" * 60)
    cfg.print_summary()

    # Yêu cầu bắt buộc (Hard requirement): spconv phải được cài đặt
    try:
        import spconv.pytorch as spconv
    except ImportError as e:
        raise RuntimeError(
            "SC-VAE training requires spconv, but spconv is unavailable. "
            "Install spconv-cu121 (or compatible build) and retry."
        ) from e
    
    # ---- WandB ----
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or f"sc_vae_{time.strftime('%m%d_%H%M')}",
            tags=cfg.wandb.tags + ["sc_vae"],
            config={
                "stage": "sc_vae",
                "batch_size": vae_cfg.batch_size,
                "lr": vae_cfg.learning_rate,
                "epochs": vae_cfg.num_epochs,
                "latent_dim": vae_cfg.latent_dim,
                "kl_weight": vae_cfg.kl_weight,
            }
        )
    
    # ---- Dataset ----
    print("\n[1/4] Loading Dataset...")
    datasets_to_concat = []

    if disable_id_filters:
        faceverse_include_ids = None
        faceverse_exclude_ids = None
        facescape_include_ids = None
        facescape_exclude_ids = None
        print("  [Split] Disabled identity filtering for custom dataset run.")
    else:
        faceverse_include_ids = load_identity_set(faceverse_train_ids_file)
        faceverse_exclude_ids = load_identity_set(faceverse_test_ids_file)
        facescape_include_ids = load_identity_set(facescape_train_ids_file)
        facescape_exclude_ids = load_identity_set(facescape_test_ids_file)
    
    if cfg.data.active_dataset in ["faceverse", "both"] and os.path.isdir(cfg.data.faceverse_root):
        fv_dataset = VoxelDataset(
            data_root=cfg.data.faceverse_root,
            dataset_name="faceverse",
            max_voxels=vae_cfg.max_voxels_per_mesh,
            cache_dir=os.path.join("data", "ovoxel_cache_recached", "faceverse"),
            use_ovoxel_converter=vae_cfg.use_ovoxel_converter,
            ovoxel_resolution=vae_cfg.ovoxel_resolution,
            require_ovoxel_converter=vae_cfg.require_ovoxel_converter,
            target_in_channels=vae_cfg.in_channels,
            feature_mode=vae_cfg.input_feature_mode,
            include_ids=faceverse_include_ids,
            exclude_ids=faceverse_exclude_ids,
            lmdb_dir=cfg.data.lmdb_dir,
            lmdb_readahead=cfg.data.lmdb_readahead,
            lmdb_only=bool(lmdb_only),
            device=cfg.device,
        )
        if len(fv_dataset) > 0:
            datasets_to_concat.append(fv_dataset)
            
    if cfg.data.active_dataset in ["facescape", "both"] and os.path.isdir(cfg.data.facescape_root):
        fs_dataset = VoxelDataset(
            data_root=cfg.data.facescape_root,
            dataset_name="facescape",
            max_voxels=vae_cfg.max_voxels_per_mesh,
            cache_dir=os.path.join("data", "ovoxel_cache_recached", "facescape"),
            use_ovoxel_converter=vae_cfg.use_ovoxel_converter,
            ovoxel_resolution=vae_cfg.ovoxel_resolution,
            require_ovoxel_converter=vae_cfg.require_ovoxel_converter,
            target_in_channels=vae_cfg.in_channels,
            feature_mode=vae_cfg.input_feature_mode,
            include_ids=facescape_include_ids,
            exclude_ids=facescape_exclude_ids,
            lmdb_dir=cfg.data.lmdb_dir,
            lmdb_readahead=cfg.data.lmdb_readahead,
            lmdb_only=bool(lmdb_only),
            device=cfg.device,
        )
        if len(fs_dataset) > 0:
            datasets_to_concat.append(fs_dataset)
            
    if not datasets_to_concat:
        raise ValueError(f"Không tìm thấy Mesh nào cho dataset {cfg.data.active_dataset}!")
        
    if len(datasets_to_concat) > 1:
        dataset = ConcatDataset(datasets_to_concat)
    else:
        dataset = datasets_to_concat[0]
    
    val_dataloader = None
    train_dataset = dataset
    val_count = int(len(dataset) * max(float(vae_cfg.val_split), 0.0))
    val_count = min(max(val_count, 0), max(len(dataset) - 1, 0))
    if val_count > 0:
        train_count = len(dataset) - val_count
        split_gen = torch.Generator().manual_seed(cfg.seed)
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=split_gen)
    else:
        val_dataset = None

    num_workers = max(int(cfg.data.num_workers), 0)
    dataloader_timeout = max(int(getattr(cfg.data, "dataloader_timeout", 0)), 0)
    if num_workers == 0:
        dataloader_timeout = 0

    # LƯU Ý: SC-VAE hiện sử dụng một hàm collate tối ưu hóa có chức năng nối (concatenates)
    # các tensor trong các tiến trình worker. Việc bật pin_memory=True ở đây cho phép
    # PyTorch song song hóa việc ghim (pinning) các tensor lớn này, giải phóng
    # luồng chính (main thread) để chỉ tập trung hoàn toàn vào các hoạt động GPU.
    effective_pin_memory = bool(cfg.data.pin_memory)
    train_loader_common = {
        "num_workers": num_workers,
        "pin_memory": effective_pin_memory,
        "timeout": dataloader_timeout,
        "collate_fn": functools.partial(
            collate_voxels, 
            split_points=max(0, int(loader_split_points)),
            spatial_size=cfg.data.voxel_resolution,
            max_points_per_batch=int(vae_cfg.max_points_per_batch)
        ),
    }
    val_loader_common = {
        "num_workers": num_workers,
        "pin_memory": effective_pin_memory,
        "timeout": dataloader_timeout,
        "collate_fn": functools.partial(
            collate_voxels,
            spatial_size=cfg.data.voxel_resolution,
            max_points_per_batch=int(vae_cfg.max_points_per_batch)
        ),
    }
    if num_workers > 0:
        prefetch = max(int(cfg.data.prefetch_factor), 1)
        persistent = bool(cfg.data.persistent_workers)
        train_loader_common["prefetch_factor"] = prefetch
        train_loader_common["persistent_workers"] = persistent
        val_loader_common["prefetch_factor"] = prefetch
        val_loader_common["persistent_workers"] = persistent

    # Sử dụng ChunkedRandomSampler để tối ưu IOPS cho HDD (User recommendation)
    chunk_sampler = ChunkedRandomSampler(train_dataset, chunk_size=500)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=vae_cfg.batch_size,
        sampler=chunk_sampler,
        drop_last=True,
        **train_loader_common,
    )
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=vae_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **val_loader_common,
        )

    print(f"  Dataset: {len(dataset)} meshes | train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0}")
    print(f"  Train batches/epoch: {len(dataloader)}")
    if dataloader_timeout > 0:
        print(f"  DataLoader timeout: {dataloader_timeout}s")
    if cfg.data.lmdb_dir:
        print(f"  LMDB mode: {'lmdb-only (no disk cache probes)' if bool(lmdb_only) else 'lmdb+disk-fallback'}")
    if int(max(0, loader_split_points)) > 0:
        print(f"  Load-balanced micro-batch split: {int(loader_split_points)} points/pack")
    if val_dataloader is not None:
        print(f"  Val batches/epoch: {len(val_dataloader)}")
    if stage2_point_cap > 0 and base_point_cap > 0 and stage2_point_cap < base_point_cap:
        print(f"  Stage-2 point cap reserve: {stage2_point_cap} (base cap {base_point_cap})")
    
    # ---- Model ----
    print("\n[2/4] Building SC-VAE...")
    encoder_dims = getattr(vae_cfg, "encoder_dims", [64, 128, 256, 512])
    num_res_blocks = getattr(vae_cfg, "num_res_blocks", 2)
    model = SC_VAE(
        in_channels=vae_cfg.in_channels,
        latent_dim=vae_cfg.latent_dim,
        device=cfg.device,
        rho_prune_threshold=float(getattr(vae_cfg, "rho_prune_threshold", 0.5)),
        encoder_dims=encoder_dims,
        num_res_blocks=num_res_blocks,
    ).to(device)
    print(f"  Architecture: {vae_cfg.in_channels} -> {encoder_dims} -> {vae_cfg.latent_dim}d latent")
    print(f"  Res blocks per level: {num_res_blocks}")
    
    loss_fn = SCVAELoss(
        kl_weight=vae_cfg.kl_weight,
        use_bce_for_geom=vae_cfg.use_bce_for_geom,
        rho_loss_weight=vae_cfg.rho_loss_weight,
    )
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # ---- Optimizer ----
    # Thử nghiệm Fused AdamW nếu trên CUDA (yêu cầu torch >= 2.0)
    optimizer_kwargs = {
        "lr": vae_cfg.learning_rate,
        "weight_decay": vae_cfg.weight_decay,
        "betas": (0.9, 0.999),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    steps_per_epoch_estimate = max(len(dataloader) // max(gradient_accumulation_steps, 1), 1)
    scheduler = get_lr_scheduler(optimizer, vae_cfg, steps_per_epoch=steps_per_epoch_estimate)
    print(f"  LR scheduler: cosine over {steps_per_epoch_estimate * vae_cfg.num_epochs} total steps ({steps_per_epoch_estimate}/epoch)")

    data_signature = _build_data_signature(
        cfg,
        vae_cfg,
        dataset,
        train_dataset,
        val_dataset,
        faceverse_train_ids_file=faceverse_train_ids_file,
        faceverse_test_ids_file=faceverse_test_ids_file,
        facescape_train_ids_file=facescape_train_ids_file,
        facescape_test_ids_file=facescape_test_ids_file,
        lmdb_only=bool(lmdb_only),
    )
    resume_contract = _build_resume_contract(
        vae_cfg,
        steps_per_epoch_estimate=steps_per_epoch_estimate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loader_split_points=loader_split_points,
        lmdb_only=bool(lmdb_only),
    )
    checkpoint_metadata = {
        "data_signature": data_signature,
        "resume_contract": resume_contract,
    }
    
    # spconv v2.x không hỗ trợ bfloat16 trong một số ops (KeyError), 
    # nên chúng ta dùng float16. Các bộ kẹp (Clamps) đã thêm ở bước trước 
    # sẽ bảo vệ float16 khỏi bị tràn số (overflow).
    amp_dtype = torch.float16
    print(f"  AMP: Using {amp_dtype} for spconv compatibility.")
    print(
        f"  Resume metadata: data={data_signature['digest'][:12]} "
        f"contract={resume_contract['digest'][:12]}"
    )
    
    # Độ chính xác hỗn hợp (Mixed precision)
    scaler = torch.amp.GradScaler('cuda', enabled=vae_cfg.use_amp)
    if device.type == 'cuda' and vae_cfg.use_amp:
        print(f"  AMP autocast dtype: {amp_dtype}")
    
    # ---- Resume ----
    start_epoch = 0
    best_metric = float('inf')
    global_step = 0
    last_loss = float('inf')
    last_loss_tensor = None
    epoch_time_hist = []
    
    if vae_cfg.resume_from and os.path.exists(vae_cfg.resume_from):
        try:
            ckpt = load_checkpoint(
                vae_cfg.resume_from,
                model,
                optimizer,
                scheduler,
                scaler,
                load_optimizer=not vae_cfg.resume_model_only,
                strict_model_load=True,
                expected_data_signature=data_signature,
                expected_resume_contract=resume_contract,
                allow_unsafe_resume=bool(allow_unsafe_resume),
            )
        except RuntimeError as exc:
            if vae_cfg.resume_model_only:
                print("  ⚠️ Strict resume failed in model-only mode. Retrying with partial load...")
                ckpt = load_checkpoint(
                    vae_cfg.resume_from,
                    model,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    load_optimizer=False,
                    strict_model_load=False,
                    expected_data_signature=data_signature,
                    expected_resume_contract=resume_contract,
                    allow_unsafe_resume=bool(allow_unsafe_resume),
                )
            else:
                raise exc
        start_epoch = int(ckpt.get('epoch', 0))
        global_step = int(ckpt.get('global_step', start_epoch * len(dataloader)) or 0)
        ckpt_batch_size = ckpt.get('batch_size', None)
        if ckpt_batch_size is not None and ckpt_batch_size != vae_cfg.batch_size and not vae_cfg.resume_model_only:
            print(
                "  ⚠️ Batch size changed from checkpoint "
                f"({ckpt_batch_size} -> {vae_cfg.batch_size}). "
                "If training becomes unstable, resume with --resume-model-only."
            )

        # Optional: rebuild scheduler for fine-tune extension past the original cosine end.
        # Triggered by --resume-scheduler-mode {cosine_restart, constant_min_lr}.
        resume_sched_mode = getattr(vae_cfg, "resume_scheduler_mode", None) or "continue"
        if resume_sched_mode != "continue":
            try:
                ckpt_sched_state = ckpt.get('scheduler_state_dict', {}) or {}
                resume_step = int(ckpt_sched_state.get('_step_count', 0)) or max(
                    int(global_step) // max(int(gradient_accumulation_steps), 1), 0
                )
                extend_epochs = int(getattr(vae_cfg, "resume_extend_epochs", 100))
                target_min_lr = float(getattr(vae_cfg, "resume_target_min_lr", vae_cfg.min_lr))
                scheduler = get_resume_scheduler(
                    optimizer,
                    vae_cfg,
                    steps_per_epoch=steps_per_epoch_estimate,
                    resume_step=resume_step,
                    mode=resume_sched_mode,
                    extend_epochs=extend_epochs,
                    target_min_lr=target_min_lr,
                )
                # Pull the just-rebuilt scheduler's LR into the optimizer so the next step
                # uses the new schedule immediately (no jump back to base_lr).
                cur_scale = scheduler.get_last_lr()[0] / max(float(vae_cfg.learning_rate), 1e-12)
                for pg in optimizer.param_groups:
                    pg['lr'] = float(vae_cfg.learning_rate) * cur_scale
                print(
                    f"  🔄 Resume scheduler rebuilt: mode={resume_sched_mode} "
                    f"extend={extend_epochs}ep target_min_lr={target_min_lr:.1e} "
                    f"start_lr={cur_scale * float(vae_cfg.learning_rate):.3e}"
                )
            except Exception as exc:
                print(f"  ⚠️ Failed to rebuild resume scheduler ({exc!r}); keeping the loaded one.")
    
    # ---- Compilation (RTX 4090 Optimization) ----
    # LƯU Ý: spconv + torch.compile + activation checkpointing có thể gây ra
    # mã sinh (codegen) Triton/Inductor không ổn định trên một số lô thưa thớt (sparse batches).
    if device.type == "cuda" and hasattr(torch, "compile") and bool(enable_torch_compile):
        if use_activation_checkpointing:
            print(
                "\n[3.5/4] Skipping torch.compile because activation checkpointing "
                "with sparse spconv kernels can be unstable on Inductor/Triton."
            )
        else:
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.suppress_errors = True
            except Exception:
                pass
            print("\n[3.5/4] Compiling model with torch.compile (reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
    elif not bool(enable_torch_compile):
        print("\n[3.5/4] torch.compile disabled by CLI flag (--no-torch-compile).")

    # ---- Training ----
    print(f"\n[4/4] Training for {vae_cfg.num_epochs} epochs...")
    os.makedirs(vae_cfg.checkpoint_dir, exist_ok=True)
    if global_step == 0:
        steps_per_epoch_est = max(len(dataloader) // max(gradient_accumulation_steps, 1), 1)
        global_step = start_epoch * steps_per_epoch_est

    current_epoch = start_epoch
    interrupted = False
    try:
        # [TRELLIS.2] Initialize Adaptive Grad Clipper history
        grad_norm_history = collections.deque(maxlen=1000)

        for epoch in range(start_epoch, vae_cfg.num_epochs):
            current_epoch = epoch
            model.train()
            epoch_loss = torch.zeros((), device=device, dtype=torch.float32)
            epoch_recon = torch.zeros((), device=device, dtype=torch.float32)
            epoch_kl = torch.zeros((), device=device, dtype=torch.float32)
            epoch_rho = torch.zeros((), device=device, dtype=torch.float32)
            epoch_oom_skips = 0
            epoch_sparse_skips = 0
            epoch_nan_skips = 0
            consecutive_oom_skips = 0
            max_consecutive_oom_skips = 4
            epoch_points_before = 0
            epoch_points_after = 0
            epoch_capped_batches = 0
            epoch_data_time = 0.0
            epoch_forward_time = 0.0
            epoch_backward_time = 0.0
            epoch_step_time = 0.0
            epoch_cap_time = 0.0
            epoch_sparse_build_time = 0.0
            epoch_model_forward_time = 0.0
            epoch_stage2_time = 0.0
            epoch_stage2_skips = 0
            epoch_stage2_skip_logs = 0
            stage2_active = (
                bool(vae_cfg.use_stage2_render_loss)
                and epoch >= int(vae_cfg.stage2_render_start_epoch)
                and stage2_point_cap > 0
            )
            epoch_point_cap_limit = _resolve_point_cap(stage2_active)

            # KL Annealing: tăng dần trọng số KL qua kl_warmup_epochs
            if epoch < vae_cfg.kl_warmup_epochs:
                kl_scale = epoch / max(vae_cfg.kl_warmup_epochs, 1)
            else:
                kl_scale = 1.0

            if epoch < int(getattr(vae_cfg, "rho_warmup_epochs", 0)):
                rho_scale = epoch / max(int(getattr(vae_cfg, "rho_warmup_epochs", 1)), 1)
            else:
                rho_scale = 1.0

            t_start = time.time()
            batch_fetch_t0 = time.perf_counter()

            for batch_idx, batch_payload in enumerate(dataloader):
                data_time = time.perf_counter() - batch_fetch_t0

                if isinstance(batch_payload, dict) and batch_payload.get("is_pre_concatenated"):
                    micro_batches = [batch_payload]
                elif _is_packed_micro_batch(batch_payload):
                    micro_batches = [b for b in batch_payload if isinstance(b, list) and len(b) > 0]
                else:
                    micro_batches = [batch_payload] if isinstance(batch_payload, list) and len(batch_payload) > 0 else []

                if len(micro_batches) == 0:
                    batch_fetch_t0 = time.perf_counter()
                    continue

                micro_count = len(micro_batches)

                # Tích lũy gradient (Gradient accumulation): zero gradients và reset stats chỉ ở đầu chu kỳ tích lũy.
                acc_step = batch_idx % gradient_accumulation_steps
                if acc_step == 0:
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                    batch_data_time = 0.0
                    batch_recon_sum = torch.zeros((), device=device, dtype=torch.float32)
                    batch_kl_sum = torch.zeros((), device=device, dtype=torch.float32)
                    batch_rho_sum = torch.zeros((), device=device, dtype=torch.float32)
                    
                    batch_forward_time = 0.0
                    batch_backward_time = 0.0
                    batch_cap_time = 0.0
                    batch_sparse_build_time = 0.0
                    batch_model_forward_time = 0.0
                    batch_stage2_time = 0.0
                    
                    stage2_render_last = torch.zeros((), device=device, dtype=torch.float32)
                    stage2_perceptual_last = torch.zeros((), device=device, dtype=torch.float32)
                    stage2_total_last = torch.zeros((), device=device, dtype=torch.float32)

                micro_failed = False
                micro_fail_stage = ""
                micro_fail_exc = None
                micro_is_sparse_recoverable = False
                micro_points_after = 0

                for micro_items in micro_batches:
                    fw_t0 = time.perf_counter()
                    cap_t0 = time.perf_counter()

                    # Xử lý các mục đã được nối sẵn (pre-concatenated) từ collate tối ưu hóa
                    is_pre_cat = isinstance(micro_items, dict) and micro_items.get("is_pre_concatenated")
                    
                    if is_pre_cat:
                        pts_before = int(micro_items["feats_cat"].shape[0])
                        pts_after = pts_before
                        render_items = None
                        if epoch_point_cap_limit > 0 and pts_before > epoch_point_cap_limit:
                            render_items = materialize_batch_items(micro_items)
                            render_items, pts_before, pts_after = cap_points_per_batch(
                                render_items,
                                epoch_point_cap_limit,
                            )
                            micro_items = render_items
                            is_pre_cat = False
                    else:
                        micro_items, pts_before, pts_after = cap_points_per_batch(
                            micro_items,
                            epoch_point_cap_limit,
                        )
                        render_items = micro_items

                    cap_time = time.perf_counter() - cap_t0
                    batch_cap_time += cap_time
                    epoch_points_before += pts_before
                    epoch_points_after += pts_after
                    micro_points_after = pts_after
                    if pts_after < pts_before:
                        epoch_capped_batches += 1

                    sparse_build_t0 = time.perf_counter()
                    try:
                        voxels_sparse, target_feats = build_sparse_batch(
                            micro_items,
                            device=device,
                            spatial_size=cfg.data.voxel_resolution,
                        )
                    except RuntimeError as exc:
                        if _is_oom_error(exc):
                            micro_failed = True
                            micro_fail_stage = "sparse-build"
                            micro_fail_exc = exc
                            micro_is_sparse_recoverable = False
                            break
                        if _is_sparse_runtime_error(exc):
                            micro_failed = True
                            micro_fail_stage = "sparse-build"
                            micro_fail_exc = exc
                            micro_is_sparse_recoverable = True
                            break
                        raise
                    model_input = voxels_sparse
                    target_x = target_feats
                    sparse_build_time = time.perf_counter() - sparse_build_t0
                    batch_sparse_build_time += sparse_build_time

                    model_forward_time = 0.0
                    stage2_time = 0.0

                    try:
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=vae_cfg.use_amp):
                            model_t0 = time.perf_counter()
                            if use_activation_checkpointing:
                                recon, mu, logvar, rho_logits_list, rho_targets_list, out_indices = checkpoint(
                                    model, model_input, use_reentrant=False
                                )
                            else:
                                recon, mu, logvar, rho_logits_list, rho_targets_list, out_indices = model(model_input)
                            model_forward_time = time.perf_counter() - model_t0

                            recon_aligned, target_aligned, was_mismatch = align_recon_target(
                                recon, out_indices, target_x, model_input.indices
                            )
                            if was_mismatch and batch_idx % 20 == 0:
                                print(
                                    "  [WARN] Sparse recon/target size mismatch in train micro-batch "
                                    f"(recon={recon.shape[0]}, target={target_x.shape[0]}). "
                                    f"Using aligned length={recon_aligned.shape[0]}."
                                )

                            loss_dict = loss_fn(
                                recon_aligned,
                                target_aligned,
                                mu,
                                logvar,
                                feature_mode=vae_cfg.input_feature_mode,
                                rho_logits_list=rho_logits_list,
                                rho_targets_list=rho_targets_list,
                            )

                            stage2_render = target_x.new_zeros(())
                            stage2_perceptual = target_x.new_zeros(())
                            stage2_total = target_x.new_zeros(())
                            if (
                                bool(vae_cfg.use_stage2_render_loss)
                                and epoch >= int(vae_cfg.stage2_render_start_epoch)
                                and not was_mismatch
                                and batch_idx % 2 == 0  # [PERF] Chạy mỗi 2 bước, tiết kiệm ~4%
                            ):
                                # Tái tạo render_items từ dữ liệu đã nối sẵn (pre-cat data) nếu cần
                                if render_items is None and is_pre_cat:
                                    render_items = materialize_batch_items(micro_items)
                                
                                if render_items:
                                    try:
                                        stage2_t0 = time.perf_counter()
                                        # Hàm loss nhận thức/kết xuất giai đoạn 2 (Stage2 render/perceptual loss) sử dụng LPIPS và SSIM,
                                        # chúng không ổn định về mặt số học dưới dạng fp16 autocast. Buộc sử dụng float32.
                                        with torch.amp.autocast('cuda', enabled=False):
                                            stage2_dict = compute_stage2_render_perceptual_loss(
                                                recon_aligned.float(),
                                                target_aligned.float(),
                                                render_items,
                                                feature_mode=vae_cfg.input_feature_mode,
                                                image_size=int(vae_cfg.stage2_render_image_size),
                                                num_views=int(vae_cfg.stage2_render_views),
                                            )
                                        stage2_time = time.perf_counter() - stage2_t0
                                        stage2_render = stage2_dict["render_loss"]
                                        stage2_perceptual = stage2_dict["perceptual_loss"]
                                        stage2_total = float(vae_cfg.stage2_render_weight) * (
                                            stage2_render + float(vae_cfg.stage2_perceptual_weight) * stage2_perceptual
                                        )
                                        # Ngăn chặn NaN từ hàm loss kết xuất lan truyền tới bộ tối ưu hóa (optimizer)
                                        if torch.isnan(stage2_total) or torch.isinf(stage2_total):
                                            stage2_total = target_x.new_zeros(())
                                            stage2_render = target_x.new_zeros(())
                                            stage2_perceptual = target_x.new_zeros(())
                                    except Exception as exc:
                                        epoch_stage2_skips += 1
                                        if epoch_stage2_skip_logs < 5 or (epoch_stage2_skips % 50) == 0:
                                            print(
                                                f"  [WARN] Stage-2 render loss skipped at epoch {epoch+1}, "
                                                f"batch {batch_idx+1}: {exc}"
                                            )
                                            epoch_stage2_skip_logs += 1
                                        if device.type == "cuda" and _is_oom_error(exc):
                                            torch.cuda.empty_cache()
                                        stage2_time = 0.0
                                        stage2_render = target_x.new_zeros(())
                                        stage2_perceptual = target_x.new_zeros(())
                                        stage2_total = target_x.new_zeros(())

                            loss = (
                                loss_dict['recon_loss']
                                + kl_scale * vae_cfg.kl_weight * loss_dict['kl_loss']
                                + rho_scale * float(getattr(vae_cfg, "rho_loss_weight", 0.0)) * loss_dict.get('rho_loss', target_x.new_zeros(()))
                                + stage2_total
                            )
                            if not torch.isfinite(loss):
                                micro_failed = True
                                micro_fail_stage = "non-finite"
                                micro_fail_exc = RuntimeError("Non-finite loss detected")
                                micro_is_sparse_recoverable = False
                                break
                            loss_normalized = loss / float(max(gradient_accumulation_steps * micro_count, 1))
                    except RuntimeError as exc:
                        if not _is_oom_error(exc) and not _is_sparse_runtime_error(exc):
                            raise
                        micro_failed = True
                        micro_fail_stage = "forward/loss"
                        micro_fail_exc = exc
                        micro_is_sparse_recoverable = _is_sparse_runtime_error(exc)
                        break

                    batch_model_forward_time += model_forward_time
                    batch_stage2_time += stage2_time
                    batch_forward_time += (time.perf_counter() - fw_t0)

                    bw_t0 = time.perf_counter()
                    try:
                        scaler.scale(loss_normalized).backward()
                    except RuntimeError as exc:
                        if not _is_oom_error(exc) and not _is_sparse_runtime_error(exc):
                            raise
                        micro_failed = True
                        micro_fail_stage = "backward"
                        micro_fail_exc = exc
                        micro_is_sparse_recoverable = _is_sparse_runtime_error(exc)
                        break
                    batch_backward_time += (time.perf_counter() - bw_t0)

                    loss_detached = loss.detach()
                    batch_loss_sum += loss_detached
                    batch_recon_sum += loss_dict['recon_loss'].detach()
                    batch_kl_sum += loss_dict['kl_loss'].detach()
                    batch_rho_sum += loss_dict.get('rho_loss', target_x.new_zeros(())).detach()
                    stage2_render_last = stage2_render.detach()
                    stage2_perceptual_last = stage2_perceptual.detach()
                    stage2_total_last = stage2_total.detach()

                if micro_failed:
                    if micro_is_sparse_recoverable:
                        epoch_sparse_skips += 1
                        consecutive_oom_skips = 0
                        print(
                            f"  [SPARSE] Skipping train batch {batch_idx+1}/{len(dataloader)} "
                            f"(epoch {epoch+1}) during {micro_fail_stage}: {micro_fail_exc}"
                        )
                    elif micro_fail_stage == "non-finite":
                        epoch_nan_skips += 1
                        consecutive_oom_skips = 0
                        print(
                            f"  [NAN] Skipping train batch {batch_idx+1}/{len(dataloader)} "
                            f"(epoch {epoch+1}) during {micro_fail_stage}: {micro_fail_exc}"
                        )
                    else:
                        epoch_oom_skips += 1
                        consecutive_oom_skips += 1
                        print(
                            f"  [OOM] Skipping train batch {batch_idx+1}/{len(dataloader)} "
                            f"(epoch {epoch+1}) during {micro_fail_stage}: {micro_fail_exc}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if (not micro_is_sparse_recoverable) and consecutive_oom_skips >= max_consecutive_oom_skips:
                        epoch_point_cap_limit = _apply_oom_backoff(
                            stage2_active,
                            micro_points_after,
                            reason="micro-batch",
                        )
                        consecutive_oom_skips = 0
                        batch_fetch_t0 = time.perf_counter()
                        continue
                    batch_fetch_t0 = time.perf_counter()
                    continue

                # Kiểm tra xem đây có phải là bước tích lũy cuối cùng hay không.
                is_last_acc_step = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader))

                # Chỉ cập nhật bộ tối ưu hóa (optimizer) ở bước tích lũy cuối cùng.
                if is_last_acc_step:
                    step_t0 = time.perf_counter()
                    if getattr(vae_cfg, 'use_adaptive_clip', False):
                        scaler.unscale_(optimizer)
                        
                        # Tính tổng norm của tất cả các gradient
                        parameters = [p for p in model.parameters() if p.grad is not None]
                        
                        if len(parameters) > 0:
                            # ÉP KIỂU SANG FLOAT32 để tránh tràn số khi bình phương (norm) đối với bfloat16
                            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach().to(torch.float32), 2) for p in parameters]), 2).item()
                            
                            # [BẢO VỆ CHỐNG NAN]: Kiểm tra Gradient thủ công
                            if math.isnan(total_norm) or math.isinf(total_norm):
                                print(f"  [WARN] NaN/Inf detected in gradients at batch {batch_idx+1}, skipping step.")
                                optimizer.zero_grad(set_to_none=True)
                                scaler.update()
                                continue
                                
                            # Cắt Gradient thích ứng (Adaptive Clipping)
                            if len(grad_norm_history) > 100:
                                percentile_95 = np.percentile(list(grad_norm_history), getattr(vae_cfg, 'adaptive_clip_percentile', 95.0))
                                clip_target = percentile_95 * getattr(vae_cfg, 'adaptive_clip_max_norm', 1.0)
                                clip_coef = clip_target / (total_norm + 1e-6)
                                
                                if clip_coef < 1.0:
                                    for p in parameters:
                                        p.grad.detach().mul_(clip_coef)
                                        
                            grad_norm_history.append(total_norm)
                            
                    elif vae_cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        
                        # [BẢO VỆ CHỐNG NAN] Dành cho Fallback
                        found_inf_nan = False
                        for p in model.parameters():
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                found_inf_nan = True
                                break
                        
                        if found_inf_nan:
                            print(f"  [WARN] NaN/Inf detected in gradients at batch {batch_idx+1}, skipping step.")
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            continue
                            
                        torch.nn.utils.clip_grad_norm_(model.parameters(), vae_cfg.grad_clip)

                    prev_scale = scaler.get_scale()
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as exc:
                        if not _is_oom_error(exc):
                            raise
                        epoch_oom_skips += 1
                        consecutive_oom_skips += 1
                        print(
                            f"  [OOM] Skipping optimizer step at batch {batch_idx+1}/{len(dataloader)} "
                            f"(epoch {epoch+1}): {exc}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        if consecutive_oom_skips >= max_consecutive_oom_skips:
                            epoch_point_cap_limit = _apply_oom_backoff(
                                stage2_active,
                                epoch_point_cap_limit,
                                reason="optimizer-step",
                            )
                            consecutive_oom_skips = 0
                            batch_fetch_t0 = time.perf_counter()
                            continue
                        batch_fetch_t0 = time.perf_counter()
                        continue
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                    step_time = time.perf_counter() - step_t0
                else:
                    step_time = 0.0

                if is_last_acc_step:
                    consecutive_oom_skips = 0

                    batch_loss_avg = batch_loss_sum / float(max(gradient_accumulation_steps * micro_count, 1))
                    batch_recon_avg = batch_recon_sum / float(max(gradient_accumulation_steps * micro_count, 1))
                    batch_kl_avg = batch_kl_sum / float(max(gradient_accumulation_steps * micro_count, 1))
                    batch_rho_avg = batch_rho_sum / float(max(gradient_accumulation_steps * micro_count, 1))

                    epoch_loss += batch_loss_avg
                    epoch_recon += batch_recon_avg
                    epoch_kl += batch_kl_avg
                    epoch_rho += batch_rho_avg
                    epoch_data_time += batch_data_time
                    epoch_forward_time += batch_forward_time
                    epoch_backward_time += batch_backward_time
                    epoch_step_time += step_time
                    epoch_cap_time += batch_cap_time
                    epoch_sparse_build_time += batch_sparse_build_time
                    epoch_model_forward_time += batch_model_forward_time
                    epoch_stage2_time += batch_stage2_time
                    global_step += 1
                    last_loss_tensor = batch_loss_avg.detach()

                    if clear_cache_freq > 0 and ((batch_idx + 1) % clear_cache_freq == 0):
                        torch.cuda.empty_cache()

                    if vae_cfg.save_every_steps > 0 and global_step % vae_cfg.save_every_steps == 0:
                        if last_loss_tensor is not None:
                            last_loss = float(last_loss_tensor.item())
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            last_loss,
                            os.path.join(vae_cfg.checkpoint_dir, "latest_step.pt"),
                            global_step=global_step,
                            batch_size=vae_cfg.batch_size,
                            metadata=checkpoint_metadata,
                        )

                    if cfg.wandb.enabled and WANDB_AVAILABLE and global_step % cfg.wandb.log_every_steps == 0:
                        wandb.log({
                            "train/loss": float(batch_loss_avg.item()),
                            "train/recon_loss": float(batch_recon_avg.item()),
                            "train/kl_loss": float(batch_kl_avg.item()),
                            "train/rho_loss": float(batch_rho_avg.item()),
                            "train/stage2_render_loss": float(stage2_render_last.item()),
                            "train/stage2_perceptual_loss": float(stage2_perceptual_last.item()),
                            "train/stage2_total_loss": float(stage2_total_last.item()),
                            "train/kl_scale": kl_scale,
                            "train/rho_scale": rho_scale,
                            "perf/data_ms": batch_data_time * 1000.0,
                            "perf/forward_ms": batch_forward_time * 1000.0,
                            "perf/backward_ms": batch_backward_time * 1000.0,
                            "perf/step_ms": step_time * 1000.0,
                            "perf/cap_ms": batch_cap_time * 1000.0,
                            "perf/sparse_build_ms": batch_sparse_build_time * 1000.0,
                            "perf/model_forward_ms": batch_model_forward_time * 1000.0,
                            "perf/stage2_ms": batch_stage2_time * 1000.0,
                            "train/lr": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                        }, step=global_step)

                    if perf_log_every_steps > 0 and (global_step % perf_log_every_steps == 0):
                        print(
                            f"  [PERF] step={global_step} data={data_time*1000.0:.1f}ms "
                            f"fw={batch_forward_time*1000.0:.1f}ms bw={batch_backward_time*1000.0:.1f}ms "
                            f"opt={step_time*1000.0:.1f}ms "
                            f"(cap={batch_cap_time*1000.0:.1f}/sparse={batch_sparse_build_time*1000.0:.1f}/"
                            f"model={batch_model_forward_time*1000.0:.1f}/stage2={batch_stage2_time*1000.0:.1f})"
                        )
                    
                    # IN LOSS ĐỊNH KỲ (Mỗi 50 bước) để theo dõi độ hội tụ
                    if global_step % 50 == 0:
                        print(
                            f"  [TRAIN] Step {global_step} | Epoch {epoch+1} | "
                            f"Loss: {batch_loss_avg:.4f} (recon={batch_recon_avg:.4f}, kl={batch_kl_avg:.4f})"
                        )

                batch_fetch_t0 = time.perf_counter()

            # Thống kê Epoch (Epoch stats)
            n_batches = max(len(dataloader) // max(gradient_accumulation_steps, 1), 1)
            avg_loss = float((epoch_loss / n_batches).item())
            avg_recon = float((epoch_recon / n_batches).item())
            avg_kl = float((epoch_kl / n_batches).item())
            avg_rho = float((epoch_rho / n_batches).item())
            elapsed = time.time() - t_start
            epoch_time_hist.append(elapsed)
            avg_epoch_time = sum(epoch_time_hist) / len(epoch_time_hist)
            epochs_left = vae_cfg.num_epochs - (epoch + 1)
            eta_seconds = max(0.0, avg_epoch_time * epochs_left)
            eta_finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds))

            vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
            avg_data_ms = (epoch_data_time / n_batches) * 1000.0
            avg_forward_ms = (epoch_forward_time / n_batches) * 1000.0
            avg_backward_ms = (epoch_backward_time / n_batches) * 1000.0
            avg_step_ms = (epoch_step_time / n_batches) * 1000.0
            avg_cap_ms = (epoch_cap_time / n_batches) * 1000.0
            avg_sparse_build_ms = (epoch_sparse_build_time / n_batches) * 1000.0
            avg_model_forward_ms = (epoch_model_forward_time / n_batches) * 1000.0
            avg_stage2_ms = (epoch_stage2_time / n_batches) * 1000.0
            cap_ratio = (epoch_points_after / max(float(epoch_points_before), 1.0))

            print(f"  Epoch {epoch+1}/{vae_cfg.num_epochs} | "
                f"Loss: {avg_loss:.4f} (recon={avg_recon:.4f}, kl={avg_kl:.4f}, rho={avg_rho:.4f}) | "
                f"KL_scale: {kl_scale:.2f} | rho_scale: {rho_scale:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s | VRAM: {vram:.0f}MB | "
                f"Perf(ms): data={avg_data_ms:.1f}/fw={avg_forward_ms:.1f}/bw={avg_backward_ms:.1f}/opt={avg_step_ms:.1f} | "
                f"PerfDetail(ms): cap={avg_cap_ms:.1f}/sparse={avg_sparse_build_ms:.1f}/model={avg_model_forward_ms:.1f}/stage2={avg_stage2_ms:.1f} | "
                f"OOM_skips: {epoch_oom_skips} | sparse_skips: {epoch_sparse_skips} | nan_skips: {epoch_nan_skips} | stage2_skips: {epoch_stage2_skips} | "
                f"cap_batches: {epoch_capped_batches} ({cap_ratio*100.0:.1f}% pts kept) | "
                f"ETA: {eta_seconds/3600:.1f}h ({eta_finish})")

            # Ghi log epoch lên WandB
            if cfg.wandb.enabled and WANDB_AVAILABLE:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/recon_loss": avg_recon,
                    "epoch/kl_loss": avg_kl,
                    "epoch/rho_loss": avg_rho,
                    "epoch/vram_peak_mb": vram,
                    "epoch/perf_data_ms": avg_data_ms,
                    "epoch/perf_forward_ms": avg_forward_ms,
                    "epoch/perf_backward_ms": avg_backward_ms,
                    "epoch/perf_step_ms": avg_step_ms,
                    "epoch/perf_cap_ms": avg_cap_ms,
                    "epoch/perf_sparse_build_ms": avg_sparse_build_ms,
                    "epoch/perf_model_forward_ms": avg_model_forward_ms,
                    "epoch/perf_stage2_ms": avg_stage2_ms,
                    "epoch/oom_skipped_batches": epoch_oom_skips,
                    "epoch/sparse_skipped_batches": epoch_sparse_skips,
                    "epoch/nan_skipped_batches": epoch_nan_skips,
                    "epoch/stage2_skipped_batches": epoch_stage2_skips,
                    "epoch/capped_batches": epoch_capped_batches,
                    "epoch/points_keep_ratio": cap_ratio,
                }, step=global_step)

            val_avg_loss = None
            if val_dataloader is not None and ((epoch + 1) % max(int(vae_cfg.val_every_epochs), 1) == 0):
                model.eval()
                val_loss_sum = 0.0
                val_cd_sum = 0.0
                val_iou_sum = 0.0
                val_nc_sum = 0.0
                val_batches = 0
                val_oom_skips = 0
                val_stage2_skips = 0

                # Lấy tập con mẫu để trực quan hóa với độ trung thực cao (high-fidelity visualization)
                viz_items = None
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        if isinstance(val_batch, dict) and val_batch.get("is_pre_concatenated"):
                            val_micro_batches = [val_batch]
                        elif _is_packed_micro_batch(val_batch):
                            val_micro_batches = [
                                batch for batch in val_batch
                                if isinstance(batch, list) and len(batch) > 0
                            ]
                        else:
                            val_micro_batches = [val_batch] if isinstance(val_batch, list) and len(val_batch) > 0 else []

                        for val_payload in val_micro_batches:
                            is_pre_cat = isinstance(val_payload, dict) and val_payload.get("is_pre_concatenated")
                            if is_pre_cat:
                                total_val_points = int(val_payload["feats_cat"].shape[0])
                                if total_val_points <= 0:
                                    continue
                                val_items = None
                                sparse_payload = val_payload
                                if epoch_point_cap_limit > 0 and total_val_points > epoch_point_cap_limit:
                                    val_items = materialize_batch_items(val_payload)
                                    val_items, _, _ = cap_points_per_batch(
                                        val_items,
                                        epoch_point_cap_limit,
                                    )
                                    total_val_points = sum(
                                        int(item["features"].shape[0])
                                        for item in val_items
                                        if isinstance(item, dict) and "features" in item
                                    )
                                    if total_val_points <= 0:
                                        continue
                                    sparse_payload = val_items
                                    is_pre_cat = False
                            else:
                                val_items = materialize_batch_items(val_payload)
                                val_items, _, _ = cap_points_per_batch(
                                    val_items,
                                    epoch_point_cap_limit,
                                )
                                if not isinstance(val_items, list) or len(val_items) == 0:
                                    continue
                                total_val_points = 0
                                for val_item in val_items:
                                    if isinstance(val_item, dict) and "features" in val_item:
                                        total_val_points += int(val_item["features"].shape[0])
                                if total_val_points <= 0:
                                    continue
                                sparse_payload = val_items

                            try:
                                val_sparse, val_target = build_sparse_batch(
                                    sparse_payload,
                                    device=device,
                                    spatial_size=cfg.data.voxel_resolution,
                                )
                                val_input = val_sparse

                                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=vae_cfg.use_amp):
                                    val_recon, val_mu, val_logvar, val_rho_logits, val_rho_targets, val_out_indices = model(val_input)
                                    val_recon_aligned, val_target_aligned, val_mismatch = align_recon_target(
                                        val_recon, val_out_indices, val_target, val_input.indices
                                    )
                                    if val_mismatch:
                                        pass  # Âm thầm bỏ qua lỗi không khớp thưa thớt (sparse mismatch) trong lô xác thực (val batch)
                                    val_loss_dict = loss_fn(
                                        val_recon_aligned,
                                        val_target_aligned,
                                        val_mu,
                                        val_logvar,
                                        feature_mode=vae_cfg.input_feature_mode,
                                        rho_logits_list=val_rho_logits,
                                        rho_targets_list=val_rho_targets,
                                    )

                                    # --- Các số liệu đo lường 3D nâng cao ---
                                    # Khoảng cách Chamfer (chỉ 4 lô đầu tiên để tiết kiệm thời gian)
                                    if val_batches < 4:
                                        val_cd_sum += compute_chamfer_distance(val_recon_aligned[:, :3], val_target_aligned[:, :3])
                                    
                                    # Tính nhất quán của Pháp tuyến (Normal Consistency)
                                    if val_target_aligned.shape[1] >= 6:
                                        val_nc_sum += compute_normal_consistency(val_recon_aligned[:, 3:6], val_target_aligned[:, 3:6])
                                    
                                    # Voxel IoU (nếu có đầu ra rho)
                                    if len(val_rho_logits) > 0 and len(val_rho_targets) > 0:
                                        # Sử dụng cấp độ chi tiết nhất (thường là cấp độ cuối cùng trong danh sách)
                                        val_iou_sum += compute_voxel_iou(val_rho_logits[-1], val_rho_targets[-1])

                                    render_items = val_items
                                    if render_items is None and is_pre_cat:
                                        render_items = materialize_batch_items(val_payload)

                                    if viz_items is None and render_items:
                                        viz_items = render_items

                                    val_stage2 = val_target.new_zeros(())
                                    if (
                                        bool(vae_cfg.use_stage2_render_loss)
                                        and epoch >= int(vae_cfg.stage2_render_start_epoch)
                                        and not val_mismatch
                                        and render_items
                                    ):
                                        try:
                                            with torch.amp.autocast('cuda', enabled=False):
                                                val_stage2_dict = compute_stage2_render_perceptual_loss(
                                                    val_recon_aligned.float(),
                                                    val_target_aligned.float(),
                                                    render_items,
                                                    feature_mode=vae_cfg.input_feature_mode,
                                                    image_size=int(vae_cfg.stage2_render_image_size),
                                                    num_views=int(vae_cfg.stage2_render_views),
                                                )
                                            val_stage2 = float(vae_cfg.stage2_render_weight) * (
                                                val_stage2_dict["render_loss"]
                                                + float(vae_cfg.stage2_perceptual_weight) * val_stage2_dict["perceptual_loss"]
                                            )
                                            if torch.isnan(val_stage2) or torch.isinf(val_stage2):
                                                val_stage2 = val_target.new_zeros(())
                                        except Exception as exc:
                                            val_stage2_skips += 1
                                            if val_stage2_skips <= 5 or (val_stage2_skips % 20) == 0:
                                                print(
                                                    f"  [WARN] Stage-2 val render skipped at epoch {epoch+1}: {exc}"
                                                )
                                            if device.type == "cuda" and _is_oom_error(exc):
                                                torch.cuda.empty_cache()
                                            val_stage2 = val_target.new_zeros(())
                                    val_loss = (
                                        val_loss_dict['recon_loss']
                                        + vae_cfg.kl_weight * val_loss_dict['kl_loss']
                                        + rho_scale * float(getattr(vae_cfg, "rho_loss_weight", 0.0)) * val_loss_dict.get('rho_loss', val_target.new_zeros(()))
                                        + val_stage2
                                    )
                            except RuntimeError as exc:
                                if not _is_oom_error(exc) and not _is_sparse_runtime_error(exc):
                                    raise
                                if _is_oom_error(exc):
                                    val_oom_skips += 1
                                else:
                                    # Lỗi nhân thưa (sparse kernel)/lô bệnh lý (pathology batch) có thể phục hồi; bỏ qua thay vì hủy bỏ quá trình chạy.
                                    pass
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                continue

                            val_loss_sum += float(val_loss.item())
                            val_batches += 1
                            
                            # Xóa bộ đệm GPU (GPU cache) trong quá trình xác thực để tránh phân mảnh bộ nhớ
                            if clear_cache_freq > 0 and (val_batches % clear_cache_freq == 0):
                                torch.cuda.empty_cache()

                model.train()
                if val_batches > 0:
                    val_avg_loss = val_loss_sum / val_batches
                    avg_cd = val_cd_sum / min(val_batches, 4)
                    avg_nc = val_nc_sum / val_batches
                    avg_iou = val_iou_sum / val_batches
                    print(
                        f"  Val {epoch+1}/{vae_cfg.num_epochs} | Loss: {val_avg_loss:.4f} | "
                        f"CD: {avg_cd:.6f} | NC: {avg_nc:.4f} | IoU: {avg_iou:.4f} | "
                        f"OOM_skips: {val_oom_skips} | stage2_skips: {val_stage2_skips}"
                    )

                    # --- Lấy mẫu định tính (Qualitative Sampling) ---
                    if (epoch + 1) % 5 == 0 and viz_items is not None:
                        print(f"  [Visual] Saving qualitative samples to points/meshes...")
                        save_validation_samples(
                            model, viz_items, device, epoch + 1, 
                            output_dir=os.path.join(vae_cfg.checkpoint_dir, "val_samples"),
                            spatial_size=cfg.data.voxel_resolution,
                            feature_mode=vae_cfg.input_feature_mode
                        )

                    if cfg.wandb.enabled and WANDB_AVAILABLE:
                        wandb.log({
                            "val/loss": val_avg_loss, 
                            "val/chamfer_distance": avg_cd,
                            "val/normal_consistency": avg_nc,
                            "val/voxel_iou": avg_iou,
                            "val/oom_skipped_batches": val_oom_skips,
                            "val/stage2_skipped_batches": val_stage2_skips,
                        }, step=global_step)

            # Lưu checkpoint
            if (epoch + 1) % vae_cfg.save_every_epochs == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch + 1, avg_loss,
                    os.path.join(vae_cfg.checkpoint_dir, f"epoch_{epoch+1}.pt"),
                    global_step=global_step,
                    batch_size=vae_cfg.batch_size,
                    metadata=checkpoint_metadata,
                )

            # Lưu checkpoint tốt nhất dựa trên loss xác thực để tránh quá khớp (overfitting) với loss huấn luyện.
            if val_avg_loss is not None and val_avg_loss < best_metric:
                best_metric = val_avg_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch + 1, val_avg_loss,
                    os.path.join(vae_cfg.checkpoint_dir, "best.pt"),
                    global_step=global_step,
                    batch_size=vae_cfg.batch_size,
                    metadata=checkpoint_metadata,
                )
            
            # Xóa bộ đệm GPU ở cuối epoch để đảm bảo trạng thái sạch sẽ cho epoch tiếp theo
            if clear_cache_freq > 0:
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPT] Caught Ctrl+C. Saving interrupt checkpoint...")
        if last_loss_tensor is not None:
            last_loss = float(last_loss_tensor.item())
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            current_epoch,
            last_loss,
            os.path.join(vae_cfg.checkpoint_dir, "interrupt.pt"),
            global_step=global_step,
            batch_size=vae_cfg.batch_size,
            metadata=checkpoint_metadata,
        )
        print("[INTERRUPT] Safe checkpoint saved. You can resume later.")
    except Exception as exc:
        print(f"\n[CRASH] Unhandled exception at epoch {current_epoch+1}: {exc}")
        if last_loss_tensor is not None:
            last_loss = float(last_loss_tensor.item())
        crash_ckpt_path = os.path.join(vae_cfg.checkpoint_dir, "crash_latest.pt")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            current_epoch,
            last_loss,
            crash_ckpt_path,
            global_step=global_step,
            batch_size=vae_cfg.batch_size,
            metadata=checkpoint_metadata,
        )
        report_path = _write_crash_report(
            vae_cfg.checkpoint_dir,
            exc,
            current_epoch=current_epoch,
            global_step=global_step,
        )
        print(f"[CRASH] Saved recovery checkpoint: {crash_ckpt_path}")
        print(f"[CRASH] Saved traceback report: {report_path}")
        raise
    
    
    # ---- Hoàn thành ----
    if interrupted:
        if best_metric < float('inf'):
            print(f"\n[4/4] Training paused safely. Best val loss so far: {best_metric:.4f}")
        else:
            print("\n[4/4] Training paused safely. No validation checkpoint selected yet.")
    else:
        if best_metric < float('inf'):
            print(f"\n[4/4] Training complete! Best val loss: {best_metric:.4f}")
        else:
            print("\n[4/4] Training complete! No validation checkpoint selected.")
    print(f"  Checkpoints: {vae_cfg.checkpoint_dir}/")
    
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FaceDiff SC-VAE Training (Stage 1)")
    parser.add_argument("--dataset", type=str, choices=["facescape", "faceverse", "both"], default=None, help="Chọn Dataset tải lên")
    parser.add_argument("--facescape-root", type=str, default=None, help="Override FaceScape data root")
    parser.add_argument("--faceverse-root", type=str, default=None, help="Override FaceVerse data root")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--max-voxels", type=int, default=None, help="Override max_voxels_per_mesh")
    parser.add_argument("--max-points-per-batch", type=int, default=None, help="Cap total sparse points in each batch (0 disables)")
    parser.add_argument("--lmdb-dir", type=str, default=None, help="Đường dẫn đến thư mục LMDB đã pre-pack (để tăng tốc IO DataLoader)")
    parser.add_argument("--lmdb-readahead", action="store_true", help="Enable LMDB OS readahead to reduce random read stalls")
    parser.add_argument("--no-lmdb-readahead", action="store_true", help="Disable LMDB OS readahead")
    parser.add_argument("--lmdb-only", action="store_true", help="When LMDB is enabled, skip local .pt cache probes/writebacks to avoid HDD metadata stalls")
    parser.add_argument("--resume-model-only", action="store_true", help="Resume model weights only (ignore optimizer/scheduler states)")
    parser.add_argument("--allow-unsafe-resume", action="store_true", help="Override resume signature/contract checks and force full-state resume")
    parser.add_argument("--save-every-steps", type=int, default=None, help="Save latest_step checkpoint every N steps")
    parser.add_argument("--val-split", type=float, default=None, help="Validation split ratio (0.0 disables validation)")
    parser.add_argument("--val-every-epochs", type=int, default=None, help="Run validation every N epochs")
    parser.add_argument("--in-channels", type=int, default=None, help="Override SC-VAE in_channels")
    parser.add_argument("--feature-mode", type=str, default=None, choices=["geom6", "geom_mat12", "mat6", "rgb3", "shape_native", "shape_mat"], help="Override O-Voxel feature branch")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--enable-stage2-render-loss", action="store_true", help="Enable stage-2 render/perceptual loss")
    parser.add_argument("--stage2-render-start-epoch", type=int, default=None, help="Epoch to start stage-2 render/perceptual loss")
    parser.add_argument("--stage2-render-weight", type=float, default=None, help="Global weight for stage-2 render branch")
    parser.add_argument("--stage2-perceptual-weight", type=float, default=None, help="Weight for stage-2 perceptual term")
    parser.add_argument("--stage2-render-views", type=int, default=None, choices=[2, 3, 4], help="Number of orthographic views for stage-2 render loss")
    parser.add_argument("--stage2-render-image-size", type=int, default=None, help="Projection map size for stage-2 render loss")
    parser.add_argument("--rho-loss-weight", type=float, default=None, help="Weight for early-pruning rho BCE supervision")
    parser.add_argument("--rho-warmup-epochs", type=int, default=None, help="Warmup epochs for rho loss scale")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override DataLoader prefetch_factor")
    parser.add_argument("--dataloader-timeout", type=int, default=None, help="DataLoader worker timeout in seconds (0 disables)")
    parser.add_argument("--loader-split-points", type=int, default=0, help="Load-balance each batch into micro-packs by sparse point budget (0 disables)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=33, help="Gradient accumulation steps (effective_batch = batch_size * this). Default 33 for batch=4 → effective 132")
    parser.add_argument("--use-activation-checkpointing", action="store_true", help="Enable activation checkpointing to save VRAM (recompute activations during backward)")
    parser.add_argument("--clear-cache-freq", type=int, default=0, help="Clear GPU cache every N batches (0 disables, prevents VRAM flicker)")
    parser.add_argument("--perf-log-every-steps", type=int, default=0, help="Print per-step perf timings every N steps (0 disables)")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile and run eager mode")
    parser.add_argument("--wandb-log-every-steps", type=int, default=None, help="Override WandB logging interval in steps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--faceverse-train-ids", type=str, default="train_faceverse_ids.txt", help="Path to FaceVerse train IDs file")
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt", help="Path to FaceVerse test IDs file (excluded from train)")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt", help="Path to FaceScape train IDs file")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt", help="Path to FaceScape test IDs file (excluded from train)")
    parser.add_argument("--disable-id-filters", action="store_true", help="Disable train/test identity file filtering for custom datasets")
    # Resume scheduler extension flags (TRELLIS.2-style fine-tune past original cosine end)
    parser.add_argument(
        "--resume-scheduler-mode",
        type=str,
        default="continue",
        choices=["continue", "constant_min_lr", "cosine_restart"],
        help=(
            "How to schedule LR after a resume: 'continue' = re-use the original cosine schedule (default, "
            "preserves backward-compat); 'constant_min_lr' = pin LR at --resume-target-min-lr for the rest of "
            "training (safe for late-stage refinement); 'cosine_restart' = SGDR-style half-cosine warm restart "
            "from the current LR down to --resume-target-min-lr over --resume-extend-epochs epochs (recommended "
            "when resuming from a near-end checkpoint such as epoch 397 of an original 500-epoch run)."
        ),
    )
    parser.add_argument(
        "--resume-extend-epochs",
        type=int,
        default=100,
        help="Number of epochs the post-resume cosine_restart spans (only used when --resume-scheduler-mode=cosine_restart).",
    )
    parser.add_argument(
        "--resume-target-min-lr",
        type=float,
        default=None,
        help="Floor LR for the post-resume schedule (defaults to cfg.sc_vae.min_lr, normally 1e-6).",
    )
    args = parser.parse_args()
    
    cfg = TrainConfig()
    
    if args.dataset:
        cfg.data.active_dataset = args.dataset
    if args.facescape_root:
        cfg.data.facescape_root = args.facescape_root
    if args.faceverse_root:
        cfg.data.faceverse_root = args.faceverse_root
    if args.resume:
        cfg.sc_vae.resume_from = args.resume
    if args.epochs:
        cfg.sc_vae.num_epochs = args.epochs
    if args.batch_size:
        cfg.sc_vae.batch_size = args.batch_size
    if args.lr:
        cfg.sc_vae.learning_rate = args.lr
    if args.max_voxels:
        cfg.sc_vae.max_voxels_per_mesh = args.max_voxels
    if args.max_points_per_batch is not None:
        cfg.sc_vae.max_points_per_batch = max(0, int(args.max_points_per_batch))
    if args.resume_model_only:
        cfg.sc_vae.resume_model_only = True
    if args.save_every_steps is not None:
        cfg.sc_vae.save_every_steps = args.save_every_steps
    if args.val_split is not None:
        cfg.sc_vae.val_split = max(0.0, min(0.5, args.val_split))
    if args.val_every_epochs is not None:
        cfg.sc_vae.val_every_epochs = max(1, args.val_every_epochs)
    if args.in_channels is not None:
        cfg.sc_vae.in_channels = max(1, int(args.in_channels))
    if args.feature_mode is not None:
        cfg.sc_vae.input_feature_mode = args.feature_mode
    if args.checkpoint_dir is not None:
        cfg.sc_vae.checkpoint_dir = args.checkpoint_dir
    if args.enable_stage2_render_loss:
        cfg.sc_vae.use_stage2_render_loss = True
    if args.stage2_render_start_epoch is not None:
        cfg.sc_vae.stage2_render_start_epoch = max(0, int(args.stage2_render_start_epoch))
    if args.stage2_render_weight is not None:
        cfg.sc_vae.stage2_render_weight = max(0.0, float(args.stage2_render_weight))
    if args.stage2_perceptual_weight is not None:
        cfg.sc_vae.stage2_perceptual_weight = max(0.0, float(args.stage2_perceptual_weight))
    if args.stage2_render_views is not None:
        cfg.sc_vae.stage2_render_views = int(args.stage2_render_views)
    if args.stage2_render_image_size is not None:
        cfg.sc_vae.stage2_render_image_size = max(16, int(args.stage2_render_image_size))
    if args.rho_loss_weight is not None:
        cfg.sc_vae.rho_loss_weight = max(0.0, float(args.rho_loss_weight))
    if args.rho_warmup_epochs is not None:
        cfg.sc_vae.rho_warmup_epochs = max(0, int(args.rho_warmup_epochs))
    if args.num_workers is not None:
        cfg.data.num_workers = max(0, int(args.num_workers))
    if args.prefetch_factor is not None:
        cfg.data.prefetch_factor = max(1, int(args.prefetch_factor))
    if getattr(args, "dataloader_timeout", None) is not None:
        cfg.data.dataloader_timeout = max(0, int(args.dataloader_timeout))
    if getattr(args, "lmdb_dir", None) is not None:
        cfg.data.lmdb_dir = args.lmdb_dir
    if args.lmdb_readahead:
        cfg.data.lmdb_readahead = True
    if args.no_lmdb_readahead:
        cfg.data.lmdb_readahead = False
    if args.wandb_log_every_steps is not None:
        cfg.wandb.log_every_steps = max(1, int(args.wandb_log_every_steps))
    # Resume scheduler extension overrides (read by train_sc_vae() through cfg.sc_vae attrs)
    setattr(cfg.sc_vae, "resume_scheduler_mode", str(args.resume_scheduler_mode))
    setattr(cfg.sc_vae, "resume_extend_epochs", int(args.resume_extend_epochs))
    if args.resume_target_min_lr is not None:
        setattr(cfg.sc_vae, "resume_target_min_lr", float(args.resume_target_min_lr))

    # Trích xuất các tham số tối ưu hóa mới
    loader_split_points = max(0, int(args.loader_split_points))
    gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    use_activation_checkpointing = args.use_activation_checkpointing
    clear_cache_freq = max(0, args.clear_cache_freq)
    perf_log_every_steps = max(0, int(args.perf_log_every_steps))
    
    if args.no_wandb:
        cfg.wandb.enabled = False
    
    train_sc_vae(
        cfg,
        faceverse_train_ids_file=args.faceverse_train_ids,
        faceverse_test_ids_file=args.faceverse_test_ids,
        facescape_train_ids_file=args.facescape_train_ids,
        facescape_test_ids_file=args.facescape_test_ids,
        loader_split_points=loader_split_points,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_activation_checkpointing=use_activation_checkpointing,
        clear_cache_freq=clear_cache_freq,
        perf_log_every_steps=perf_log_every_steps,
        disable_id_filters=bool(args.disable_id_filters),
        lmdb_only=bool(args.lmdb_only),
        enable_torch_compile=not bool(args.no_torch_compile),
        allow_unsafe_resume=bool(args.allow_unsafe_resume),
    )


if __name__ == "__main__":
    main()
