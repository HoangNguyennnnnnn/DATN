"""
FaceDiff — Giai đoạn 0/1: Huấn luyện Trình tạo Cấu trúc Thưa thớt (Sparse Structure Generator Training)
==========================================================
Học một bố cục chiếm chỗ tiềm ẩn thưa thớt (coarse sparse latent occupancy layout) (mặc định là 16^3 tokens)
được điều hướng bởi vector ngữ cảnh lai (hybrid context vector).

Cách sử dụng:
    python src/train_structure.py
    python src/train_structure.py --dataset both --epochs 200
    python src/train_structure.py --use-extractor-context
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainConfig
from src.data.ovoxel_converter import OVoxelConverter
from src.models.structure_generator import SparseStructureGenerator
from src.utils import extract_identity_from_obj_path, load_identity_set


try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def _stable_context_from_path(path: str, context_dim: int) -> torch.Tensor:
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vec = torch.randn(context_dim, generator=gen)
    vec = F.normalize(vec, p=2, dim=0)
    return vec.to(dtype=torch.float32)


class StructureDataset(Dataset):
    """Dataset của các cặp (structure_mask, context_vector)."""

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        converter: OVoxelConverter,
        slat_length: int,
        ovoxel_resolution: int,
        context_dim: int,
        cache_dir: str,
        include_ids=None,
        exclude_ids=None,
        mesh_renderer: Optional[object] = None,
        arcface: Optional[object] = None,
        flame: Optional[object] = None,
        dinov2: Optional[object] = None,
        hybrid_context_builder=None,
    ):
        self.data_root = data_root
        self.dataset_name = str(dataset_name)
        self.converter = converter
        self.slat_length = int(slat_length)
        self.ovoxel_resolution = int(ovoxel_resolution)
        self.context_dim = int(context_dim)
        self.cache_dir = cache_dir
        self.include_ids = include_ids
        self.exclude_ids = exclude_ids
        self.mesh_renderer = mesh_renderer
        self.arcface = arcface
        self.flame = flame
        self.dinov2 = dinov2
        self.hybrid_context_builder = hybrid_context_builder
        self.samples = []

        grid = round(self.slat_length ** (1.0 / 3.0))
        if int(grid) ** 3 != self.slat_length:
            raise ValueError(
                f"slat_length={self.slat_length} is not a perfect cube; "
                "unable to build structure occupancy tokens."
            )
        self.latent_grid = int(grid)

        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.isdir(self.data_root):
            for root_dir, _, files in os.walk(self.data_root):
                for name in sorted(files):
                    if not name.endswith(".obj"):
                        continue
                    obj_path = os.path.join(root_dir, name)
                    identity = extract_identity_from_obj_path(
                        obj_path, self.data_root, self.dataset_name
                    )
                    if self.include_ids is not None and identity not in self.include_ids:
                        continue
                    if self.exclude_ids is not None and identity in self.exclude_ids:
                        continue
                    self.samples.append(obj_path)

        print(f"[StructureDataset] Found {len(self.samples)} meshes from {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, obj_path: str) -> str:
        rel = os.path.relpath(obj_path, self.data_root)
        safe = rel.replace(os.path.sep, "_").replace(
            ".obj",
            f".structure.g{self.latent_grid}.r{self.ovoxel_resolution}.pt",
        )
        return os.path.join(self.cache_dir, safe)

    def _coords_to_structure_mask(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(dtype=torch.int64)
        scale = max(1, self.ovoxel_resolution // self.latent_grid)
        coarse = torch.div(coords, scale, rounding_mode="floor")
        coarse = coarse.clamp(0, self.latent_grid - 1)

        # Việc lập chỉ mục token (Token indexing) phải khớp với trình tạo (generator) _get_slat_grid_indices (theo thứ tự z, y, x).
        idx = (
            coarse[:, 0] * (self.latent_grid * self.latent_grid)
            + coarse[:, 1] * self.latent_grid
            + coarse[:, 2]
        )
        mask = torch.zeros(self.slat_length, dtype=torch.float32)
        if idx.numel() > 0:
            mask[idx.unique()] = 1.0
        return mask

    def _extract_context(self, obj_path: str) -> torch.Tensor:
        if (
            self.mesh_renderer is not None
            and self.arcface is not None
            and self.flame is not None
            and self.dinov2 is not None
            and self.hybrid_context_builder is not None
        ):
            try:
                front, back = self.mesh_renderer.render_front_and_back(obj_path)
                identity = self.arcface.extract_identity(front)
                expr = self.flame.extract_from_image(front)
                back_sh = self.dinov2.extract_features(back)
                ctx = self.hybrid_context_builder(identity, expr, back_sh).squeeze(0)
                return ctx.to(dtype=torch.float32).cpu()
            except Exception as exc:
                print(f"[StructureDataset] Context extractor fallback for {obj_path}: {exc}")

        return _stable_context_from_path(obj_path, self.context_dim)

    def __getitem__(self, idx: int):
        obj_path = self.samples[idx]
        cache_path = self._cache_path(obj_path)

        if os.path.exists(cache_path):
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                mask = payload.get("structure_mask", None)
                context = payload.get("context", None)
                if (
                    isinstance(mask, torch.Tensor)
                    and isinstance(context, torch.Tensor)
                    and mask.ndim == 1
                    and context.ndim == 1
                    and mask.shape[0] == self.slat_length
                    and context.shape[0] == self.context_dim
                ):
                    return mask.to(dtype=torch.float32), context.to(dtype=torch.float32)
            except Exception:
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        ovoxel = self.converter.process_mesh(obj_path)
        coords = torch.as_tensor(ovoxel["coords"], dtype=torch.int64)
        mask = self._coords_to_structure_mask(coords)
        context = self._extract_context(obj_path)

        torch.save({"structure_mask": mask, "context": context}, cache_path)
        return mask, context


def collate_structure(batch):
    masks = torch.stack([item[0] for item in batch], dim=0)
    contexts = torch.stack([item[1] for item in batch], dim=0)
    return masks, contexts


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    loss: float,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "loss": float(loss),
            "model_config": {
                "context_dim": int(model.context_dim),
                "slat_length": int(model.slat_length),
                "hidden_dim": int(model.hidden_dim),
                "num_layers": int(model.num_layers),
                "num_heads": int(model.num_heads),
                "num_context_tokens": int(model.num_context_tokens),
            },
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict", None) is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return int(ckpt.get("epoch", 0))


def train_structure(
    cfg: TrainConfig,
    faceverse_train_ids_file: str = "train_faceverse_ids.txt",
    faceverse_test_ids_file: str = "test_faceverse_ids.txt",
    facescape_train_ids_file: str = "train_facescape_ids.txt",
    facescape_test_ids_file: str = "test_facescape_ids.txt",
    disable_id_filters: bool = False,
    use_extractor_context: bool = False,
):
    device = torch.device(cfg.device)
    st_cfg = cfg.structure

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 60)
    print("  FACEDIFF — STAGE 0/1: STRUCTURE GENERATOR TRAINING")
    print("=" * 60)

    converter = OVoxelConverter(
        resolution=int(st_cfg.ovoxel_resolution),
        device="cpu",
    )

    renderer = arcface = flame = dinov2 = None
    hybrid_context_builder = None
    if use_extractor_context:
        print("[StructureTrain] Initializing context extractors...")
        from src.data.mesh_renderer import MeshRenderer
        from src.data.arcface_extractor import ArcFaceExtractor
        from src.data.flame_adapter import FLAMEExpressionAdapter, create_hybrid_context
        from src.data.feature_extractor import DinoV3Extractor

        renderer = MeshRenderer(device=cfg.device, image_size=512)
        arcface = ArcFaceExtractor(device=cfg.device)
        flame = FLAMEExpressionAdapter(expression_dim=50, device=cfg.device)
        dinov2 = DinoV3Extractor(model_name="facebook/dinov2-small", device=cfg.device)
        hybrid_context_builder = create_hybrid_context

    if disable_id_filters:
        faceverse_include_ids = None
        faceverse_exclude_ids = None
        facescape_include_ids = None
        facescape_exclude_ids = None
    else:
        faceverse_include_ids = load_identity_set(faceverse_train_ids_file)
        faceverse_exclude_ids = load_identity_set(faceverse_test_ids_file)
        facescape_include_ids = load_identity_set(facescape_train_ids_file)
        facescape_exclude_ids = load_identity_set(facescape_test_ids_file)

    datasets = []
    if cfg.data.active_dataset in {"faceverse", "both"} and os.path.isdir(cfg.data.faceverse_root):
        datasets.append(
            StructureDataset(
                data_root=cfg.data.faceverse_root,
                dataset_name="faceverse",
                converter=converter,
                slat_length=int(st_cfg.slat_length),
                ovoxel_resolution=int(st_cfg.ovoxel_resolution),
                context_dim=int(st_cfg.context_dim),
                cache_dir=str(st_cfg.cache_dir),
                include_ids=faceverse_include_ids,
                exclude_ids=faceverse_exclude_ids,
                mesh_renderer=renderer,
                arcface=arcface,
                flame=flame,
                dinov2=dinov2,
                hybrid_context_builder=hybrid_context_builder,
            )
        )

    if cfg.data.active_dataset in {"facescape", "both"} and os.path.isdir(cfg.data.facescape_root):
        datasets.append(
            StructureDataset(
                data_root=cfg.data.facescape_root,
                dataset_name="facescape",
                converter=converter,
                slat_length=int(st_cfg.slat_length),
                ovoxel_resolution=int(st_cfg.ovoxel_resolution),
                context_dim=int(st_cfg.context_dim),
                cache_dir=str(st_cfg.cache_dir) + "_facescape",
                include_ids=facescape_include_ids,
                exclude_ids=facescape_exclude_ids,
                mesh_renderer=renderer,
                arcface=arcface,
                flame=flame,
                dinov2=dinov2,
                hybrid_context_builder=hybrid_context_builder,
            )
        )

    if len(datasets) == 0:
        raise ValueError("No structure datasets found for selected roots/dataset mode.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=int(st_cfg.batch_size),
        shuffle=True,
        num_workers=max(0, int(cfg.data.num_workers)),
        pin_memory=bool(cfg.data.pin_memory and device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_structure,
    )

    print(f"[StructureTrain] Dataset size={len(dataset)} | batches/epoch={len(dataloader)}")

    model = SparseStructureGenerator(
        context_dim=int(st_cfg.context_dim),
        slat_length=int(st_cfg.slat_length),
        hidden_dim=int(st_cfg.hidden_dim),
        num_layers=int(st_cfg.num_layers),
        num_heads=int(st_cfg.num_heads),
        num_context_tokens=int(st_cfg.num_context_tokens),
        dropout=float(st_cfg.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(st_cfg.learning_rate),
        weight_decay=float(st_cfg.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(st_cfg.num_epochs) * max(1, len(dataloader))),
        eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(st_cfg.use_amp and device.type == "cuda"))

    start_epoch = 0
    best_loss = float("inf")
    if st_cfg.resume_from and os.path.exists(st_cfg.resume_from):
        start_epoch = _load_checkpoint(
            st_cfg.resume_from,
            model,
            optimizer=None if st_cfg.resume_model_only else optimizer,
            scheduler=None if st_cfg.resume_model_only else scheduler,
            scaler=None if st_cfg.resume_model_only else scaler,
        )
        print(f"[StructureTrain] Resumed from epoch={start_epoch}")

    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or f"structure_{time.strftime('%m%d_%H%M')}",
            tags=cfg.wandb.tags + ["structure"],
            config={
                "stage": "structure",
                "batch_size": int(st_cfg.batch_size),
                "lr": float(st_cfg.learning_rate),
                "epochs": int(st_cfg.num_epochs),
                "slat_length": int(st_cfg.slat_length),
            },
        )

    os.makedirs(st_cfg.checkpoint_dir, exist_ok=True)
    global_step = 0

    for epoch in range(start_epoch, int(st_cfg.num_epochs)):
        model.train()
        t_start = time.time()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_pred_active = 0.0
        epoch_tgt_active = 0.0

        for masks, contexts in dataloader:
            masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
            contexts = contexts.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(st_cfg.use_amp and device.type == "cuda")):
                logits = model(contexts)

                pos_ratio = masks.mean().detach()
                pos_w = ((1.0 - pos_ratio) / (pos_ratio + 1e-6)).clamp(
                    min=1.0, max=float(st_cfg.max_pos_weight)
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    masks,
                    pos_weight=pos_w,
                )

            scaler.scale(loss).backward()
            if float(st_cfg.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(st_cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= float(st_cfg.occupancy_threshold)).to(masks.dtype)
                inter = (pred * masks).sum(dim=1)
                union = ((pred + masks) > 0).to(masks.dtype).sum(dim=1)
                iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))

                pred_active = pred.mean()
                tgt_active = masks.mean()

            epoch_loss += float(loss.item())
            epoch_iou += float(iou.mean().item())
            epoch_pred_active += float(pred_active.item())
            epoch_tgt_active += float(tgt_active.item())
            global_step += 1

            if cfg.wandb.enabled and WANDB_AVAILABLE and global_step % cfg.wandb.log_every_steps == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/iou": float(iou.mean().item()),
                        "train/pred_active_ratio": float(pred_active.item()),
                        "train/tgt_active_ratio": float(tgt_active.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "train/epoch": int(epoch),
                    },
                    step=global_step,
                )

        n_batches = max(1, len(dataloader))
        avg_loss = epoch_loss / n_batches
        avg_iou = epoch_iou / n_batches
        avg_pred_active = epoch_pred_active / n_batches
        avg_tgt_active = epoch_tgt_active / n_batches
        elapsed = time.time() - t_start

        print(
            f"Epoch {epoch+1}/{int(st_cfg.num_epochs)} | "
            f"loss={avg_loss:.4f} | iou={avg_iou:.4f} | "
            f"active(pred/tgt)={avg_pred_active:.4f}/{avg_tgt_active:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        if cfg.wandb.enabled and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch/iou": avg_iou,
                    "epoch/pred_active_ratio": avg_pred_active,
                    "epoch/tgt_active_ratio": avg_tgt_active,
                },
                step=global_step,
            )

        if (epoch + 1) % int(st_cfg.save_every_epochs) == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                avg_loss,
                os.path.join(st_cfg.checkpoint_dir, f"epoch_{epoch+1}.pt"),
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                avg_loss,
                os.path.join(st_cfg.checkpoint_dir, "best.pt"),
            )

    print(f"[StructureTrain] Done. Best loss={best_loss:.4f}")
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="FaceDiff Sparse Structure Generator Training")
    parser.add_argument("--dataset", type=str, choices=["facescape", "faceverse", "both"], default=None)
    parser.add_argument("--facescape-root", type=str, default=None)
    parser.add_argument("--faceverse-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-extractor-context", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--faceverse-train-ids", type=str, default="train_faceverse_ids.txt")
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt")
    parser.add_argument("--disable-id-filters", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.dataset is not None:
        cfg.data.active_dataset = args.dataset
    if args.facescape_root is not None:
        cfg.data.facescape_root = args.facescape_root
    if args.faceverse_root is not None:
        cfg.data.faceverse_root = args.faceverse_root
    if args.resume is not None:
        cfg.structure.resume_from = args.resume
    if args.resume_model_only:
        cfg.structure.resume_model_only = True
    if args.epochs is not None:
        cfg.structure.num_epochs = int(max(1, args.epochs))
    if args.batch_size is not None:
        cfg.structure.batch_size = int(max(1, args.batch_size))
    if args.lr is not None:
        cfg.structure.learning_rate = float(max(1e-7, args.lr))
    if args.checkpoint_dir is not None:
        cfg.structure.checkpoint_dir = args.checkpoint_dir
    if args.num_workers is not None:
        cfg.data.num_workers = int(max(0, args.num_workers))
    if args.no_wandb:
        cfg.wandb.enabled = False

    train_structure(
        cfg,
        faceverse_train_ids_file=args.faceverse_train_ids,
        faceverse_test_ids_file=args.faceverse_test_ids,
        facescape_train_ids_file=args.facescape_train_ids,
        facescape_test_ids_file=args.facescape_test_ids,
        disable_id_filters=bool(args.disable_id_filters),
        use_extractor_context=bool(args.use_extractor_context),
    )


if __name__ == "__main__":
    main()
