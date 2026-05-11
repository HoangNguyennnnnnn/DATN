"""
FaceDiff — Giai đoạn 2: Huấn luyện iMF (iMF Training)
=================================
Huấn luyện mô hình Improved Mean Flow trên Slat tokens.
Yêu cầu: SC-VAE đã train xong (Giai đoạn 1) để mã hóa (encode) dữ liệu thành các mục tiêu Slat (Slat targets).

Tính năng:
- EMA (Trung bình trượt hàm mũ - Exponential Moving Average) cho trọng số mô hình → chất lượng lấy mẫu (sampling) tốt hơn
- Điều hướng ArcFace (ArcFace conditioning) (vector danh tính 512 chiều)
- Độ chính xác hỗn hợp (Mixed precision - bfloat16)
- Lưu/Phục hồi checkpoint
- Ghi log bằng WandB
- Cắt gradient (Gradient clipping)

Cách sử dụng:
    python src/train_imf.py
    python src/train_imf.py --resume checkpoints/imf_unet/epoch_100.pt
    python src/train_imf.py --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt
    python src/train_imf.py --dual-branch --shape-sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt --material-sc-vae-ckpt checkpoints/sc_vae_material/latest_step.pt
"""

import sys
import os
import time
import json
import hashlib
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainConfig
from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.sc_vae import SC_VAE, SPCONV_AVAILABLE
from src.data.mesh_renderer import MeshRenderer
from src.data.ovoxel_converter import OVoxelConverter
from src.data.feature_extractor import DinoV3Extractor
from src.data.arcface_extractor import ArcFaceExtractor
from src.data.flame_adapter import FLAMEExpressionAdapter, create_hybrid_context

from src.utils import load_identity_set, extract_identity_from_obj_path

if SPCONV_AVAILABLE:
    import spconv.pytorch as spconv
else:
    spconv = None

# ============================================================
# WandB
# ============================================================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Train] wandb not installed. pip install wandb")


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
class EMA:
    """
    Trung bình trượt hàm mũ (Exponential Moving Average) cho trọng số mô hình.
    Giữ bản sao trung bình chạy (running average) của các tham số → lấy mẫu (sampling) mượt hơn.
    
    Theo DDPM, hệ số suy giảm (decay) EMA 0.9999 cho kết quả tốt nhất.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters() if param.requires_grad}
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply(self, model: nn.Module):
        """Áp dụng trọng số EMA vào mô hình (cho suy luận - inference)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Khôi phục weights gốc (sau inference)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ============================================================
# Slat Dataset — Encode meshes thành Slat tokens bằng pretrained SC-VAE
# ============================================================
class SlatDataset(Dataset):
    """
    Dataset trả về (slat_tokens, identity_vector) cho huấn luyện iMF.
    
    Luồng xử lý (Workflow):
    1. Tải lưới (Load mesh) → O-Voxel features (tái sử dụng logic của VoxelDataset)
    2. Mã hóa qua SC-VAE đã được huấn luyện trước → Slat tokens [slat_length, latent_dim]
    3. Trích xuất vector danh tính ArcFace [512] từ hình ảnh được kết xuất (rendered image)
    
    Bộ đệm (Caching): Slat tokens được lưu đệm vào các tệp .pt để tăng tốc độ.
    """
    CACHE_SCHEMA_VERSION = 2
    
    def __init__(self, data_root: str, sc_vae: SC_VAE,
                 dataset_name: str,
                 mesh_renderer=None, arcface=None, flame=None, dinov2=None,
                 slat_length: int = 4096, latent_dim: int = 16,
                 cache_dir: str = "data/slat_cache", device: str = "cuda:0",
                 include_ids=None, exclude_ids=None,
                 dual_branch: bool = False,
                 shape_sc_vae: SC_VAE = None,
                 material_sc_vae: SC_VAE = None,
                 shape_feature_mode: str = "shape_native",
                 material_feature_mode: str = "rgb3",
                 shape_target_in_channels: int = 7,
                 material_target_in_channels: int = 3,
                 context_dim: int = 946,
                 single_sc_vae_checkpoint: str | None = None,
                 shape_sc_vae_checkpoint: str | None = None,
                 material_sc_vae_checkpoint: str | None = None,
                 ovoxel_resolution: int = 256):
        self.data_root = data_root
        self.sc_vae = sc_vae
        self.dataset_name = str(dataset_name)
        self.mesh_renderer = mesh_renderer
        self.arcface = arcface
        self.flame = flame
        self.dinov2 = dinov2
        self.slat_length = slat_length
        self.latent_dim = latent_dim
        self.cache_dir = cache_dir
        self.device = device
        self.include_ids = include_ids
        self.exclude_ids = exclude_ids
        self.dual_branch = bool(dual_branch)
        self.shape_sc_vae = shape_sc_vae
        self.material_sc_vae = material_sc_vae
        self.shape_feature_mode = shape_feature_mode
        self.material_feature_mode = material_feature_mode
        self.shape_target_in_channels = int(shape_target_in_channels)
        self.material_target_in_channels = int(material_target_in_channels)
        self.context_dim = int(context_dim)
        self.single_sc_vae_checkpoint = single_sc_vae_checkpoint
        self.shape_sc_vae_checkpoint = shape_sc_vae_checkpoint
        self.material_sc_vae_checkpoint = material_sc_vae_checkpoint
        self.ovoxel_resolution = int(max(8, ovoxel_resolution))
        self.ovoxel_converter = None
        self.samples = []

        try:
            self.ovoxel_converter = OVoxelConverter(
                resolution=self.ovoxel_resolution,
                device="cpu",
            )
        except Exception as exc:
            print(f"[SlatDataset] Warning: O-Voxel converter init failed, using mesh fallback: {exc}")
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_contract = self._build_cache_contract()
        contract_blob = json.dumps(self.cache_contract, sort_keys=True, separators=(",", ":"))
        self.cache_tag = f"slatv{self.CACHE_SCHEMA_VERSION}_{hashlib.sha1(contract_blob.encode('utf-8')).hexdigest()[:12]}"
        print(f"[SlatDataset] Cache tag: {self.cache_tag}")
        
        # Quét các tệp mesh đệ quy
        skipped_by_include = 0
        skipped_by_exclude = 0
        if os.path.isdir(data_root):
            for root_dir, _, files in os.walk(data_root):
                for f in sorted(files):
                    if f.endswith('.obj'):
                        obj_path = os.path.join(root_dir, f)
                        identity = extract_identity_from_obj_path(obj_path, self.data_root, self.dataset_name)
                        if self.include_ids is not None and identity not in self.include_ids:
                            skipped_by_include += 1
                            continue
                        if self.exclude_ids is not None and identity in self.exclude_ids:
                            skipped_by_exclude += 1
                            continue
                        self.samples.append(obj_path)
        
        print(f"[SlatDataset] Found {len(self.samples)} meshes from {data_root}")
        if self.include_ids is not None or self.exclude_ids is not None:
            print(
                f"[SlatDataset] Filtered: include_skip={skipped_by_include}, "
                f"exclude_skip={skipped_by_exclude}"
            )

    def _checkpoint_signature(self, ckpt_path: str | None) -> str:
        if not ckpt_path:
            return "none"
        abs_path = os.path.abspath(ckpt_path)
        if not os.path.exists(abs_path):
            return f"missing:{abs_path}"
        stat = os.stat(abs_path)
        return f"{os.path.basename(abs_path)}:{stat.st_size}:{stat.st_mtime_ns}"

    def _build_cache_contract(self) -> dict:
        return {
            "schema_version": int(self.CACHE_SCHEMA_VERSION),
            "dataset_name": self.dataset_name,
            "dual_branch": bool(self.dual_branch),
            "slat_length": int(self.slat_length),
            "latent_dim": int(self.latent_dim),
            "context_dim": int(self.context_dim),
            "ovoxel_resolution": int(self.ovoxel_resolution),
            "shape_feature_mode": str(self.shape_feature_mode or "shape_native").strip().lower(),
            "material_feature_mode": str(self.material_feature_mode or "rgb3").strip().lower(),
            "shape_target_in_channels": int(self.shape_target_in_channels),
            "material_target_in_channels": int(self.material_target_in_channels),
            "single_sc_vae_checkpoint": self._checkpoint_signature(self.single_sc_vae_checkpoint),
            "shape_sc_vae_checkpoint": self._checkpoint_signature(self.shape_sc_vae_checkpoint),
            "material_sc_vae_checkpoint": self._checkpoint_signature(self.material_sc_vae_checkpoint),
        }
    
    def __len__(self) -> int:
        return len(self.samples)

    def cache_file_path(self, idx: int) -> str:
        """Absolute path to the on-disk cache file for sample ``idx`` (same naming as ``__getitem__``)."""
        obj_path = self.samples[idx]
        rel_path = os.path.relpath(obj_path, self.data_root)
        suffix = ".dual.pt" if self.dual_branch else ".pt"
        base_name = rel_path.replace(os.path.sep, "_").replace(".obj", "")
        safe_name = f"{base_name}.{self.cache_tag}{suffix}"
        return os.path.join(self.cache_dir, safe_name)

    def has_valid_cache(self, idx: int) -> bool:
        """Return True if a readable cache exists with matching ``cache_tag``."""
        cache_path = self.cache_file_path(idx)
        if not os.path.isfile(cache_path):
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            meta = payload.get("meta", {})
            return meta.get("cache_tag") == self.cache_tag and "slat" in payload and "context" in payload
        except Exception:
            return False

    @torch.no_grad()
    def __getitem__(self, idx: int):
        obj_path = self.samples[idx]
        
        # Tạo tên bộ đệm chống trùng lặp (vd: id125_1_neutral.pt)
        rel_path = os.path.relpath(obj_path, self.data_root)
        suffix = '.dual.pt' if self.dual_branch else '.pt'
        base_name = rel_path.replace(os.path.sep, '_').replace('.obj', '')
        safe_name = f"{base_name}.{self.cache_tag}{suffix}"
        cache_path = os.path.join(self.cache_dir, safe_name)
        
        # Thử tải bộ đệm (Tự chữa lành nếu file bị rỗng do Ctrl+C)
        if os.path.exists(cache_path):
            try:
                cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                cache_meta = cache_payload.get("meta", {})
                if cache_meta.get("cache_tag") != self.cache_tag:
                    raise ValueError("cache contract mismatch")
                return cache_payload['slat'], cache_payload['context']
            except Exception as e:
                print(f"\n[CẢNH BÁO] Cache hỏng (Corrupted): {safe_name}. Đang xoá và tái tạo lại...")
                os.remove(cache_path)
        
        # Nếu bật --offline-data (sc_vae=None) mà không tìm thấy bộ đệm -> Báo lỗi
        if self.sc_vae is None:
            raise RuntimeError(
                "\n[OFFLINE MODE ERROR] Missing slat/context cache for: "
                f"{safe_name}. "
                "Please precompute cache files before training."
            )
            
        # --- Quy trình Mã hóa (Encode) (khi không bật offline-data) ---
        # Mã hóa lưới (Encode mesh) → Slat tokens
        slat, context = self._encode_mesh(obj_path)
        
        # Cache
        slat = slat.cpu()
        context = context.cpu()
        torch.save(
            {
                'slat': slat,
                'context': context,
                'meta': {
                    'cache_tag': self.cache_tag,
                    'cache_contract': self.cache_contract,
                    'relative_path': rel_path,
                },
            },
            cache_path,
        )
        
        return slat, context
    
    @torch.no_grad()
    def _encode_mesh(self, obj_path: str):
        """Mã hóa lưới đơn (Encode single mesh) thành Slat tokens + ngữ cảnh danh tính (identity context)."""
        shape_mat_features, coords = self._load_ovoxel_shape_mat(obj_path)

        if self.dual_branch and self.shape_sc_vae is not None and self.material_sc_vae is not None:
            shape_feats = self._build_branch_features(shape_mat_features, self.shape_feature_mode, self.shape_target_in_channels)
            mat_feats = self._build_branch_features(shape_mat_features, self.material_feature_mode, self.material_target_in_channels)

            shape_mu = self._encode_latents(self.shape_sc_vae, shape_feats, coords, self.shape_target_in_channels)
            mat_mu = self._encode_latents(self.material_sc_vae, mat_feats, coords, self.material_target_in_channels)
            slat = torch.cat([shape_mu, mat_mu], dim=-1)
        else:
            single_mode = self.shape_feature_mode if self.shape_feature_mode is not None else "shape_native"
            target_in_channels = int(getattr(self.sc_vae, "in_channels", self.shape_target_in_channels))
            single_feats = self._build_branch_features(shape_mat_features, single_mode, target_in_channels)
            slat = self._encode_latents(self.sc_vae, single_feats, coords, target_in_channels)
        
        # Pad/truncate to slat_length
        n = slat.shape[0]
        latent_width = int(slat.shape[1]) if slat.ndim == 2 else int(self.latent_dim)
        if n > self.slat_length:
            indices = torch.randperm(n)[:self.slat_length]
            slat = slat[indices]
        elif n < self.slat_length:
            pad = torch.zeros(self.slat_length - n, latent_width, dtype=slat.dtype)
            slat = torch.cat([slat, pad], dim=0)
        
        # 3. Ngữ cảnh lai (Hybrid context) v4.1 — Trích xuất thực tế nếu có bộ trích xuất (extractors)
        if self.mesh_renderer and self.arcface and self.flame and self.dinov2:
            try:
                # Kết xuất (Render) 2D
                front, back = self.mesh_renderer.render_front_and_back(obj_path)
                # Trích xuất đặc trưng
                identity = self.arcface.extract_identity(front)
                expr = self.flame.extract_from_image(front)
                back_sh = self.dinov2.extract_features(back)
                # Gộp [1, 946] rồi nén (squeeze) thành [946]
                context = create_hybrid_context(identity, expr, back_sh).squeeze(0)
            except Exception as e:
                print(f"[SlatDataset] Lỗi khi trích xuất vector do renderer: {e}. Fallback ngẫu nhiên.")
                context = torch.nn.functional.normalize(torch.randn(946), p=2, dim=-1)
        else:
            context = torch.randn(946)
            context = torch.nn.functional.normalize(context, p=2, dim=-1)
        
        return slat, context

    def _load_ovoxel_shape_mat(self, obj_path: str):
        """Tải payload O-Voxel 10 kênh hợp nhất [shape7, rgb3]."""
        if self.ovoxel_converter is not None:
            try:
                payload = self.ovoxel_converter.process_mesh(obj_path)
                feats = torch.as_tensor(payload["shape_mat_features"], dtype=torch.float32)
                coords = torch.as_tensor(payload["coords"], dtype=torch.int32)
                if feats.ndim != 2 or feats.shape[1] < 10:
                    raise ValueError(f"Unexpected shape_mat_features shape: {tuple(feats.shape)}")
                return feats[:, :10].contiguous(), coords.contiguous()
            except Exception as exc:
                print(f"[SlatDataset] O-Voxel conversion failed for {obj_path}, fallback mesh proxy: {exc}")

        # Fallback path keeps training runnable even when converter/runtime is unavailable.
        import trimesh

        mesh = trimesh.load(obj_path, force='mesh', process=False)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
        else:
            normals = torch.zeros_like(verts)

        center = verts.mean(dim=0)
        verts = verts - center
        scale = verts.abs().max() + 1e-8
        verts = (verts / scale).clamp(-1.0, 1.0)

        delta = normals.abs().clamp(0.0, 1.0)
        gamma = torch.ones((verts.shape[0], 1), dtype=torch.float32)

        colors = None
        try:
            vcol = getattr(mesh.visual, 'vertex_colors', None)
            if vcol is not None and len(vcol) == len(mesh.vertices):
                colors = torch.tensor(vcol[:, :3], dtype=torch.float32) / 255.0
        except Exception:
            colors = None
        if colors is None:
            colors = torch.full((verts.shape[0], 3), 0.5, dtype=torch.float32)

        shape_mat = torch.cat([verts, delta, gamma, colors], dim=-1)
        coords = ((verts + 1.0) * 0.5 * float(self.ovoxel_resolution - 1)).round().to(torch.int32)
        coords = coords.clamp(0, self.ovoxel_resolution - 1)
        return shape_mat, coords

    def _fit_channels(self, feats: torch.Tensor, target_in_channels: int) -> torch.Tensor:
        if target_in_channels <= 0:
            return feats.new_zeros((feats.shape[0], 0))
        if feats.shape[1] > target_in_channels:
            return feats[:, :target_in_channels]
        if feats.shape[1] < target_in_channels:
            pad = feats.new_zeros((feats.shape[0], target_in_channels - feats.shape[1]))
            return torch.cat([feats, pad], dim=-1)
        return feats

    def _build_branch_features(self, shape_mat_features: torch.Tensor, feature_mode: str, target_in_channels: int):
        """Xây dựng các đặc trưng cụ thể cho từng nhánh từ bố cục bộ đệm hợp nhất [shape7, rgb3]."""
        mode = str(feature_mode or "shape_native").strip().lower()
        shape7 = shape_mat_features[:, :7]
        rgb3 = shape_mat_features[:, 7:10]

        if mode in {"none", "off", "disabled"}:
            feats = shape_mat_features.new_zeros((shape_mat_features.shape[0], 0))
        elif mode == "shape_native":
            feats = shape7
        elif mode == "shape_mat":
            feats = shape_mat_features[:, :10]
        elif mode == "rgb3":
            feats = rgb3
        elif mode == "rgb1":
            luma = 0.299 * rgb3[:, 0] + 0.587 * rgb3[:, 1] + 0.114 * rgb3[:, 2]
            feats = luma.unsqueeze(-1)
        elif mode == "mat6":
            raise ValueError("UNSUPPORTED: 'mat6' mode removed because face datasets lack PBR (metallic/roughness) data.")
        elif mode == "geom6":
            pseudo_normals = (shape7[:, 3:6] * 2.0 - 1.0).clamp(-1.0, 1.0)
            feats = torch.cat([shape7[:, :3], pseudo_normals], dim=-1)
        elif mode == "geom_mat12":
            raise ValueError("UNSUPPORTED: 'geom_mat12' mode removed because face datasets lack PBR (metallic/roughness) data.")
        else:
            feats = shape7

        return self._fit_channels(feats, int(target_in_channels))

    def _encode_latents(self, model: SC_VAE, feats: torch.Tensor, coords: torch.Tensor, target_in_channels: int):
        """Mã hóa các đặc trưng trên mỗi voxel (per-voxel features) bằng SC-VAE encoder, ưu tiên luồng thưa thớt (sparse path) khi có sẵn."""
        feats = self._fit_channels(feats, int(target_in_channels)).contiguous()
        if feats.shape[0] == 0:
            return torch.zeros((1, int(getattr(model, "latent_dim", self.latent_dim))), dtype=torch.float32)

        if (
            spconv is not None
            and coords is not None
            and coords.shape[0] == feats.shape[0]
            and coords.shape[1] == 3
        ):
            bcol = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
            sparse_indices = torch.cat([bcol, coords.to(torch.int32)], dim=1).to(self.device)
            sparse_feats = feats.to(self.device)
            sparse_input = spconv.SparseConvTensor(
                features=sparse_feats,
                indices=sparse_indices,
                spatial_shape=[self.ovoxel_resolution] * 3,
                batch_size=1,
            )
            mu, _ = model.encode(sparse_input)
            return mu.detach().cpu()

        mu, _ = model.encode(feats.to(self.device))
        return mu.detach().cpu()


def collate_slats(batch):
    """Đối chiếu (Collate) Slat tokens + các vector ngữ cảnh."""
    slats = torch.stack([item[0] for item in batch])    # [B, slat_length, latent_dim]
    contexts = torch.stack([item[1] for item in batch])  # [B, 946]
    return slats, contexts


def _resolve_material_config(imf_cfg) -> None:
    """Chuẩn hóa các tùy chọn vật liệu để tránh huấn luyện các kênh dư thừa."""
    mode = str(getattr(imf_cfg, "material_feature_mode", "rgb3")).strip().lower()
    mode_to_channels = {
        "rgb1": 1,
        "rgb3": 3,
        "mat6": 6,
        "geom_mat12": 12,
    }

    if mode in {"none", "off", "disabled"}:
        if bool(getattr(imf_cfg, "dual_branch", False)):
            print("[Config] material_feature_mode=none -> disabling dual-branch to skip material training.")
        imf_cfg.material_feature_mode = "none"
        imf_cfg.material_target_in_channels = 0
        imf_cfg.material_loss_weight = 0.0
        imf_cfg.material_condition_dropout = 0.0
        imf_cfg.dual_branch = False
        return

    if mode not in mode_to_channels:
        raise ValueError(
            f"Unsupported material_feature_mode={mode}. "
            "Use one of: none, rgb1, rgb3, mat6, geom_mat12."
        )

    expected_channels = int(mode_to_channels[mode])
    configured_channels = int(getattr(imf_cfg, "material_target_in_channels", expected_channels))
    if configured_channels != expected_channels:
        print(
            f"[Config] material_feature_mode={mode} expects {expected_channels} channels, "
            f"overriding material_target_in_channels {configured_channels} -> {expected_channels}."
        )
    imf_cfg.material_feature_mode = mode
    imf_cfg.material_target_in_channels = expected_channels


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(model, optimizer, scheduler, scaler, ema, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    if ema is not None:
        state['ema_state_dict'] = ema.state_dict()
    torch.save(state, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if ema and 'ema_state_dict' in ckpt:
        ema.load_state_dict(ckpt['ema_state_dict'])
    print(f"  ✅ Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt['epoch']


def get_lr_scheduler(optimizer, cfg, steps_per_epoch: int = 100):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.lr_warmup_steps)
    # T_max is the remaining steps after warmup
    total_steps = cfg.num_epochs * steps_per_epoch
    t_max = max(1, total_steps - cfg.lr_warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-7)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.lr_warmup_steps])


# ============================================================
# Training Loop
# ============================================================
def train_imf(
    cfg: TrainConfig,
    faceverse_train_ids_file: str = "train_faceverse_ids.txt",
    faceverse_test_ids_file: str = "test_faceverse_ids.txt",
    facescape_train_ids_file: str = "train_facescape_ids.txt",
    facescape_test_ids_file: str = "test_facescape_ids.txt",
    disable_id_filters: bool = False,
):
    """Vòng lặp huấn luyện chính (Main training loop) cho iMF U-Net."""
    
    device = torch.device(cfg.device)
    imf_cfg = cfg.imf
    _resolve_material_config(imf_cfg)
    
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    print("=" * 60)
    print("  FACEDIFF — STAGE 2: iMF U-Net TRAINING")
    print("=" * 60)
    cfg.print_summary()
    
    # ---- WandB ----
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or f"imf_unet_{time.strftime('%m%d_%H%M')}",
            tags=cfg.wandb.tags + ["imf_unet"],
            config={
                "stage": "imf_unet",
                "batch_size": imf_cfg.batch_size,
                "lr": imf_cfg.learning_rate,
                "epochs": imf_cfg.num_epochs,
                "hidden_dims": imf_cfg.hidden_dims,
                "context_dim": imf_cfg.context_dim,
                "ema_decay": imf_cfg.ema_decay,
            }
        )

    single_ckpt_path = imf_cfg.sc_vae_checkpoint
    shape_ckpt_path = imf_cfg.shape_sc_vae_checkpoint or single_ckpt_path
    material_ckpt_path = imf_cfg.material_sc_vae_checkpoint or single_ckpt_path
    sc_vae = None
    shape_sc_vae = None
    material_sc_vae = None
    
    # ---- Khởi tạo Thực Thể Giải Mã V4.1 Context ----
    if imf_cfg.use_precomputed_data:
        print("\n[1/5 & 2/5] ⚡ [OFFLINE MODE]: Bỏ qua nạp Extractors để tiết kiệm tối đa VRAM. Dataset sẽ chỉ đọc từ file .pt")
        renderer = None
        arcface = None
        flame = None
        dinov2 = None
    else:
        # ---- Tải SC-VAE đã được huấn luyện trước (Load Pretrained SC-VAE) ----
        if imf_cfg.dual_branch:
            print("\n[1/5] Loading dual SC-VAE checkpoints (shape + material)...")
            shape_sc_vae = SC_VAE(in_channels=imf_cfg.shape_target_in_channels, latent_dim=imf_cfg.input_dim, device=cfg.device).to(device)
            material_sc_vae = SC_VAE(in_channels=imf_cfg.material_target_in_channels, latent_dim=imf_cfg.input_dim, device=cfg.device).to(device)
            shape_sc_vae.eval()
            material_sc_vae.eval()

            if os.path.exists(shape_ckpt_path):
                ckpt = torch.load(shape_ckpt_path, map_location=device, weights_only=False)
                shape_sc_vae.load_state_dict(ckpt['model_state_dict'])
                print(f"  ✅ Loaded shape SC-VAE from {shape_ckpt_path}")
            else:
                print(f"  ⚠️ Shape SC-VAE checkpoint not found: {shape_ckpt_path}")

            if os.path.exists(material_ckpt_path):
                ckpt = torch.load(material_ckpt_path, map_location=device, weights_only=False)
                material_sc_vae.load_state_dict(ckpt['model_state_dict'])
                print(f"  ✅ Loaded material SC-VAE from {material_ckpt_path}")
            else:
                print(f"  ⚠️ Material SC-VAE checkpoint not found: {material_ckpt_path}")

            for param in shape_sc_vae.parameters():
                param.requires_grad = False
            for param in material_sc_vae.parameters():
                param.requires_grad = False

            sc_vae = None
        else:
            print("\n[1/5] Loading SC-VAE checkpoint...")
            sc_vae = SC_VAE(
                in_channels=imf_cfg.shape_target_in_channels,
                latent_dim=imf_cfg.input_dim,
                device=cfg.device,
            ).to(device)
            sc_vae.eval()

            if os.path.exists(single_ckpt_path):
                ckpt = torch.load(single_ckpt_path, map_location=device, weights_only=False)
                sc_vae.load_state_dict(ckpt['model_state_dict'])
                print(f"  ✅ Loaded SC-VAE from {single_ckpt_path}")
            else:
                print(f"  ⚠️ SC-VAE checkpoint not found: {single_ckpt_path}")

            for param in sc_vae.parameters():
                param.requires_grad = False
        
        # ---- Khởi tạo Thực Thể Giải Mã V4.1 Context ----
        print("\n[2/5] Initializing Hybrid Context Extractors...")
        renderer = MeshRenderer(device=cfg.device, image_size=512)
        arcface = ArcFaceExtractor(device=cfg.device)
        flame = FLAMEExpressionAdapter(expression_dim=50, device=cfg.device)
        dinov2 = DinoV3Extractor(model_name="facebook/dinov2-small", device=cfg.device)
    
    if imf_cfg.use_precomputed_data:
        shape_sc_vae = None
        material_sc_vae = None
    
    # ---- Dataset ----
    print("\n[3/5] Building Slat Dataset...")
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
        fv_dataset = SlatDataset(
            data_root=cfg.data.faceverse_root,
            sc_vae=sc_vae,
            dataset_name="faceverse",
            mesh_renderer=renderer,
            arcface=arcface,
            flame=flame,
            dinov2=dinov2,
            slat_length=imf_cfg.slat_length,
            latent_dim=imf_cfg.input_dim,
            cache_dir="data/slat_cache",
            device=cfg.device,
            include_ids=faceverse_include_ids,
            exclude_ids=faceverse_exclude_ids,
            dual_branch=imf_cfg.dual_branch,
            shape_sc_vae=shape_sc_vae if imf_cfg.dual_branch else None,
            material_sc_vae=material_sc_vae if imf_cfg.dual_branch else None,
            shape_feature_mode=imf_cfg.shape_feature_mode,
            material_feature_mode=imf_cfg.material_feature_mode,
            shape_target_in_channels=imf_cfg.shape_target_in_channels,
            material_target_in_channels=imf_cfg.material_target_in_channels,
            context_dim=imf_cfg.context_dim,
            single_sc_vae_checkpoint=single_ckpt_path if not imf_cfg.dual_branch else None,
            shape_sc_vae_checkpoint=shape_ckpt_path,
            material_sc_vae_checkpoint=material_ckpt_path,
            ovoxel_resolution=int(getattr(cfg.sc_vae, "ovoxel_resolution", 256)),
        )
        if len(fv_dataset) > 0:
            datasets_to_concat.append(fv_dataset)

    # Thêm FaceScape
    if cfg.data.active_dataset in ["facescape", "both"] and os.path.isdir(cfg.data.facescape_root):
        fs_dataset = SlatDataset(
            data_root=cfg.data.facescape_root,
            sc_vae=sc_vae,
            dataset_name="facescape",
            mesh_renderer=renderer,
            arcface=arcface,
            flame=flame,
            dinov2=dinov2,
            slat_length=imf_cfg.slat_length,
            latent_dim=imf_cfg.input_dim,
            cache_dir="data/slat_cache_facescape",
            device=cfg.device,
            include_ids=facescape_include_ids,
            exclude_ids=facescape_exclude_ids,
            dual_branch=imf_cfg.dual_branch,
            shape_sc_vae=shape_sc_vae if imf_cfg.dual_branch else None,
            material_sc_vae=material_sc_vae if imf_cfg.dual_branch else None,
            shape_feature_mode=imf_cfg.shape_feature_mode,
            material_feature_mode=imf_cfg.material_feature_mode,
            shape_target_in_channels=imf_cfg.shape_target_in_channels,
            material_target_in_channels=imf_cfg.material_target_in_channels,
            context_dim=imf_cfg.context_dim,
            single_sc_vae_checkpoint=single_ckpt_path if not imf_cfg.dual_branch else None,
            shape_sc_vae_checkpoint=shape_ckpt_path,
            material_sc_vae_checkpoint=material_ckpt_path,
            ovoxel_resolution=int(getattr(cfg.sc_vae, "ovoxel_resolution", 256)),
        )
        if len(fs_dataset) > 0:
            datasets_to_concat.append(fs_dataset)
            
    if not datasets_to_concat:
        raise ValueError(f"Không tìm thấy Mesh nào cho dataset {cfg.data.active_dataset}!")
        
    if len(datasets_to_concat) > 1:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets_to_concat)
    else:
        dataset = datasets_to_concat[0]
    
    num_workers = int(max(0, getattr(imf_cfg, "num_workers", cfg.data.num_workers)))
    pin_memory = bool(getattr(imf_cfg, "pin_memory", getattr(cfg.data, "pin_memory", True)) and device.type == "cuda")
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": imf_cfg.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_slats,
        "drop_last": True,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(max(1, getattr(imf_cfg, "prefetch_factor", getattr(cfg.data, "prefetch_factor", 2))))
        dataloader_kwargs["persistent_workers"] = bool(getattr(imf_cfg, "persistent_workers", getattr(cfg.data, "persistent_workers", True)))
    dataloader = DataLoader(**dataloader_kwargs)
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # ---- Model ----
    model_input_dim = imf_cfg.input_dim * 2 if imf_cfg.dual_branch else imf_cfg.input_dim
    
    if getattr(imf_cfg, "use_voxel_mamba", True):
        # Voxel Mamba v5.0 - O(N) complexity
        print("\n[4/5] Building Voxel Mamba v5.0...")
        from src.models.voxel_mamba import VoxelMamba
        model = VoxelMamba(
            input_dim=model_input_dim,
            hidden_dim=imf_cfg.mamba_hidden_dim,
            num_layers=imf_cfg.mamba_num_layers,
            slat_length=imf_cfg.slat_length,
            context_dim=imf_cfg.context_dim,
            backend=str(getattr(imf_cfg, "voxel_mamba_backend", "auto")),
            strict=bool(getattr(imf_cfg, "voxel_mamba_strict", False)),
            num_context_tokens=int(getattr(imf_cfg, "mamba_num_context_tokens", 8)),
            num_time_tokens=int(getattr(imf_cfg, "mamba_num_time_tokens", 4)),
            num_r_tokens=int(getattr(imf_cfg, "mamba_num_r_tokens", 4)),
            num_interval_tokens=int(getattr(imf_cfg, "mamba_num_interval_tokens", 4)),
            num_guidance_tokens=int(getattr(imf_cfg, "mamba_num_guidance_tokens", 4)),
            d_state=imf_cfg.mamba_d_state,
            d_conv=imf_cfg.mamba_d_conv,
            expand=imf_cfg.mamba_expand,
            dropout=imf_cfg.dropout,
        ).to(device)
        print(f"  Architecture: Voxel Mamba [D={imf_cfg.mamba_hidden_dim}, L={imf_cfg.mamba_num_layers}]")
        print(f"  Backend: {getattr(model, 'backend', 'unknown')}")
        print(f"  Complexity: O(N) linear scan (vs O(N²) attention)")
    else:
        # Hybrid U-DiT backbone — chỉ khi `imf.use_voxel_mamba=False` (checkpoint cũ).
        from src.models.generative_unet import IMFUNet1D

        print("\n[4/5] Building iMF U-Net v4.1...")
        model = IMFUNet1D(
            input_dim=model_input_dim,
            hidden_dims=imf_cfg.hidden_dims,
            context_dim=imf_cfg.context_dim,
            slat_length=imf_cfg.slat_length,
            num_bottleneck_layers=imf_cfg.num_bottleneck_layers,
        ).to(device)
        print(f"  Architecture: Hybrid U-DiT {imf_cfg.hidden_dims}")

    # ---- Compilation (RTX 4090 Optimization) ----
    # torch.compile tương thích với IMFUNet1D và VoxelMamba (GRU fallback).
    # Chỉ bỏ qua khi dùng mamba-ssm CUDA kernels (không tương thích).
    _can_compile = device.type == "cuda" and hasattr(torch, "compile")
    if _can_compile:
        _use_mamba_native = getattr(imf_cfg, "use_voxel_mamba", True) and getattr(model, "use_mamba", False)
        if _use_mamba_native:
            print("\n[4.5/5] Skipping torch.compile (mamba-ssm CUDA kernels không tương thích)")
        else:
            print("\n[4.5/5] Compiling model with torch.compile (reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
    
    imf = ImprovedMeanFlow(
        sigma_min=imf_cfg.sigma_min,
        ratio_r_neq_t=imf_cfg.ratio_r_neq_t,
        t_sampler=imf_cfg.t_sampler,
        t_loc=imf_cfg.t_loc,
        t_scale=imf_cfg.t_scale,
        curriculum_switch_ratio=imf_cfg.curriculum_switch_ratio,
        curriculum_uniform_prob=imf_cfg.curriculum_uniform_prob,
        cfg_omega_min=float(getattr(imf_cfg, "cfg_omega_min", 1.0)),
        cfg_omega_max=float(getattr(imf_cfg, "cfg_omega_max", 8.0)),
        cfg_omega_power_beta=float(getattr(imf_cfg, "cfg_omega_power_beta", 1.0)),
        enable_cfg_interval_conditioning=bool(getattr(imf_cfg, "cfg_interval_conditioning", True)),
        adaptive_loss_weighting=bool(getattr(imf_cfg, "adaptive_loss_weighting", True)),
    )
    
    # ---- Auxiliary v-head (for v-loss) ----
    v_head = None
    if getattr(imf_cfg, "use_v_loss", True) and getattr(imf_cfg, "use_auxiliary_v_head", True):
        print("  [v-loss] Adding auxiliary v-head...")
        v_head_dim = getattr(imf_cfg, "v_head_dim", 512)
        v_head = nn.Sequential(
            nn.Linear(model.hidden_dim if hasattr(model, 'hidden_dim') else imf_cfg.mamba_hidden_dim, v_head_dim),
            nn.SiLU(),
            nn.Linear(v_head_dim, model_input_dim)
        ).to(device)
        # Khởi tạo bằng không (Zero initialization)
        nn.init.zeros_(v_head[-1].weight)
        nn.init.zeros_(v_head[-1].bias)
        print(f"  [v-head] Hidden dim: {v_head_dim}, Output: {model_input_dim}")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if v_head is not None:
        param_count += sum(p.numel() for p in v_head.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    if imf_cfg.dual_branch:
        print(
            "  Dual-branch objective: "
            f"shape_dim={imf_cfg.input_dim}, material_dim={imf_cfg.input_dim}, "
            f"material_loss_weight={float(imf_cfg.material_loss_weight):.3f}"
        )
    if getattr(imf_cfg, "use_v_loss", True):
        print(f"  [v-loss] Enabled with boundary_ratio={getattr(imf_cfg, 'boundary_condition_ratio', 0.5)}")
    
    # EMA
    ema = EMA(model, decay=imf_cfg.ema_decay) if imf_cfg.use_ema else None
    if ema:
        print(f"  EMA: enabled (decay={imf_cfg.ema_decay})")
    
    # ---- Optimizer ----
    adamw_kwargs = {
        "lr": imf_cfg.learning_rate,
        "weight_decay": imf_cfg.weight_decay,
        "betas": (0.9, 0.999),
    }
    
    # Thu thập tất cả các tham số (Collect all parameters)
    all_params = list(model.parameters())
    if v_head is not None:
        all_params.extend(list(v_head.parameters()))
    
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(all_params, fused=True, **adamw_kwargs)
            print("  Optimizer: AdamW (fused=True)")
        except Exception:
            optimizer = torch.optim.AdamW(all_params, **adamw_kwargs)
            print("  Optimizer: AdamW (fused unavailable -> fallback)")
    else:
        optimizer = torch.optim.AdamW(all_params, **adamw_kwargs)
    scheduler = get_lr_scheduler(optimizer, imf_cfg, len(dataloader))
    scaler = torch.amp.GradScaler('cuda', enabled=imf_cfg.use_amp)
    
    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    
    if imf_cfg.resume_from and os.path.exists(imf_cfg.resume_from):
        if imf_cfg.resume_model_only:
            print("  [Resume] model-only mode: loading model weights and epoch, skipping optimizer/scheduler/scaler/ema states.")
        start_epoch = load_checkpoint(
            imf_cfg.resume_from,
            model,
            optimizer=None if imf_cfg.resume_model_only else optimizer,
            scheduler=None if imf_cfg.resume_model_only else scheduler,
            scaler=None if imf_cfg.resume_model_only else scaler,
            ema=None if imf_cfg.resume_model_only else ema,
        )
    
    # ---- Training ----
    print(f"\n[4/5] Training for {imf_cfg.num_epochs} epochs...")
    os.makedirs(imf_cfg.checkpoint_dir, exist_ok=True)
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, imf_cfg.num_epochs):
        if imf_cfg.num_epochs > 1:
            imf.set_progress(epoch / float(max(imf_cfg.num_epochs - 1, 1)))
        model.train()
        epoch_loss = 0.0
        epoch_shape_loss = 0.0
        epoch_material_loss = 0.0
        epoch_boundary_loss = 0.0
        epoch_jvp_loss = 0.0
        epoch_cfg_context_keep = 0.0
        t_start = time.time()
        
        for batch_idx, (slat_targets, contexts) in enumerate(dataloader):
            # slat_targets: [B, slat_length, latent_dim]
            # contexts: [B, 512]
            slat_targets = slat_targets.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=imf_cfg.use_amp):
                # Using the comprehensive compute_loss from ImprovedMeanFlow framework
                # This automatically handles JVP, CFG, dual-branch, and auxiliary v-head.
                loss_out = imf.compute_loss(
                    model,
                    slat_targets,
                    contexts,
                    v_head=v_head,
                    material_loss_weight=float(getattr(imf_cfg, "material_loss_weight", 1.0)),
                    dual_branch=bool(imf_cfg.dual_branch),
                    shape_dim=int(imf_cfg.input_dim),
                    material_condition_source=str(getattr(imf_cfg, "material_condition_source", "gt")),
                    material_condition_dropout=float(getattr(imf_cfg, "material_condition_dropout", 0.0)),
                    cfg_conditioning=bool(getattr(imf_cfg, "cfg_conditioning_enable", False)),
                    cfg_omega_min=float(getattr(imf_cfg, "cfg_omega_min", 1.0)),
                    cfg_omega_max=float(getattr(imf_cfg, "cfg_omega_max", 8.0)),
                    cfg_omega_power_beta=float(getattr(imf_cfg, "cfg_omega_power_beta", 1.0)),
                    cfg_interval_conditioning=bool(getattr(imf_cfg, "cfg_interval_conditioning", True)),
                    cfg_context_dropout=float(getattr(imf_cfg, "cfg_context_dropout", 0.1)),
                    return_components=True,
                )

                loss = loss_out["loss"]
                loss_shape_val = float(loss_out.get("loss_shape", 0.0))
                loss_material_val = float(loss_out.get("loss_material", 0.0))
                loss_boundary_val = float(loss_out.get("loss_boundary", 0.0))
                loss_jvp_val = float(loss_out.get("loss_jvp", 0.0))
                loss_v_head_val = float(loss_out.get("loss_v_head", 0.0))
                material_keep_val = float(loss_out.get("material_supervision_keep_ratio", 1.0))
                cfg_keep_val = float(loss_out.get("cfg_context_keep_ratio", 1.0))
            
            scaler.scale(loss).backward()
            
            if imf_cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                # Clip toàn bộ trainable params (model + v_head) — trước đây chỉ
                # clip `model.parameters()` nên gradient v-head không bị giới hạn,
                # có thể gây bất ổn khi auxiliary FM loss bùng phát.
                torch.nn.utils.clip_grad_norm_(all_params, imf_cfg.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # EMA update
            if ema:
                ema.update(model)
            
            epoch_loss += loss.item()
            if loss_shape_val is not None:
                epoch_shape_loss += loss_shape_val
                epoch_material_loss += loss_material_val
                epoch_boundary_loss += loss_boundary_val
                epoch_jvp_loss += loss_jvp_val
                epoch_cfg_context_keep += cfg_keep_val
            global_step += 1
            
            # WandB step logging
            if cfg.wandb.enabled and WANDB_AVAILABLE and global_step % cfg.wandb.log_every_steps == 0:
                step_payload = {
                    "train/velocity_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                }
                if loss_shape_val is not None:
                    step_payload.update({
                        "train/loss_shape": loss_shape_val,
                        "train/loss_material": loss_material_val,
                        "train/loss_boundary": loss_boundary_val,
                        "train/loss_jvp": loss_jvp_val,
                        "train/material_supervision_keep_ratio": material_keep_val,
                        "train/cfg_context_keep_ratio": cfg_keep_val,
                    })
                wandb.log(step_payload, step=global_step)
        
        # Thống kê Epoch (Epoch stats)
        n_batches = max(len(dataloader), 1)
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t_start
        vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
        
        print(f"  Epoch {epoch+1}/{imf_cfg.num_epochs} | "
              f"Velocity Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.1f}s | VRAM: {vram:.0f}MB")
        if imf_cfg.dual_branch:
            print(
                "    Branch losses | "
                f"shape: {epoch_shape_loss / n_batches:.4f} | "
                f"material: {epoch_material_loss / n_batches:.4f} | "
                f"boundary: {epoch_boundary_loss / n_batches:.4f} | "
                f"jvp: {epoch_jvp_loss / n_batches:.4f} | "
                f"cfg_ctx_keep: {epoch_cfg_context_keep / n_batches:.3f}"
            )
        
        # Ghi log epoch lên WandB
        if cfg.wandb.enabled and WANDB_AVAILABLE:
            epoch_payload = {
                "epoch/velocity_loss": avg_loss,
                "epoch/vram_peak_mb": vram,
            }
            if imf_cfg.dual_branch:
                epoch_payload.update({
                    "epoch/loss_shape": epoch_shape_loss / n_batches,
                    "epoch/loss_material": epoch_material_loss / n_batches,
                    "epoch/loss_boundary": epoch_boundary_loss / n_batches,
                    "epoch/loss_jvp": epoch_jvp_loss / n_batches,
                    "epoch/cfg_context_keep_ratio": epoch_cfg_context_keep / n_batches,
                })
            wandb.log(epoch_payload, step=global_step)
        
        # Lưu checkpoint
        if (epoch + 1) % imf_cfg.save_every_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, ema, epoch + 1, avg_loss,
                os.path.join(imf_cfg.checkpoint_dir, f"epoch_{epoch+1}.pt")
            )
        
        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, ema, epoch + 1, avg_loss,
                os.path.join(imf_cfg.checkpoint_dir, "best.pt")
            )
    
    # ---- Hoàn thành ----
    print(f"\n[5/5] Training complete! Best velocity loss: {best_loss:.4f}")
    print(f"  Checkpoints: {imf_cfg.checkpoint_dir}/")
    
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FaceDiff iMF U-Net Training (Stage 2)")
    parser.add_argument("--dataset", type=str, choices=["facescape", "faceverse", "both"], default=None, help="Chọn Dataset tải lên")
    parser.add_argument("--facescape-root", type=str, default=None, help="Override FaceScape data root")
    parser.add_argument("--faceverse-root", type=str, default=None, help="Override FaceVerse data root")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--resume-model-only", action="store_true", help="Resume only model weights/epoch; skip optimizer/scheduler/scaler/ema state")
    parser.add_argument("--sc-vae-ckpt", type=str, default=None, help="Path to SC-VAE checkpoint")
    parser.add_argument("--dual-branch", action="store_true", help="Use separate shape/material SC-VAE latents and concat them for iMF training")
    parser.add_argument("--shape-sc-vae-ckpt", type=str, default=None, help="Path to shape SC-VAE checkpoint")
    parser.add_argument("--material-sc-vae-ckpt", type=str, default=None, help="Path to material SC-VAE checkpoint")
    parser.add_argument("--shape-feature-mode", type=str, default=None, choices=["shape_native", "shape_mat", "geom6"], help="Feature mode for shape SC-VAE or unified SC-VAE path")
    parser.add_argument("--material-feature-mode", type=str, default=None, choices=["none", "rgb1", "rgb3", "mat6", "geom_mat12"], help="Feature mode for material SC-VAE when dual-branch is enabled")
    parser.add_argument("--shape-target-in-channels", type=int, default=None, help="Input channels for shape SC-VAE when dual-branch is enabled")
    parser.add_argument("--material-target-in-channels", type=int, default=None, help="Input channels for material SC-VAE when dual-branch is enabled")
    parser.add_argument("--disable-material-branch", action="store_true", help="Force shape-only mode: disable material branch and its loss.")
    parser.add_argument("--material-condition-source", type=str, default=None, choices=["gt", "pred_detached"], help="Material branch conditioning source for dual-branch mode")
    parser.add_argument("--material-condition-dropout", type=float, default=None, help="Dropout ratio for shape-conditioned material branch")
    parser.add_argument("--material-loss-weight", type=float, default=None, help="Loss weight for material branch objective")
    parser.add_argument("--enable-cfg-conditioning", action="store_true", help="Enable iMF flexible guidance conditioning (omega + interval).")
    parser.add_argument("--disable-cfg-conditioning", action="store_true", help="Disable iMF flexible guidance conditioning.")
    parser.add_argument("--cfg-omega-min", type=float, default=None, help="Lower bound of guidance scale omega (>=1).")
    parser.add_argument("--cfg-omega-max", type=float, default=None, help="Upper bound of guidance scale omega.")
    parser.add_argument("--cfg-omega-beta", type=float, default=None, help="Power-law beta for sampling omega with p(omega)~omega^-beta.")
    parser.add_argument("--cfg-context-dropout", type=float, default=None, help="Context dropout ratio for conditional branch during CFG-conditioning training.")
    parser.add_argument("--disable-cfg-interval-conditioning", action="store_true", help="Disable interval conditioning on [tmin, tmax].")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0 to avoid EGL crash over caches)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--offline-data", action="store_true", help="Kích hoạt chế độ tải Slat từ ổ cứng, bỏ qua Instantiate model Extractor để tiết kiệm VRAM.")
    parser.add_argument("--faceverse-train-ids", type=str, default="train_faceverse_ids.txt", help="Path to FaceVerse train IDs file")
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt", help="Path to FaceVerse test IDs file (excluded from train)")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt", help="Path to FaceScape train IDs file")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt", help="Path to FaceScape test IDs file (excluded from train)")
    parser.add_argument("--disable-id-filters", action="store_true", help="Disable train/test identity file filtering for custom datasets")
    args = parser.parse_args()
    
    cfg = TrainConfig()
    
    if args.dataset:
        cfg.data.active_dataset = args.dataset
    if args.facescape_root:
        cfg.data.facescape_root = args.facescape_root
    if args.faceverse_root:
        cfg.data.faceverse_root = args.faceverse_root
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.resume:
        cfg.imf.resume_from = args.resume
    if args.resume_model_only:
        cfg.imf.resume_model_only = True
    if args.sc_vae_ckpt:
        cfg.imf.sc_vae_checkpoint = args.sc_vae_ckpt
    if args.dual_branch:
        cfg.imf.dual_branch = True
    if args.shape_sc_vae_ckpt:
        cfg.imf.shape_sc_vae_checkpoint = args.shape_sc_vae_ckpt
    if args.material_sc_vae_ckpt:
        cfg.imf.material_sc_vae_checkpoint = args.material_sc_vae_ckpt
    if args.shape_feature_mode:
        cfg.imf.shape_feature_mode = args.shape_feature_mode
    if args.material_feature_mode:
        cfg.imf.material_feature_mode = args.material_feature_mode
    if args.shape_target_in_channels is not None:
        cfg.imf.shape_target_in_channels = max(1, int(args.shape_target_in_channels))
    if args.material_target_in_channels is not None:
        cfg.imf.material_target_in_channels = max(0, int(args.material_target_in_channels))
    if args.disable_material_branch:
        cfg.imf.material_feature_mode = "none"
        cfg.imf.material_target_in_channels = 0
        cfg.imf.material_loss_weight = 0.0
        cfg.imf.dual_branch = False
    if args.material_condition_source is not None:
        cfg.imf.material_condition_source = args.material_condition_source
    if args.material_condition_dropout is not None:
        cfg.imf.material_condition_dropout = float(max(0.0, min(1.0, args.material_condition_dropout)))
    if args.material_loss_weight is not None:
        cfg.imf.material_loss_weight = float(max(0.0, args.material_loss_weight))
    if args.enable_cfg_conditioning:
        cfg.imf.cfg_conditioning_enable = True
    if args.disable_cfg_conditioning:
        cfg.imf.cfg_conditioning_enable = False
    if args.cfg_omega_min is not None:
        cfg.imf.cfg_omega_min = float(max(1.0, args.cfg_omega_min))
    if args.cfg_omega_max is not None:
        cfg.imf.cfg_omega_max = float(max(cfg.imf.cfg_omega_min, args.cfg_omega_max))
    if args.cfg_omega_beta is not None:
        cfg.imf.cfg_omega_power_beta = float(max(0.0, args.cfg_omega_beta))
    if args.cfg_context_dropout is not None:
        cfg.imf.cfg_context_dropout = float(max(0.0, min(1.0, args.cfg_context_dropout)))
    if args.disable_cfg_interval_conditioning:
        cfg.imf.cfg_interval_conditioning = False
    if args.epochs:
        cfg.imf.num_epochs = args.epochs
    if args.batch_size:
        cfg.imf.batch_size = args.batch_size
    if args.lr:
        cfg.imf.learning_rate = args.lr
    if args.checkpoint_dir:
        cfg.imf.checkpoint_dir = args.checkpoint_dir
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers   # Cấu hình đè cực quan trọng chống EGL multiprocessing crash!
        
    if args.no_wandb:
        cfg.wandb.enabled = False
    if args.no_ema:
        cfg.imf.use_ema = False
    if args.offline_data:
        cfg.imf.use_precomputed_data = True
    
    train_imf(
        cfg,
        faceverse_train_ids_file=args.faceverse_train_ids,
        faceverse_test_ids_file=args.faceverse_test_ids,
        facescape_train_ids_file=args.facescape_train_ids,
        facescape_test_ids_file=args.facescape_test_ids,
        disable_id_filters=bool(args.disable_id_filters),
    )


if __name__ == "__main__":
    main()
