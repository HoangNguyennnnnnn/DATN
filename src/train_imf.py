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
import math
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Speedup flags (zero quality impact với bf16 autocast workflow):
# - TF32: ép fp32 matmul dùng tensor cores (chỉ ảnh hưởng fp32 ops ngoài autocast)
# - cudnn.benchmark: auto-tune kernel cho input shapes ổn định
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

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
    
    def load_state_dict(self, state_dict, device=None):
        # Merge (don't replace) so new params e.g. context_gate keep init shadow.
        for k, v in state_dict.items():
            t = v.clone()
            if device is not None:
                t = t.to(device)
            self.shadow[k] = t


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
    CACHE_SCHEMA_VERSION = 4
    _lmdb_env_cache: dict[str, object] = {}  # class-level: share envs across instances

    @classmethod
    def _open_lmdb_env(cls, lmdb_dir: str):
        """One LMDB env per path per process (FaceVerse + FaceScape share slat LMDB)."""
        import lmdb

        abs_dir = os.path.abspath(lmdb_dir)
        if abs_dir not in cls._lmdb_env_cache:
            cls._lmdb_env_cache[abs_dir] = lmdb.open(
                abs_dir,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
            )
        return cls._lmdb_env_cache[abs_dir]

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
                 ovoxel_resolution: int = 256,
                 allow_random_context_fallback: bool = False,
                 allow_mesh_proxy_fallback: bool = False,
                 context_lmdb_dir: str | None = None,
                 manifest_list: list[str] | None = None,
                 ovoxel_lmdb_dir: str | None = None,
                 ovoxel_lmdb_in_channels: int = 10,
                 ovoxel_lmdb_feature_mode: str = "shape_mat",
                 ovoxel_lmdb_max_voxels: int = 350000,
                 slat_lmdb_dir: str | None = None,
                 unique_identities: bool = False):
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
        self.allow_random_context_fallback = bool(allow_random_context_fallback)
        self.allow_mesh_proxy_fallback = bool(allow_mesh_proxy_fallback)
        self.ovoxel_converter = None
        self.unique_identities = bool(unique_identities)
        self.samples = []
        self.parent_pid = os.getpid()

        # LMDB cho Hybrid Context (Offline mode)
        self.context_lmdb_dir = context_lmdb_dir
        self.context_lmdb_env = None
        self.context_lmdb_txn = None

        if self.context_lmdb_dir and os.path.isdir(self.context_lmdb_dir):
            self.context_lmdb_env = self._open_lmdb_env(self.context_lmdb_dir)
            self.context_lmdb_txn = self.context_lmdb_env.begin(write=False)
            print(f"[SlatDataset] Connected to Hybrid Context LMDB: {self.context_lmdb_dir}")

        # LMDB cho O-Voxel data (dùng thay mesh files khi không có .obj trên server mới)
        self.ovoxel_lmdb_dir = ovoxel_lmdb_dir
        self.ovoxel_lmdb_env = None
        self.ovoxel_lmdb_txn = None
        self.ovoxel_lmdb_in_channels = int(ovoxel_lmdb_in_channels)
        self.ovoxel_lmdb_feature_mode = str(ovoxel_lmdb_feature_mode)
        self.ovoxel_lmdb_max_voxels = int(ovoxel_lmdb_max_voxels)

        if self.ovoxel_lmdb_dir and os.path.isdir(self.ovoxel_lmdb_dir):
            self.ovoxel_lmdb_env = self._open_lmdb_env(self.ovoxel_lmdb_dir)
            self.ovoxel_lmdb_txn = self.ovoxel_lmdb_env.begin(write=False)
            print(f"[SlatDataset] Connected to O-Voxel LMDB: {self.ovoxel_lmdb_dir}")

        # LMDB cho merged slat+context (pack_slat_lmdb.py output)
        self.slat_lmdb_dir = slat_lmdb_dir
        self.slat_lmdb_env = None
        self.slat_lmdb_txn = None

        if self.slat_lmdb_dir and os.path.isdir(self.slat_lmdb_dir):
            self.slat_lmdb_env = self._open_lmdb_env(self.slat_lmdb_dir)
            self.slat_lmdb_txn = self.slat_lmdb_env.begin(write=False)
            print(f"[SlatDataset] Connected to Slat+Context LMDB: {self.slat_lmdb_dir}")

        needs_encoder = (self.sc_vae is not None or
                         (self.dual_branch and (self.shape_sc_vae is not None or self.material_sc_vae is not None)))
        has_ovoxel_lmdb = self.ovoxel_lmdb_txn is not None
        if needs_encoder and not has_ovoxel_lmdb:
            try:
                self.ovoxel_converter = OVoxelConverter(
                    resolution=self.ovoxel_resolution,
                    device="cpu",
                )
            except Exception as exc:
                if self.allow_mesh_proxy_fallback:
                    warnings.warn(
                        f"[SlatDataset] O-Voxel converter init failed; using explicit debug mesh proxy fallback: {exc}",
                        RuntimeWarning,
                    )
                else:
                    raise RuntimeError(
                        "O-Voxel converter init failed and mesh-proxy fallback is disabled. "
                        "Enable debug fallback explicitly with allow_mesh_proxy_fallback=True."
                    ) from exc
        elif has_ovoxel_lmdb:
            print(f"[SlatDataset] Skipping OVoxelConverter init (using O-Voxel LMDB instead)")
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_contract = self._build_cache_contract()
        contract_blob = json.dumps(self.cache_contract, sort_keys=True, separators=(",", ":"))
        self.cache_tag = f"slatv{self.CACHE_SCHEMA_VERSION}_{hashlib.sha1(contract_blob.encode('utf-8')).hexdigest()[:12]}"
        print(f"[SlatDataset] Cache tag: {self.cache_tag}")
        
        # Quét các tệp mesh (hoặc dùng manifest nếu có)
        skipped_by_include = 0
        skipped_by_exclude = 0

        if manifest_list is not None:
            print(f"[SlatDataset] Using manifest for {self.dataset_name} ({len(manifest_list)} entries)")
            for rel_path in manifest_list:
                obj_path = os.path.join(data_root, rel_path)
                identity = extract_identity_from_obj_path(obj_path, self.data_root, self.dataset_name)
                if self.include_ids is not None and identity not in self.include_ids:
                    skipped_by_include += 1
                    continue
                if self.exclude_ids is not None and identity in self.exclude_ids:
                    skipped_by_exclude += 1
                    continue
                self.samples.append(obj_path)
        elif os.path.isdir(data_root):
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
        
        # FaceScape Unique Identities Filter: 1 neutral mesh per subject to avoid duplicate expressions
        if self.unique_identities and self.dataset_name == "facescape":
            by_id = {}
            for path in self.samples:
                identity = extract_identity_from_obj_path(path, self.data_root, self.dataset_name)
                if identity not in by_id:
                    by_id[identity] = []
                by_id[identity].append(path)
            
            filtered_samples = []
            for identity, paths in by_id.items():
                # Prefer "1_neutral.obj" if available
                neutral_path = None
                for p in paths:
                    if "1_neutral.obj" in p:
                        neutral_path = p
                        break
                if neutral_path is None:
                    for p in paths:
                        if "neutral" in p.lower():
                            neutral_path = p
                            break
                if neutral_path is None:
                    neutral_path = paths[0]
                filtered_samples.append(neutral_path)
            
            before_len = len(self.samples)
            self.samples = sorted(filtered_samples)
            print(f"[SlatDataset] Filtered FaceScape to unique identities only: {before_len} -> {len(self.samples)} samples")
        
        print(f"[SlatDataset] Found {len(self.samples)} meshes from {data_root}")
        if self.include_ids is not None or self.exclude_ids is not None:
            print(
                f"[SlatDataset] Filtered: include_skip={skipped_by_include}, "
                f"exclude_skip={skipped_by_exclude}"
            )

        # Offline mode: filter out samples missing from slat LMDB
        if self.slat_lmdb_txn is not None and self.sc_vae is None:
            before = len(self.samples)
            valid = []
            for obj_path in self.samples:
                rel_path = os.path.relpath(obj_path, self.data_root)
                lmdb_key = f"{self.dataset_name}/{rel_path}".encode("utf-8")
                if self.slat_lmdb_txn.get(lmdb_key) is not None:
                    valid.append(obj_path)
            self.samples = valid
            dropped = before - len(self.samples)
            if dropped > 0:
                print(f"[SlatDataset] Dropped {dropped} samples missing from slat LMDB ({dropped/before*100:.2f}%)")

        # Offline .pt cache: drop manifest entries with no precomputed slat file
        elif self.sc_vae is None:
            import glob as _glob
            before = len(self.samples)
            valid = []
            for obj_path in self.samples:
                rel_path = os.path.relpath(obj_path, self.data_root)
                base_name = rel_path.replace(os.path.sep, "_").replace(".obj", "")
                if _glob.glob(os.path.join(self.cache_dir, f"{base_name}.slatv3_*.pt")):
                    valid.append(obj_path)
            self.samples = valid
            dropped = before - len(self.samples)
            if dropped > 0:
                print(
                    f"[SlatDataset] Dropped {dropped} samples missing from .pt cache "
                    f"({dropped/before*100:.2f}%)"
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
        if self.allow_random_context_fallback:
            context_source_policy = "random_debug"
        else:
            context_source_policy = "hybrid_required"
        if self.ovoxel_lmdb_dir:
            ovoxel_source_policy = "ovoxel_lmdb"
        elif self.allow_mesh_proxy_fallback:
            ovoxel_source_policy = "mesh_proxy_debug"
        else:
            ovoxel_source_policy = "ovoxel_required"
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
            "context_source_policy": context_source_policy,
            "ovoxel_source_policy": ovoxel_source_policy,
            "allow_random_context_fallback": bool(self.allow_random_context_fallback),
            "allow_mesh_proxy_fallback": bool(self.allow_mesh_proxy_fallback),
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

    def _make_random_context(self, reason: str) -> torch.Tensor:
        if not self.allow_random_context_fallback:
            raise RuntimeError(
                f"{reason}. Hybrid context fallback is disabled; "
                "enable allow_random_context_fallback=True only for debug runs."
            )
        warnings.warn(
            f"[SlatDataset] Using explicit debug random-context fallback: {reason}",
            RuntimeWarning,
        )
        context = torch.randn(self.context_dim)
        return torch.nn.functional.normalize(context, p=2, dim=-1)

    def _check_and_init_lmdb_workers(self):
        """Đảm bảo các tiến trình con (dataloader workers) tự khởi tạo lại LMDB transactions riêng biệt."""
        current_pid = os.getpid()
        if current_pid != getattr(self, "_last_initialized_pid", self.parent_pid):
            self._last_initialized_pid = current_pid
            self.slat_lmdb_txn = None
            self.context_lmdb_txn = None
            self.ovoxel_lmdb_txn = None
            self.slat_lmdb_env = None
            self.context_lmdb_env = None
            self.ovoxel_lmdb_env = None

    @torch.no_grad()
    def __getitem__(self, idx: int):
        self._check_and_init_lmdb_workers()
        obj_path = self.samples[idx]

        # Tạo tên bộ đệm chống trùng lặp (vd: id125_1_neutral.pt)
        rel_path = os.path.relpath(obj_path, self.data_root)
        suffix = '.dual.pt' if self.dual_branch else '.pt'
        base_name = rel_path.replace(os.path.sep, '_').replace('.obj', '')
        safe_name = f"{base_name}.{self.cache_tag}{suffix}"
        cache_path = os.path.join(self.cache_dir, safe_name)

        # Priority 0: Merged slat+context LMDB (fastest path, cache_tag-independent)
        # Lazy re-open txn in forked DataLoader workers (LMDB txns don't survive fork)
        if self.slat_lmdb_dir and self.slat_lmdb_txn is None:
            self.slat_lmdb_env = self._open_lmdb_env(self.slat_lmdb_dir)
            self.slat_lmdb_txn = self.slat_lmdb_env.begin(write=False)
        if self.slat_lmdb_txn is not None:
            lmdb_key = f"{self.dataset_name}/{rel_path}".encode("utf-8")
            data = self.slat_lmdb_txn.get(lmdb_key)
            if data is not None:
                import io
                payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                return payload["slat"], payload["context"]

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

        # Offline fallback: SC-VAE checkpoint đổi → cache_tag khác nhưng cùng mesh stem
        if self.sc_vae is None:
            import glob as _glob
            alt_pattern = os.path.join(self.cache_dir, f"{base_name}.slatv3_*.pt")
            alt_matches = sorted(_glob.glob(alt_pattern), key=os.path.getmtime, reverse=True)
            if alt_matches:
                try:
                    cache_payload = torch.load(alt_matches[0], map_location="cpu", weights_only=False)
                    if "slat" in cache_payload and "context" in cache_payload:
                        return cache_payload["slat"], cache_payload["context"]
                except Exception:
                    pass
        
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

        # Nếu context trả về None (do offline mode), thử lấy từ LMDB
        if context is None and self.context_lmdb_dir:
            if self.context_lmdb_txn is None:
                self.context_lmdb_env = self._open_lmdb_env(self.context_lmdb_dir)
                self.context_lmdb_txn = self.context_lmdb_env.begin(write=False)
            
            if self.context_lmdb_txn is not None:
                rel_path_portable = os.path.relpath(obj_path, self.data_root)
                key = f"{self.dataset_name}/{rel_path_portable}".encode('utf-8')
                context_data = self.context_lmdb_txn.get(key)
                if context_data is not None:
                    import io
                    context = torch.load(io.BytesIO(context_data), map_location="cpu", weights_only=False).float()
                    if context.ndim == 0:
                        context = context.unsqueeze(0)

        if context is None:
            context = self._make_random_context(f"Context not found in LMDB or Extractor for {obj_path}")

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
                context = self._make_random_context(
                    f"Hybrid context extraction failed for {obj_path}: {e}"
                )
        else:
            # Trả về None để __getitem__ thử lấy từ LMDB
            context = None
        
        return slat, context

    def _ovoxel_lmdb_key(self, obj_path: str) -> str:
        """Tạo LMDB key khớp với format của pack_lmdb_fast / VoxelDataset."""
        rel = os.path.relpath(obj_path, self.data_root)
        c = self.ovoxel_lmdb_in_channels
        fm = self.ovoxel_lmdb_feature_mode
        mx = self.ovoxel_lmdb_max_voxels
        safe = rel.replace(os.sep, '_').replace('.obj', f'.c{c}.{fm}.mx{mx}.pt')
        return safe

    def _try_load_ovoxel_from_lmdb(self, obj_path: str):
        """Thử đọc O-Voxel payload từ LMDB. Trả về (features, coords) hoặc None."""
        self._check_and_init_lmdb_workers()
        if self.ovoxel_lmdb_dir and self.ovoxel_lmdb_txn is None:
            import lmdb as _lmdb
            self.ovoxel_lmdb_env = _lmdb.open(
                self.ovoxel_lmdb_dir, readonly=True, lock=False,
                readahead=True, meminit=False, max_readers=512,
            )
            self.ovoxel_lmdb_txn = self.ovoxel_lmdb_env.begin(write=False)
            
        if self.ovoxel_lmdb_txn is None:
            return None
        key = self._ovoxel_lmdb_key(obj_path)
        data = self.ovoxel_lmdb_txn.get(key.encode('utf-8'))
        if data is None:
            return None
        import io
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(payload, torch.Tensor):
            feats = payload.to(dtype=torch.float32)
            if feats.ndim == 2 and feats.shape[1] >= 10:
                return feats[:, :10].contiguous(), None
            return feats, None
        if isinstance(payload, dict):
            feats = payload["features"].to(dtype=torch.float32)
            coords = payload.get("coords", None)
            if coords is not None:
                coords = coords.to(dtype=torch.int32)
            if feats.ndim == 2 and feats.shape[1] >= 10:
                feats = feats[:, :10].contiguous()
            return feats, coords
        return None

    def _load_ovoxel_shape_mat(self, obj_path: str):
        """Tải payload O-Voxel 10 kênh hợp nhất [shape7, rgb3]."""
        # Ưu tiên đọc từ O-Voxel LMDB (cho server mới không có mesh files)
        lmdb_result = self._try_load_ovoxel_from_lmdb(obj_path)
        if lmdb_result is not None:
            return lmdb_result

        if self.ovoxel_converter is not None:
            try:
                payload = self.ovoxel_converter.process_mesh(obj_path)
                feats = torch.as_tensor(payload["shape_mat_features"], dtype=torch.float32)
                coords = torch.as_tensor(payload["coords"], dtype=torch.int32)
                if feats.ndim != 2 or feats.shape[1] < 10:
                    raise ValueError(f"Unexpected shape_mat_features shape: {tuple(feats.shape)}")
                return feats[:, :10].contiguous(), coords.contiguous()
            except Exception as exc:
                if not self.allow_mesh_proxy_fallback:
                    raise RuntimeError(
                        f"O-Voxel conversion failed for {obj_path} and mesh-proxy fallback is disabled: {exc}"
                    ) from exc
                warnings.warn(
                    f"[SlatDataset] O-Voxel conversion failed; using explicit debug mesh-proxy fallback for {obj_path}: {exc}",
                    RuntimeWarning,
                )
        elif not self.allow_mesh_proxy_fallback:
            raise RuntimeError(
                f"O-Voxel converter is unavailable for {obj_path} and mesh-proxy fallback is disabled."
            )

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
            mu, _, x_idx, x_shape = model.encode(sparse_input, return_indices=True)
            
            # Khôi phục tensor thưa thớt cho mu để gọi .dense()
            mu_sparse = spconv.SparseConvTensor(mu, x_idx, x_shape, 1)
            mu_dense = mu_sparse.dense() # [1, latent_dim, D, H, W]
            
            # Flatten thành [D*H*W, latent_dim] (ví dụ: [4096, 32])
            mu_flat = mu_dense.view(1, mu.shape[-1], -1).transpose(1, 2).squeeze(0)
            return mu_flat.detach().cpu()

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


def _build_stage2_model_config(model: nn.Module, imf_cfg, model_input_dim: int) -> dict:
    if str(getattr(imf_cfg, "backbone", "voxel_mamba")) == "unet3d":
        return {
            "arch": "unet3d",
            "input_dim": int(model_input_dim),
            "context_dim": int(getattr(imf_cfg, "context_dim", 946)),
            "slat_length": int(getattr(imf_cfg, "slat_length", 4096)),
            "slat_stats_path": getattr(imf_cfg, "slat_stats_path", None),
            "base": int(getattr(imf_cfg, "unet_base", 128)),
            "cond_dim": int(getattr(imf_cfg, "unet_cond_dim", 512)),
            "grid_size": int(round(int(getattr(imf_cfg, "slat_length", 4096)) ** (1.0 / 3.0))),
            "context_use_arcface_only": bool(getattr(imf_cfg, "context_use_arcface_only", False)),
            "num_ctx_tokens": int(getattr(model, "num_ctx_tokens", 16)),
            "context_whiten_path": getattr(imf_cfg, "context_whiten_path", None) or None,
        }
    return {
        "arch": "voxel_mamba",
        "input_dim": int(model_input_dim),
        "context_dim": int(getattr(imf_cfg, "context_dim", 946)),
        "slat_length": int(getattr(imf_cfg, "slat_length", 4096)),
        "slat_stats_path": getattr(imf_cfg, "slat_stats_path", None),
        "hidden_dim": int(getattr(model, "hidden_dim", getattr(imf_cfg, "mamba_hidden_dim", 512))),
        "num_layers": int(getattr(imf_cfg, "mamba_num_layers", 12)),
        "backend": str(getattr(model, "backend", getattr(imf_cfg, "voxel_mamba_backend", "auto"))),
        "strict": bool(getattr(imf_cfg, "voxel_mamba_strict", False)),
        "num_context_tokens": int(getattr(model, "num_context_tokens", getattr(imf_cfg, "mamba_num_context_tokens", 8))),
        "num_time_tokens": int(getattr(model, "num_time_tokens", getattr(imf_cfg, "mamba_num_time_tokens", 4))),
        "num_r_tokens": int(getattr(model, "num_r_tokens", getattr(imf_cfg, "mamba_num_r_tokens", 4))),
        "num_interval_tokens": int(getattr(model, "num_interval_tokens", getattr(imf_cfg, "mamba_num_interval_tokens", 4))),
        "num_guidance_tokens": int(getattr(model, "num_guidance_tokens", getattr(imf_cfg, "mamba_num_guidance_tokens", 4))),
        "use_per_layer_context": bool(getattr(model, "use_per_layer_context", getattr(imf_cfg, "mamba_use_per_layer_context", False))),
        "context_cond_mode": str(getattr(model, "context_cond_mode", getattr(imf_cfg, "context_cond_mode", "cross_attn"))),
        "context_use_arcface_only": bool(getattr(model, "context_use_arcface_only", getattr(imf_cfg, "context_use_arcface_only", True))),
        "num_context_kv_tokens": int(getattr(model, "num_context_kv_tokens", getattr(imf_cfg, "mamba_num_context_kv_tokens", 8))),
        "context_cross_attn_heads": int(getattr(model, "context_cross_attn_heads", getattr(imf_cfg, "mamba_context_cross_attn_heads", 8))),
        "conditioning": str(getattr(model, "context_cond_mode", "cross_attn")),
        "d_state": int(getattr(imf_cfg, "mamba_d_state", 16)),
        "d_conv": int(getattr(imf_cfg, "mamba_d_conv", 4)),
        "expand": int(getattr(imf_cfg, "mamba_expand", 2)),
        
        "dropout": float(getattr(imf_cfg, "dropout", 0.0)),
        "context_segment_weights": getattr(imf_cfg, "context_segment_weights", None),
    }


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    epoch,
    loss,
    path,
    *,
    v_head=None,
    ctx_classifier=None,
    stage2_model_config: dict | None = None,
    global_step: int | None = None,
    best_loss: float | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    loss_value = float(loss)
    if not np.isfinite(loss_value):
        raise FloatingPointError(f"Refusing to save checkpoint with non-finite loss={loss_value}: {path}")
    best_loss_value = float(best_loss if best_loss is not None else loss_value)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss_value,
        'global_step': int(global_step if global_step is not None else 0),
        'best_loss': best_loss_value,
    }
    if ema is not None:
        state['ema_state_dict'] = ema.state_dict()
    if v_head is not None:
        state['v_head_state_dict'] = v_head.state_dict()
    if ctx_classifier is not None:
        # FIX 2026-05-21 (Finding #3): save ctx_classifier for full-state resume
        state['ctx_classifier_state_dict'] = ctx_classifier.state_dict()
    if stage2_model_config is not None:
        state['stage2_model_config'] = dict(stage2_model_config)
    torch.save(state, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    v_head=None,
    ctx_classifier=None,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        gate_miss = [k for k in missing if "context_gate" in k]
        if gate_miss:
            print(f"  [Resume] New context_gate params ({len(gate_miss)}), default init=1.0")
        other = [k for k in missing if "context_gate" not in k]
        if other:
            print(f"  [Resume] Missing keys ({len(other)}): {other[:5]}{'...' if len(other) > 5 else ''}")
    if unexpected:
        print(f"  [Resume] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    downgrade_reasons = []

    ckpt_v_head_state = ckpt.get('v_head_state_dict')
    if v_head is not None:
        if isinstance(ckpt_v_head_state, dict):
            try:
                v_head.load_state_dict(ckpt_v_head_state, strict=True)
            except (RuntimeError, ValueError) as e:
                print(f"  [Resume] v-head shape mismatch → re-init fresh: {e}")
                downgrade_reasons.append("v-head architecture changed; re-init")
        else:
            downgrade_reasons.append("checkpoint missing v_head_state_dict")
    elif isinstance(ckpt_v_head_state, dict):
        downgrade_reasons.append("checkpoint contains v_head_state_dict but current run has no v_head")

    # FIX 2026-05-21 (Finding #3): load ctx_classifier
    ckpt_ctx_cls = ckpt.get('ctx_classifier_state_dict')
    if ctx_classifier is not None:
        if isinstance(ckpt_ctx_cls, dict):
            try:
                ctx_classifier.load_state_dict(ckpt_ctx_cls, strict=True)
            except (RuntimeError, ValueError) as e:
                print(f"  [Resume] ctx_classifier shape mismatch → re-init fresh: {e}")
                downgrade_reasons.append("ctx_classifier architecture changed; re-init")
        else:
            downgrade_reasons.append("checkpoint missing ctx_classifier_state_dict")
    elif isinstance(ckpt_ctx_cls, dict):
        downgrade_reasons.append("checkpoint contains ctx_classifier_state_dict but current run has no classifier")

    resumed_full = True
    if downgrade_reasons:
        resumed_full = False
    else:
        try:
            if optimizer and 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if scaler and ckpt.get('scaler_state_dict'):
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            if ema and 'ema_state_dict' in ckpt:
                ema_device = next(model.parameters()).device
                ema.load_state_dict(ckpt['ema_state_dict'], device=ema_device)
        except (ValueError, RuntimeError, KeyError) as exc:
            # KeyError '_schedulers': SequentialLR mới vs scheduler_state_dict cũ thiếu key.
            # → auto-downgrade: model+epoch giữ nguyên, scheduler thay ConstantLR (LR=1e-4
            # constant, không warmup lại), EMA re-init từ model weights. An toàn để resume tiếp.
            resumed_full = False
            downgrade_reasons.append(f"optimizer/scheduler state mismatch: {exc}")

    if resumed_full:
        print(f"  ✅ Resumed full stage-2 state from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    else:
        print(
            "  ⚠️ Auto-downgraded to model-only resume: "
            + "; ".join(downgrade_reasons)
        )
        print(f"  ✅ Loaded model weights from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    return {
        "epoch": int(ckpt.get('epoch', 0)),
        "loss": float(ckpt.get('loss', 0.0)),
        "best_loss": float(ckpt.get('best_loss', ckpt.get('loss', float('inf')))),
        "global_step": int(ckpt.get('global_step', 0)),
        "resumed_full": bool(resumed_full),
        "stage2_model_config": ckpt.get('stage2_model_config'),
    }


def get_lr_scheduler(optimizer, cfg, steps_per_epoch: int = 100):
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.lr_warmup_steps)
    if str(getattr(cfg, "lr_scheduler", "cosine")).lower() == "constant":
        # Paper iMF Table 4: lr schedule = constant (LR giữ base_lr sau warmup)
        main = ConstantLR(optimizer, factor=1.0, total_iters=10**9)
    else:
        total_steps = cfg.num_epochs * steps_per_epoch
        t_max = max(1, total_steps - cfg.lr_warmup_steps)
        main = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-7)
    return SequentialLR(optimizer, [warmup, main], milestones=[cfg.lr_warmup_steps])


def _set_optimizer_lrs(optimizer, base_lr: float):
    """Set LR without letting a partially reset scheduler re-ramp from the wrong value."""
    lrs = []
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr
        pg["initial_lr"] = base_lr
        lrs.append(base_lr)
    return lrs


def _reset_scheduler_lrs(scheduler, lrs):
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = list(lrs)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(lrs)
    for child in getattr(scheduler, "_schedulers", []):
        _reset_scheduler_lrs(child, lrs)


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
    context_lmdb_dir: str | None = None,
    ovoxel_lmdb_dir: str | None = None,
    slat_lmdb_dir: str | None = None,
    manifest_path: str | None = None,
):
    """Vòng lặp huấn luyện chính (Main training loop) cho iMF U-Net."""
    
    device = torch.device(cfg.device)
    imf_cfg = cfg.imf
    _resolve_material_config(imf_cfg)
    if bool(getattr(imf_cfg, "neg_guidance_enable", False)) or float(getattr(imf_cfg, "neg_guidance_scale", 0.0)) > 0.0:
        print("  [Config] Warning: negative guidance training fields are currently deprecated/unused in train_imf.")
    
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
    
    # ---- Nạp Manifest (nếu có) ----
    manifest_data = None
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        print(f"  ✅ Loaded mesh manifest from: {manifest_path}")

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
            allow_random_context_fallback=bool(getattr(imf_cfg, "allow_random_context_fallback", False)),
            allow_mesh_proxy_fallback=bool(getattr(imf_cfg, "allow_mesh_proxy_fallback", False)),
            context_lmdb_dir=context_lmdb_dir,
            ovoxel_lmdb_dir=ovoxel_lmdb_dir,
            slat_lmdb_dir=slat_lmdb_dir,
            manifest_list=manifest_data.get("faceverse") if manifest_data else None,
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
            allow_random_context_fallback=bool(getattr(imf_cfg, "allow_random_context_fallback", False)),
            allow_mesh_proxy_fallback=bool(getattr(imf_cfg, "allow_mesh_proxy_fallback", False)),
            context_lmdb_dir=context_lmdb_dir,
            ovoxel_lmdb_dir=ovoxel_lmdb_dir,
            slat_lmdb_dir=slat_lmdb_dir,
            manifest_list=manifest_data.get("facescape") if manifest_data else None,
            unique_identities=cfg.data.facescape_unique_identities,
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
    
    # ---- Model: backbone selection (voxel_mamba | unet3d) ----
    model_input_dim = imf_cfg.input_dim * 2 if imf_cfg.dual_branch else imf_cfg.input_dim
    _backbone = str(getattr(imf_cfg, "backbone", "voxel_mamba"))
    if _backbone == "unet3d":
        print("\n[4/5] Building 3D UNet backbone...")
        from src.models.unet3d import VoxelUNet3D
        model = VoxelUNet3D(
            input_dim=model_input_dim,
            context_dim=int(getattr(imf_cfg, "context_dim", 946)),
            base=int(getattr(imf_cfg, "unet_base", 128)),
            cond_dim=int(getattr(imf_cfg, "unet_cond_dim", 512)),
            grid_size=int(round(int(getattr(imf_cfg, "slat_length", 4096)) ** (1.0 / 3.0))),
            context_use_arcface_only=bool(getattr(imf_cfg, "context_use_arcface_only", False)),
            context_whiten_path=getattr(imf_cfg, "context_whiten_path", None) or None,
        ).to(device)
        if getattr(imf_cfg, "context_whiten_path", None):
            print(f"  [context] whitening ENABLED: {imf_cfg.context_whiten_path} → ctx_in={model._ctx_in}")
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Architecture: 3D UNet [base={getattr(imf_cfg, 'unet_base', 128)}, {nparam:.1f}M params]")
        stage2_model_config = _build_stage2_model_config(model, imf_cfg, model_input_dim)
    else:
        print("\n[4/5] Building Voxel Mamba v5.0...")
        from src.models.voxel_mamba import VoxelMamba
        seg_w = getattr(imf_cfg, "context_segment_weights", None)
        if seg_w is not None and len(seg_w) == 3:
            seg_w = tuple(float(x) for x in seg_w)
            print(f"  [context] segment weights Arc/FLAME/DINO = {seg_w}")
        ctx_mode = str(getattr(imf_cfg, "context_cond_mode", "cross_attn"))
        arc_only = bool(getattr(imf_cfg, "context_use_arcface_only", True))
        print(f"  [context] mode={ctx_mode}, arcface_only={arc_only}")
        model = VoxelMamba(
            input_dim=model_input_dim,
            hidden_dim=imf_cfg.mamba_hidden_dim,
            num_layers=imf_cfg.mamba_num_layers,
            slat_length=imf_cfg.slat_length,
            context_dim=imf_cfg.context_dim,
            backend=str(getattr(imf_cfg, "voxel_mamba_backend", "auto")),
            strict=bool(getattr(imf_cfg, "voxel_mamba_strict", False)),
            num_context_tokens=int(getattr(imf_cfg, "mamba_num_context_tokens", 0)),
            num_time_tokens=int(getattr(imf_cfg, "mamba_num_time_tokens", 4)),
            num_r_tokens=int(getattr(imf_cfg, "mamba_num_r_tokens", 4)),
            num_interval_tokens=int(getattr(imf_cfg, "mamba_num_interval_tokens", 4)),
            num_guidance_tokens=int(getattr(imf_cfg, "mamba_num_guidance_tokens", 4)),
            use_per_layer_context=bool(getattr(imf_cfg, "mamba_use_per_layer_context", False)),
            d_state=imf_cfg.mamba_d_state,
            d_conv=imf_cfg.mamba_d_conv,
            expand=imf_cfg.mamba_expand,
            ffn_expand=int(getattr(imf_cfg, "mamba_ffn_expand", 4)),
            dropout=imf_cfg.dropout,
            context_segment_weights=seg_w if not arc_only else None,
            context_cond_mode=ctx_mode,
            context_use_arcface_only=arc_only,
            num_context_kv_tokens=int(getattr(imf_cfg, "mamba_num_context_kv_tokens", 8)),
            context_cross_attn_heads=int(getattr(imf_cfg, "mamba_context_cross_attn_heads", 8)),
        ).to(device)
        print(f"  Architecture: Voxel Mamba [D={imf_cfg.mamba_hidden_dim}, L={imf_cfg.mamba_num_layers}]")
        print(f"  Backend: {getattr(model, 'backend', 'unknown')}")
        print(f"  Complexity: O(N) linear scan (vs O(N²) attention)")
        stage2_model_config = _build_stage2_model_config(model, imf_cfg, model_input_dim)

    # ---- Compilation (RTX 4090 Optimization) ----
    # Bỏ qua torch.compile khi dùng mamba-ssm CUDA kernels (không tương thích).
    _can_compile = device.type == "cuda" and hasattr(torch, "compile")
    if _can_compile and not getattr(model, "use_mamba", False) and _backbone != "unet3d":
        print("\n[4.5/5] Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("\n[4.5/5] Skipping torch.compile (mamba-ssm CUDA kernels không tương thích)")
    
    _paper_strict = os.environ.get("IMEFLOW_PAPER_STRICT", "").strip().lower() in ("1", "true", "yes")
    if not _paper_strict:
        _paper_strict = bool(getattr(imf_cfg, "paper_strict_tr", False))
    _adaptive_mode = os.environ.get("IMEFLOW_ADAPTIVE", "").strip().lower()
    if _adaptive_mode not in ("paper", "ema"):
        _adaptive_mode = str(getattr(imf_cfg, "adaptive_loss_mode", "ema")).strip().lower()
    if _adaptive_mode not in ("paper", "ema"):
        _adaptive_mode = "ema"
    # Env override cho adaptive_loss_weighting (paper iMF BẮT BUỘC để JVP không nổ:
    # loss/(loss+eps)^p tự chuẩn hóa → chặn blowup khi (t-r)*du/dt khổng lồ).
    _adaptive_on_env = os.environ.get("IMEFLOW_ADAPTIVE_ON", "").strip().lower()
    if _adaptive_on_env in ("1", "true", "yes"):
        _adaptive_weighting = True
    elif _adaptive_on_env in ("0", "false", "no"):
        _adaptive_weighting = False
    else:
        _adaptive_weighting = bool(getattr(imf_cfg, "adaptive_loss_weighting", True))

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
        adaptive_loss_weighting=_adaptive_weighting,
        paper_strict_tr=_paper_strict,
        adaptive_loss_mode=_adaptive_mode,
        norm_p=float(getattr(imf_cfg, "norm_p", 1.0)),
        norm_eps=float(getattr(imf_cfg, "norm_eps", 0.01)),
    )
    print(f"  [iMF] paper_strict_tr={_paper_strict}  adaptive_loss_mode={_adaptive_mode}")
    
    # ---- Auxiliary v-head (for v-loss) — Paper iMF Table 4: depth=8 ----
    v_head = None
    if getattr(imf_cfg, "use_v_loss", True) and getattr(imf_cfg, "use_auxiliary_v_head", True):
        print("  [v-loss] Adding auxiliary v-head...")
        if stage2_model_config["arch"] == "voxel_mamba":
            model_hidden_dim = int(stage2_model_config["hidden_dim"])
        else:
            model_hidden_dim = int(stage2_model_config["hidden_dims"][0])
        v_head_depth = int(getattr(imf_cfg, "v_head_depth", 8))
        v_head_mlp_ratio = int(getattr(imf_cfg, "v_head_mlp_ratio", 4))
        from src.models.v_head import VHead
        v_head = VHead(
            hidden_dim=model_hidden_dim,
            out_dim=model_input_dim,
            depth=v_head_depth,
            mlp_ratio=v_head_mlp_ratio,
        ).to(device)
        v_head_params = sum(p.numel() for p in v_head.parameters())
        print(f"  [v-head] depth={v_head_depth}, hidden={model_hidden_dim}, "
              f"mlp_ratio={v_head_mlp_ratio}, out={model_input_dim}, "
              f"params={v_head_params/1e6:.1f}M")

    # Contrastive context classifier (2026-05-20): Linear(hidden, context_dim)
    # Pools backbone hidden state across sequence → predicts context vector → InfoNCE loss.
    # Forces hidden state to encode discriminative identity info.
    ctx_classifier = None
    contrastive_weight = float(getattr(imf_cfg, "contrastive_loss_weight", 0.0))
    if contrastive_weight > 0.0:
        from src.models.imf_diffusion import contrastive_target_dim
        contrastive_mode = str(getattr(imf_cfg, "contrastive_mode", "arcface"))
        ctx_out_dim = contrastive_target_dim(int(imf_cfg.context_dim), contrastive_mode)
        if stage2_model_config["arch"] == "voxel_mamba":
            model_hidden_dim = int(stage2_model_config["hidden_dim"])
        else:
            model_hidden_dim = int(stage2_model_config["hidden_dims"][0])
        ctx_classifier = nn.Sequential(
            nn.Linear(model_hidden_dim, model_hidden_dim),
            nn.SiLU(),
            nn.Linear(model_hidden_dim, ctx_out_dim),
        ).to(device)
        n_ctx = sum(p.numel() for p in ctx_classifier.parameters())
        print(f"  [contrastive] mode={contrastive_mode} out_dim={ctx_out_dim} "
              f"params={n_ctx/1e6:.2f}M, weight={contrastive_weight:.2f}, "
              f"temp={float(getattr(imf_cfg, 'contrastive_temperature', 0.1)):.2f}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if v_head is not None:
        param_count += sum(p.numel() for p in v_head.parameters() if p.requires_grad)
    if ctx_classifier is not None:
        param_count += sum(p.numel() for p in ctx_classifier.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    if imf_cfg.dual_branch:
        print(
            "  Dual-branch objective: "
            f"shape_dim={imf_cfg.input_dim}, material_dim={imf_cfg.input_dim}, "
            f"material_loss_weight={float(imf_cfg.material_loss_weight):.3f}"
        )
    if getattr(imf_cfg, "use_v_loss", True):
        boundary_ratio = 1.0 - float(getattr(imf_cfg, "ratio_r_neq_t", 0.5))
        print(f"  [v-loss] Enabled with boundary_ratio={boundary_ratio:.3f}, weight={float(getattr(imf_cfg, 'v_loss_weight', 0.1)):.3f}")
    
    # EMA
    ema = EMA(model, decay=imf_cfg.ema_decay) if imf_cfg.use_ema else None
    if ema:
        print(f"  EMA: enabled (decay={imf_cfg.ema_decay})")
    
    # ---- Optimizer ----
    adamw_kwargs = {
        "lr": imf_cfg.learning_rate,
        "weight_decay": imf_cfg.weight_decay,
        # Paper iMF Table 4: Adam β=(0.9, 0.95). β2=0.95 phù hợp gradient noise
        # cao của diffusion-style training (varying t → varying loss scale).
        "betas": (0.9, 0.95),
    }
    
    optimizer_input = list(model.parameters())
    if v_head is not None:
        optimizer_input += list(v_head.parameters())
    if ctx_classifier is not None:
        optimizer_input += list(ctx_classifier.parameters())

    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(optimizer_input, fused=True, **adamw_kwargs)
            print("  Optimizer: AdamW (fused=True)")
        except Exception:
            optimizer = torch.optim.AdamW(optimizer_input, **adamw_kwargs)
            print("  Optimizer: AdamW (fused unavailable -> fallback)")
    else:
        optimizer = torch.optim.AdamW(optimizer_input, **adamw_kwargs)
    scheduler = get_lr_scheduler(optimizer, imf_cfg, len(dataloader))
    scaler = torch.amp.GradScaler('cuda', enabled=imf_cfg.use_amp)
    grad_accum_steps = max(1, int(getattr(imf_cfg, "gradient_accumulation_steps", 1)))
    optimizer_steps_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum_steps))
    effective_batch = imf_cfg.batch_size * grad_accum_steps
    print(f"  Gradient accumulation: {grad_accum_steps} steps (effective batch = {effective_batch})")

    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    
    if imf_cfg.resume_from and os.path.exists(imf_cfg.resume_from):
        if imf_cfg.resume_model_only:
            print("  [Resume] model-only mode: loading model weights and epoch, skipping optimizer/scheduler/scaler/ema states.")
        ckpt_info = load_checkpoint(
            imf_cfg.resume_from,
            model,
            optimizer=None if imf_cfg.resume_model_only else optimizer,
            scheduler=None if imf_cfg.resume_model_only else scheduler,
            scaler=None if imf_cfg.resume_model_only else scaler,
            ema=None if imf_cfg.resume_model_only else ema,
            v_head=v_head,
            ctx_classifier=ctx_classifier,
        )
        # BUG FIX: EMA.shadow được snapshot lúc tạo (TRƯỚC resume) = weights fresh-init.
        # model-only resume KHÔNG load ema state → shadow vẫn là fresh-init → với decay 0.9999
        # cần ~10000 step mới hồi phục, nên ema checkpoint = rác (context chết, sampling hỏng).
        # Re-snapshot shadow từ model đã load để EMA khởi đầu từ weights tốt.
        if imf_cfg.resume_model_only and ema is not None:
            ema.shadow = {
                name: param.clone().detach()
                for name, param in model.named_parameters() if param.requires_grad
            }
            print("  [Resume] EMA shadow re-initialized from loaded model weights (fix fresh-init garbage)")
        start_epoch = int(ckpt_info["epoch"])
        best_loss = float(ckpt_info["best_loss"])
        resume_full_state = (not imf_cfg.resume_model_only) and bool(ckpt_info.get("resumed_full", False))
        if ckpt_info.get("global_step", 0) > 0:
            global_step = int(ckpt_info["global_step"])
        else:
            global_step = start_epoch * optimizer_steps_per_epoch

        # Full resume can keep scheduler progress, but --lr still needs to override
        # restored base_lrs. Model-only/auto-downgrade has no valid scheduler state;
        # use constant LR immediately so warmup cannot restart at epoch 500+.
        new_lr = float(imf_cfg.learning_rate)
        group_lrs = _set_optimizer_lrs(optimizer, new_lr)
        if resume_full_state:
            _reset_scheduler_lrs(scheduler, group_lrs)
            print(f"  [Resume] Full-state LR forced to {new_lr:.2e} (scheduler state preserved)")
        else:
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=10**9)
            print(f"  [Resume] Model-only scheduler reset to constant LR={new_lr:.2e}")
    else:
        global_step = start_epoch * optimizer_steps_per_epoch
    
    # ---- Slat normalization stats (per-channel mean/std, TRELLIS.2-style) ----
    slat_norm_mean = None
    slat_norm_std = None
    slat_stats_path = getattr(imf_cfg, "slat_stats_path", None)
    if slat_stats_path and os.path.exists(slat_stats_path):
        _stats = torch.load(slat_stats_path, map_location="cpu", weights_only=False)
        slat_norm_mean = _stats["mean"].to(device).view(1, 1, -1).contiguous()  # [1, 1, 32]
        slat_norm_std = _stats["std"].to(device).view(1, 1, -1).contiguous()    # [1, 1, 32]
        print(f"[Slat Norm] Loaded {slat_stats_path}: "
              f"mean range [{slat_norm_mean.min().item():.4f}, {slat_norm_mean.max().item():.4f}], "
              f"std range [{slat_norm_std.min().item():.4f}, {slat_norm_std.max().item():.4f}]")
    else:
        print(f"[Slat Norm] DISABLED: slat_stats_path={slat_stats_path} not found. "
              f"Training without slat normalization (risk of identity collapse).")

    # ---- Variance-weighted loss (Hướng B): ánh mạnh voxel mang identity ----
    voxel_variance_weights = None
    _vv_path = os.environ.get("VOXEL_VARIANCE_PATH", "")
    _vv_mult = float(os.environ.get("VOXEL_VARIANCE_MULT", "4.0"))
    if _vv_path and os.path.exists(_vv_path):
        _vv = torch.load(_vv_path, map_location="cpu", weights_only=False)["voxel_variance"]
        _q75 = _vv.quantile(0.75)
        voxel_variance_weights = torch.where(_vv > _q75,
                                             torch.tensor(_vv_mult), torch.tensor(1.0)).to(device)
        print(f"[VarWeight] {_vv_path}: top25% voxel ×{_vv_mult} (q75={_q75:.4f}, "
              f"{int((_vv > _q75).sum())} voxels ánh mạnh)")

    # ---- Training ----
    print(f"\n[4/5] Training for {imf_cfg.num_epochs} epochs...")
    os.makedirs(imf_cfg.checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, imf_cfg.num_epochs):
        if imf_cfg.num_epochs > 1:
            imf.set_progress(epoch / float(max(imf_cfg.num_epochs - 1, 1)))
        model.train()
        epoch_loss = 0.0
        epoch_shape_loss = 0.0
        epoch_material_loss = 0.0
        epoch_boundary_loss = 0.0
        epoch_jvp_loss = 0.0
        epoch_context_sep_loss = 0.0
        epoch_ctx_sep_cos = 0.0
        epoch_cfg_context_keep = 0.0
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (slat_targets, contexts) in enumerate(dataloader):
            # slat_targets: [B, slat_length, latent_dim]
            # contexts: [B, 512]
            slat_targets = slat_targets.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)

            # FIX 2026-05-21 (Finding #2): compute occupancy mask BEFORE normalization.
            # After normalize, zero rows become (-mean/std), no longer ~0 → can't detect
            # padding. Mask must be extracted from raw slat where zero is truly zero.
            occupancy_mask = (slat_targets.norm(dim=-1) > 1e-6).float()  # [B, L]

            # Per-channel normalize (TRELLIS.2-style): (slat - mean) / std → tránh
            # identity collapse khi raw slat std=0.36 << noise std=1.0.
            if slat_norm_mean is not None and slat_norm_std is not None:
                slat_targets = (slat_targets - slat_norm_mean) / slat_norm_std
                # KEEP empty space EXACTLY 0.0 so the target diffusion velocity is exactly the noise
                slat_targets = slat_targets * occupancy_mask.unsqueeze(-1)

            is_accum_step = (batch_idx + 1) % grad_accum_steps != 0

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=imf_cfg.use_amp):
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
                    v_loss_weight=float(getattr(imf_cfg, "v_loss_weight", 0.1)),
                    ctx_classifier=ctx_classifier,
                    contrastive_loss_weight=float(getattr(imf_cfg, "contrastive_loss_weight", 0.0)),
                    contrastive_temperature=float(getattr(imf_cfg, "contrastive_temperature", 0.1)),
                    contrastive_mode=str(getattr(imf_cfg, "contrastive_mode", "arcface")),
                    context_velocity_sep_weight=float(getattr(imf_cfg, "context_velocity_sep_weight", 0.0)),
                    context_velocity_sep_margin=float(getattr(imf_cfg, "context_velocity_sep_margin", 0.0)),
                    occupancy_mask=occupancy_mask,
                    empty_weight_floor=float(os.environ.get("EMPTY_WEIGHT_FLOOR", "0.0")),
                    voxel_variance_weights=voxel_variance_weights,
                    prediction_type=os.environ.get("PREDICTION_TYPE", "velocity"),
                    occupancy_loss_weight=float(os.environ.get("OCCUPANCY_LOSS_WEIGHT", "0.0")),
                    return_components=True,
                )

                raw_loss = loss_out["loss"]
                if not torch.isfinite(raw_loss).all().item():
                    debug_parts = {}
                    for key in (
                        "loss",
                        "loss_boundary",
                        "loss_jvp",
                        "loss_v_head",
                        "loss_contrastive",
                        "loss_context_sep",
                        "ctx_sep_cos",
                    ):
                        if key not in loss_out:
                            continue
                        value = loss_out[key]
                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                debug_parts[key] = float(value.detach().float().cpu())
                            else:
                                finite = torch.isfinite(value).all().item()
                                debug_parts[key] = f"tensor{tuple(value.shape)} finite={finite}"
                        else:
                            debug_parts[key] = value
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite iMF loss at epoch={epoch + 1}, "
                        f"batch={batch_idx + 1}, global_step={global_step}: {debug_parts}"
                    )

                loss = raw_loss / grad_accum_steps
                loss_shape_val = float(loss_out.get("loss_shape", 0.0))
                loss_material_val = float(loss_out.get("loss_material", 0.0))
                loss_boundary_val = float(loss_out.get("loss_boundary", 0.0))
                loss_jvp_val = float(loss_out.get("loss_jvp", 0.0))
                loss_context_sep_val = float(loss_out.get("loss_context_sep", 0.0))
                ctx_sep_cos_val = float(loss_out.get("ctx_sep_cos", 0.0))
                loss_v_head_val = float(loss_out.get("loss_v_head", 0.0))
                material_keep_val = float(loss_out.get("material_supervision_keep_ratio", 1.0))
                cfg_keep_val = float(loss_out.get("cfg_context_keep_ratio", 1.0))

            scaler.scale(loss).backward()

            # Only step optimizer after accumulating grad_accum_steps
            if not is_accum_step or (batch_idx + 1) == len(dataloader):
                if imf_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimizer_input, imf_cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # EMA update
                if ema:
                    ema.update(model)

                global_step += 1

            batch_loss_val = loss.item() * grad_accum_steps
            epoch_loss += batch_loss_val
            if loss_shape_val is not None:
                epoch_shape_loss += loss_shape_val
                epoch_material_loss += loss_material_val
                epoch_boundary_loss += loss_boundary_val
                epoch_jvp_loss += loss_jvp_val
                epoch_context_sep_loss += loss_context_sep_val
                epoch_ctx_sep_cos += ctx_sep_cos_val
                epoch_cfg_context_keep += cfg_keep_val

            # 2026-05-22: Defensive empty_cache để giảm memory fragmentation (Bug C).
            # JVP path tạo allocation pattern không-chuẩn → PyTorch caching allocator
            # accumulate fragments → OOM sau ~1000 steps. Periodic release giúp tránh.
            if (batch_idx + 1) % 500 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 200 == 0 or batch_idx == 0:
                _total_batches = len(dataloader)
                elapsed_batch = time.time() - t_start
                batches_per_sec = (batch_idx + 1) / max(elapsed_batch, 1e-6)
                eta_epoch = (_total_batches - batch_idx - 1) / max(batches_per_sec, 1e-6)
                print(
                    f"    [{batch_idx+1}/{_total_batches}] loss={batch_loss_val:.4f} | "
                    f"{batches_per_sec:.1f} batch/s | ETA epoch: {eta_epoch/60:.1f}min",
                    flush=True,
                )

            if (not is_accum_step or (batch_idx + 1) == len(dataloader)) and imf_cfg.save_every_steps > 0 and global_step > 0 and global_step % int(imf_cfg.save_every_steps) == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    epoch + 1,
                    loss.item() * grad_accum_steps,
                    os.path.join(imf_cfg.checkpoint_dir, "latest_step.pt"),
                    v_head=v_head,
                    ctx_classifier=ctx_classifier,
                    stage2_model_config=stage2_model_config,
                    global_step=global_step,
                    best_loss=min(float(best_loss), float(loss.item() * grad_accum_steps)),
                )

            # WandB step logging (also guard behind optimizer step)
            if (not is_accum_step or (batch_idx + 1) == len(dataloader)) and cfg.wandb.enabled and WANDB_AVAILABLE and global_step > 0 and global_step % cfg.wandb.log_every_steps == 0:
                step_payload = {
                    "train/velocity_loss": loss.item() * grad_accum_steps,
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
        if not np.isfinite(avg_loss):
            raise FloatingPointError(f"Non-finite epoch loss at epoch={epoch + 1}: {avg_loss}")
        elapsed = time.time() - t_start
        vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
        
        sep_w = float(getattr(imf_cfg, "context_velocity_sep_weight", 0.0))
        print(f"  Epoch {epoch+1}/{imf_cfg.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"bnd: {epoch_boundary_loss / n_batches:.4f} | "
              f"ctx_sep: {epoch_context_sep_loss / n_batches:.4f} (w={sep_w}) | "
              f"cos_u: {epoch_ctx_sep_cos / n_batches:.3f} | "
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
                os.path.join(imf_cfg.checkpoint_dir, f"epoch_{epoch+1}.pt"),
                v_head=v_head,
                ctx_classifier=ctx_classifier,
                stage2_model_config=stage2_model_config,
                global_step=global_step,
                best_loss=best_loss,
            )
        
        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, ema, epoch + 1, avg_loss,
                os.path.join(imf_cfg.checkpoint_dir, "best.pt"),
                v_head=v_head,
                ctx_classifier=ctx_classifier,
                stage2_model_config=stage2_model_config,
                global_step=global_step,
                best_loss=best_loss,
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
    parser.add_argument("--context-cond-mode", type=str, default=None, choices=["cross_attn", "adaln"], help="Conditioning mode: cross_attn or adaln")
    parser.add_argument("--context-use-all", action="store_true", help="Use full 946-d hybrid context (ArcFace + FLAME + DINOv2)")
    parser.add_argument("--context-segment-weights", type=float, nargs=3, default=None, help="Weights for ArcFace, FLAME, DINOv2 segments (e.g. 1.5 1.0 0.5)")
    parser.add_argument("--facescape-unique-identities", action="store_true", default=False, help="Filter FaceScape to only load unique identities (1 neutral expression per subject)")
    parser.add_argument("--facescape-all-expressions", action="store_true", default=False, help="Tắt filter unique-ID: nạp TOÀN BỘ FaceScape (mọi biểu cảm) cho pretrain đa dạng.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size (micro-batch per GPU)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps (effective_batch = batch_size × this)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--contrastive-loss-weight", type=float, default=None, help="InfoNCE weight on context (0=off). Use ~0.2 after balanced context LMDB.")
    parser.add_argument("--contrastive-mode", type=str, default=None, choices=["arcface", "flame", "full"],
                        help="InfoNCE target block: arcface (512), flame (50), or full 946-d")
    parser.add_argument("--context-velocity-sep-weight", type=float, default=None,
                        help="relu(cos(u|ctx_a,u|ctx_b)-margin)^2 on shared z_t (>=0). Default 0.25.")
    parser.add_argument("--context-velocity-sep-margin", type=float, default=None,
                        help="Margin for context sep loss (penalize cos above this).")
    parser.add_argument("--ratio-r-neq-t", type=float, default=None, help="Fraction of batches with r≠t (JVP). 0.5 = paper default.")
    parser.add_argument("--backbone", type=str, default=None, choices=["voxel_mamba", "unet3d"], help="Stage 2 backbone.")
    parser.add_argument("--unet-base", type=int, default=None, help="Base channels for unet3d backbone.")
    parser.add_argument("--t-sampler", type=str, default=None, choices=["uniform", "logit_normal", "curriculum"], help="Timestep sampler.")
    parser.add_argument("--context-whiten", type=str, default=None, help="Path to context whitening .pt (PCA-whiten context).")
    parser.add_argument("--v-loss-weight", type=float, default=None, help="Auxiliary v-head loss weight (paper uses 1.0; default config 0.5).")
    parser.add_argument("--mamba-num-layers", type=int, default=None, help="BidirectionalMambaBlock count (lite: 8)")
    parser.add_argument("--mamba-ffn-expand", type=int, default=None, help="FFN expansion per block (lite: 2, default: 4)")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (default: config data.num_workers)")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Batches prefetched per worker (default: config data.prefetch_factor)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--offline-data", action="store_true", help="Kích hoạt chế độ tải Slat từ ổ cứng, bỏ qua Instantiate model Extractor để tiết kiệm VRAM.")
    parser.add_argument("--allow-random-context-fallback", action="store_true", help="Debug-only: allow random hybrid context fallback when extractors fail.")
    parser.add_argument("--allow-mesh-proxy-fallback", action="store_true", help="Debug-only: allow mesh-proxy fallback when O-Voxel conversion fails.")
    parser.add_argument("--faceverse-train-ids", type=str, default="train_faceverse_ids.txt", help="Path to FaceVerse train IDs file")
    parser.add_argument("--faceverse-test-ids", type=str, default="test_faceverse_ids.txt", help="Path to FaceVerse test IDs file (excluded from train)")
    parser.add_argument("--facescape-train-ids", type=str, default="train_facescape_ids.txt", help="Path to FaceScape train IDs file")
    parser.add_argument("--facescape-test-ids", type=str, default="test_facescape_ids.txt", help="Path to FaceScape test IDs file (excluded from train)")
    parser.add_argument("--disable-id-filters", action="store_true", help="Disable train/test identity file filtering for custom datasets")
    parser.add_argument("--context-lmdb", type=str, default=None, help="Path to hybrid_context.lmdb for offline context loading")
    parser.add_argument("--ovoxel-lmdb", type=str, default=None, help="Path to ovoxel_cache_lmdb for reading O-Voxel data without mesh files")
    parser.add_argument("--slat-lmdb", type=str, default=None, help="Path to slat_context.lmdb (merged slat+context, fastest loading)")
    parser.add_argument("--slat-stats", type=str, default=None, help="Path to slat stats .pt for normalization (combined vs faceverse per phase)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to mesh_manifest.json to avoid scanning .obj files")
    args = parser.parse_args()
    
    cfg = TrainConfig()
    
    if args.dataset:
        cfg.data.active_dataset = args.dataset
    if args.facescape_root:
        cfg.data.facescape_root = args.facescape_root
    if args.faceverse_root:
        cfg.data.faceverse_root = args.faceverse_root
    if args.facescape_unique_identities:
        cfg.data.facescape_unique_identities = True
    if args.facescape_all_expressions:
        cfg.data.facescape_unique_identities = False
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
    if args.context_cond_mode:
        cfg.imf.context_cond_mode = args.context_cond_mode
    if args.context_use_all:
        cfg.imf.context_use_arcface_only = False
    if args.context_segment_weights is not None:
        cfg.imf.context_segment_weights = tuple(args.context_segment_weights)
    if args.batch_size:
        cfg.imf.batch_size = args.batch_size
    if args.gradient_accumulation_steps:
        cfg.imf.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lr:
        cfg.imf.learning_rate = args.lr
    if args.checkpoint_dir:
        cfg.imf.checkpoint_dir = args.checkpoint_dir
    if args.contrastive_loss_weight is not None:
        cfg.imf.contrastive_loss_weight = float(max(0.0, args.contrastive_loss_weight))
    if args.contrastive_mode is not None:
        cfg.imf.contrastive_mode = str(args.contrastive_mode)
    if args.context_velocity_sep_weight is not None:
        cfg.imf.context_velocity_sep_weight = float(max(0.0, args.context_velocity_sep_weight))
    if args.context_velocity_sep_margin is not None:
        cfg.imf.context_velocity_sep_margin = float(args.context_velocity_sep_margin)
    if args.ratio_r_neq_t is not None:
        cfg.imf.ratio_r_neq_t = float(max(0.0, min(1.0, args.ratio_r_neq_t)))
    if args.slat_stats is not None:
        cfg.imf.slat_stats_path = str(args.slat_stats)
    if args.backbone is not None:
        cfg.imf.backbone = str(args.backbone)
    if args.unet_base is not None:
        cfg.imf.unet_base = int(args.unet_base)
    if args.t_sampler is not None:
        cfg.imf.t_sampler = str(args.t_sampler)
    if args.context_whiten is not None:
        cfg.imf.context_whiten_path = str(args.context_whiten)
    if args.v_loss_weight is not None:
        cfg.imf.v_loss_weight = float(max(0.0, args.v_loss_weight))
    if getattr(args, "mamba_num_layers", None) is not None:
        cfg.imf.mamba_num_layers = int(max(1, args.mamba_num_layers))
    if getattr(args, "mamba_ffn_expand", None) is not None:
        cfg.imf.mamba_ffn_expand = int(max(1, args.mamba_ffn_expand))
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
        cfg.imf.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        cfg.data.prefetch_factor = max(1, int(args.prefetch_factor))
        cfg.imf.prefetch_factor = cfg.data.prefetch_factor

    if args.no_wandb:
        cfg.wandb.enabled = False
    if args.no_ema:
        cfg.imf.use_ema = False
    if args.offline_data:
        cfg.imf.use_precomputed_data = True
    if args.allow_random_context_fallback:
        cfg.imf.allow_random_context_fallback = True
    if args.allow_mesh_proxy_fallback:
        cfg.imf.allow_mesh_proxy_fallback = True
    
    train_imf(
        cfg,
        faceverse_train_ids_file=args.faceverse_train_ids,
        faceverse_test_ids_file=args.faceverse_test_ids,
        facescape_train_ids_file=args.facescape_train_ids,
        facescape_test_ids_file=args.facescape_test_ids,
        disable_id_filters=bool(args.disable_id_filters),
        context_lmdb_dir=args.context_lmdb,
        ovoxel_lmdb_dir=args.ovoxel_lmdb,
        slat_lmdb_dir=args.slat_lmdb,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
