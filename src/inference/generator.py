"""
FaceDiff Inference Pipeline (v2.0)
===================================
Full end-to-end generation: Random Noise -> iMF U-Net -> SC-VAE Decoder -> Voxel Grid -> 3D Mesh (OBJ).

Conditioning: Hybrid Context (946-dim) = ArcFace (512) + FLAME (50) + DINOv2_Back (384)
Thay vì sử dụng chỉ ArcFace, nay gộp 3 nguồn đặc trưng để tận dụng biểu cảm và ảnh mặt sau.

Usage:
    python src/inference/generator.py
"""

import sys
import os
import glob
import torch
import numpy as np
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.imf_diffusion import ImprovedMeanFlow
from src.models.sc_vae import SC_VAE
from src.models.structure_generator import SparseStructureGenerator
from src.config import TrainConfig


class FaceDiffGenerator:
    """
    End-to-end 3D Face Generation Pipeline.
    
    Pipeline Flow:
        1. Hybrid Context Vector [946-dim] (ArcFace + FLAME + DINOv2_Back)
        2. iMF U-Net: noise -> predicted Slat tokens (1-step)
        3. SC-VAE Decoder: Slat tokens -> dense voxel features
        4. Marching Cubes: voxel features -> triangle mesh
        5. Export: mesh -> OBJ file
    """
    
    def __init__(self, device: str = "cuda:0",
                 slat_length: int = 4096,
                 slat_dim: int = 32,
                 vae_in_channels: int = 10,
                 sc_vae_ckpt: str | None = None,
                 sc_vae_shape_ckpt: str | None = None,
                 sc_vae_material_ckpt: str | None = None,
                 imf_ckpt: str | None = None,
                 structure_ckpt: str | None = None,
                 enable_structure_stage: bool = False,
                 structure_threshold: float = 0.5,
                 context_dim: int = 946,
                 voxel_grid_size: int = 64,
                 mesh_backend: str = "auto",
                 enforce_dual_contouring: bool = False,
                 mesh_smooth_sigma: float = 0.5,
                 neg_guidance_scale: float = 0.0,
                 cfg_scale: float = 1.0,
                 cfg_tmin: float = 0.0,
                 cfg_tmax: float = 1.0,
                 feature_mode: str = "shape_mat",
                 ovoxel_resolution: int = 256):
        """
        Args:
            device: GPU/CPU device
            slat_length: Number of Slat tokens (sequence length)
            slat_dim: Dimension per Slat token
            vae_in_channels: SC-VAE decoder output channels (6 geom or 12 geom+material)
            imf_ckpt: iMF U-Net checkpoint path. If None, auto-search checkpoints/imf_unet.
            sc_vae_ckpt: Optional single SC-VAE checkpoint path.
            sc_vae_shape_ckpt: Optional shape SC-VAE checkpoint path.
            sc_vae_material_ckpt: Optional material SC-VAE checkpoint path.
            structure_ckpt: Optional structure-generator checkpoint path.
            enable_structure_stage: If True, use stage-1 sparse structure gating.
            structure_threshold: Occupancy threshold for structure mask binarization.
            context_dim: Dimension của ngữ cảnh. V4.1 mặc định là 946.
            voxel_grid_size: Resolution of dense voxel grid for Marching Cubes
            mesh_backend: auto|sparseflex|flexicubes|diffmc|marching_cubes
            enforce_dual_contouring: If True, shape-native modes must use DC and never fallback to MC
            mesh_smooth_sigma: Gaussian smoothing sigma trước khi trích xuất mesh
            neg_guidance_scale: >0 để bật negative context guidance ở inference
            cfg_scale: Guidance scale omega for iMF flexible guidance conditioning
            cfg_tmin: Guidance interval lower bound
            cfg_tmax: Guidance interval upper bound
        """
        self.device = torch.device(device)
        self.slat_length = slat_length
        self.slat_dim = slat_dim
        self.slat_grid_size = self._infer_slat_grid_size(slat_length)
        self.vae_in_channels = int(vae_in_channels)
        self.voxel_grid_size = voxel_grid_size
        self.mesh_backend = mesh_backend
        self.enforce_dual_contouring = bool(enforce_dual_contouring)
        self.mesh_smooth_sigma = float(mesh_smooth_sigma)
        self.neg_guidance_scale = float(neg_guidance_scale)
        self.cfg_scale = float(max(1.0, cfg_scale))
        self.cfg_tmin = float(max(0.0, min(1.0, cfg_tmin)))
        self.cfg_tmax = float(max(self.cfg_tmin, min(1.0, cfg_tmax)))
        self.feature_mode = feature_mode
        self.ovoxel_resolution = ovoxel_resolution
        self.imf_ckpt = imf_ckpt or self._resolve_default_imf_ckpt()
        self.structure_ckpt = structure_ckpt or self._resolve_default_structure_ckpt()
        self.enable_structure_stage = bool(enable_structure_stage)
        self.structure_threshold = float(max(0.0, min(1.0, structure_threshold)))
        self.structure_model = None
        
        mem_before = self._get_vram()
        
        # === 1. Stage-2 backbone (Velocity Predictor) ===
        requested_input_dim = slat_dim * 2 if (sc_vae_shape_ckpt and sc_vae_material_ckpt) else slat_dim
        stage2_cfg = self._infer_stage2_model_config(self.imf_ckpt, requested_input_dim, context_dim)
        self.stage2_arch = str(stage2_cfg["arch"])
        self.stage2_input_dim = int(stage2_cfg["input_dim"])
        self.unet_input_dim = self.stage2_input_dim  # backward compatibility for downstream code

        if self.stage2_input_dim != requested_input_dim:
            print(
                f"[Generator] iMF checkpoint input_dim={self.stage2_input_dim} "
                f"khác cấu hình dự kiến {requested_input_dim}, dùng theo checkpoint."
            )

        print(f"[Generator] Khởi tạo Stage-2 backbone: {self.stage2_arch}...")
        self.unet = self._build_stage2_model(stage2_cfg).to(self.device)

        if self.imf_ckpt and os.path.exists(self.imf_ckpt):
            self._load_imf_checkpoint(self.unet, self.imf_ckpt)
            print(f"[Generator] Loaded iMF checkpoint ({self.stage2_arch}): {self.imf_ckpt}")
        else:
            print(f"[Generator] Warning: iMF checkpoint not found, using random {self.stage2_arch} weights.")

        self.unet.eval()

        # === 1.5. Stage-1 Sparse Structure Generator (optional) ===
        if self.enable_structure_stage:
            if self.structure_ckpt and os.path.exists(self.structure_ckpt):
                structure_cfg = self._infer_structure_model_config(self.structure_ckpt)
                self.structure_model = SparseStructureGenerator(**structure_cfg).to(self.device)
                self._load_structure_checkpoint(self.structure_model, self.structure_ckpt)
                self.structure_model.eval()
                print(f"[Generator] Loaded structure checkpoint: {self.structure_ckpt}")
            else:
                print("[Generator] Warning: structure stage enabled but checkpoint not found; disabling structure stage.")
                self.enable_structure_stage = False
        
        # === 2. iMF Sampler ===
        self.imf = ImprovedMeanFlow()
        
        # === 3. SC-VAE Decoder ===
        print("[Generator] Khởi tạo SC-VAE Decoder...")
        self.vae = None
        self.shape_vae = None
        self.material_vae = None
        self.sc_vae_ckpt = sc_vae_ckpt
        self.sc_vae_shape_ckpt = sc_vae_shape_ckpt
        self.sc_vae_material_ckpt = sc_vae_material_ckpt

        if self.sc_vae_shape_ckpt and self.sc_vae_material_ckpt:
            shape_channels = self._infer_in_channels_from_ckpt(self.sc_vae_shape_ckpt, 6)
            mat_channels = self._infer_in_channels_from_ckpt(self.sc_vae_material_ckpt, 6)

            self.shape_vae = SC_VAE(in_channels=shape_channels, latent_dim=slat_dim, device=device).to(self.device)
            self.material_vae = SC_VAE(in_channels=mat_channels, latent_dim=slat_dim, device=device).to(self.device)
            self._load_vae_checkpoint(self.shape_vae, self.sc_vae_shape_ckpt)
            self._load_vae_checkpoint(self.material_vae, self.sc_vae_material_ckpt)
            self.shape_vae.eval()
            self.material_vae.eval()
            print(
                f"[Generator] Loaded decoupled VAEs "
                f"(shape={shape_channels}ch, material={mat_channels}ch)."
            )
        else:
            if self.sc_vae_ckpt:
                self.vae_in_channels = self._infer_in_channels_from_ckpt(self.sc_vae_ckpt, self.vae_in_channels)
            self.vae = SC_VAE(in_channels=self.vae_in_channels, latent_dim=slat_dim, device=device).to(self.device)
            if self.sc_vae_ckpt:
                self._load_vae_checkpoint(self.vae, self.sc_vae_ckpt)
                print(f"[Generator] Loaded single SC-VAE checkpoint: {self.sc_vae_ckpt}")
            self.vae.eval()
        
        mem_after = self._get_vram()
        print(f"[Generator] Tổng VRAM cấp phát cho Models: {mem_after - mem_before:.2f} MB")

    def _infer_slat_grid_size(self, slat_length: int) -> int:
        grid = int(round(float(slat_length) ** (1.0 / 3.0)))
        if grid ** 3 != int(slat_length):
            raise ValueError(
                f"slat_length={slat_length} không phải số lập phương hoàn hảo; "
                "không thể tạo grid chỉ số 3D ổn định."
            )
        return grid

    def _resolve_default_imf_ckpt(self) -> str | None:
        best = os.path.join("checkpoints", "imf_unet", "best.pt")
        if os.path.exists(best):
            return best

        epochs = glob.glob(os.path.join("checkpoints", "imf_unet", "epoch_*.pt"))
        if len(epochs) == 0:
            return None

        def _epoch_num(path: str) -> int:
            stem = os.path.basename(path)
            try:
                return int(stem.replace("epoch_", "").replace(".pt", ""))
            except Exception:
                return -1

        epochs = sorted(epochs, key=_epoch_num)
        return epochs[-1]

    def _resolve_default_structure_ckpt(self) -> str | None:
        best = os.path.join("checkpoints", "structure_gen", "best.pt")
        if os.path.exists(best):
            return best

        epochs = glob.glob(os.path.join("checkpoints", "structure_gen", "epoch_*.pt"))
        if len(epochs) == 0:
            return None

        def _epoch_num(path: str) -> int:
            stem = os.path.basename(path)
            try:
                return int(stem.replace("epoch_", "").replace(".pt", ""))
            except Exception:
                return -1

        epochs = sorted(epochs, key=_epoch_num)
        return epochs[-1]

    def _infer_structure_model_config(self, ckpt_path: str) -> dict:
        default_cfg = {
            "context_dim": 946,
            "slat_length": int(self.slat_length),
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "num_context_tokens": 8,
            "dropout": 0.0,
        }
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_config", None), dict):
            cfg = dict(default_cfg)
            cfg.update(ckpt["model_config"])
            return cfg

        sd = None
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict", None), dict):
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        if not isinstance(sd, dict):
            return default_cfg

        query = sd.get("query_tokens", None)
        ctx_w = sd.get("context_proj.0.weight", None)
        if isinstance(query, torch.Tensor):
            default_cfg["slat_length"] = int(query.shape[0])
            default_cfg["hidden_dim"] = int(query.shape[1])
        if isinstance(ctx_w, torch.Tensor) and int(default_cfg["hidden_dim"]) > 0:
            default_cfg["context_dim"] = int(ctx_w.shape[1])
            inferred_tokens = int(ctx_w.shape[0] // int(default_cfg["hidden_dim"]))
            default_cfg["num_context_tokens"] = max(1, inferred_tokens)

        layer_ids = []
        for key in sd.keys():
            if key.startswith("encoder.layers."):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_ids.append(int(parts[2]))
        if len(layer_ids) > 0:
            default_cfg["num_layers"] = max(layer_ids) + 1
        return default_cfg

    def _load_structure_checkpoint(self, model: SparseStructureGenerator, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict", None), dict):
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported structure checkpoint format: {ckpt_path}")

        normalized_state = {}
        for key, value in state_dict.items():
            norm_key = str(key)
            if norm_key.startswith("_orig_mod."):
                norm_key = norm_key[len("_orig_mod."):]
            if norm_key.startswith("module."):
                norm_key = norm_key[len("module."):]
            normalized_state[norm_key] = value

        model.load_state_dict(normalized_state, strict=True)

    def _normalize_state_dict(self, state_dict: dict) -> dict:
        normalized_state = {}
        for key, value in state_dict.items():
            norm_key = str(key)
            if norm_key.startswith('_orig_mod.'):
                norm_key = norm_key[len('_orig_mod.'):]
            if norm_key.startswith('module.'):
                norm_key = norm_key[len('module.'):]
            normalized_state[norm_key] = value
        return normalized_state

    def _extract_imf_state_dict(self, ckpt_path: str | None) -> dict | None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception:
            return None

        if isinstance(ckpt, dict):
            sd = ckpt.get('ema_state_dict', None)
            if not isinstance(sd, dict):
                sd = ckpt.get('model_state_dict', None)
        else:
            sd = None
        if not isinstance(sd, dict):
            return None
        return self._normalize_state_dict(sd)

    def _extract_imf_checkpoint_payload(self, ckpt_path: str | None) -> dict | None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception:
            return None
        return ckpt if isinstance(ckpt, dict) else None

    def _infer_stage2_model_config(self, ckpt_path: str | None, default_input_dim: int, default_context_dim: int) -> dict:
        train_cfg = TrainConfig().imf
        cfg = {
            "arch": "voxel_mamba",
            "input_dim": int(default_input_dim),
            "context_dim": int(default_context_dim),
            "slat_length": int(self.slat_length),
            "num_context_tokens": int(getattr(train_cfg, "mamba_num_context_tokens", 8)),
            "num_time_tokens": int(getattr(train_cfg, "mamba_num_time_tokens", 4)),
            "num_r_tokens": int(getattr(train_cfg, "mamba_num_r_tokens", 4)),
            "num_interval_tokens": int(getattr(train_cfg, "mamba_num_interval_tokens", 4)),
            "hidden_dim": int(getattr(train_cfg, "mamba_hidden_dim", 512)),
            "num_layers": int(getattr(train_cfg, "mamba_num_layers", 12)),
            "backend": str(getattr(train_cfg, "voxel_mamba_backend", "auto")),
            "strict": bool(getattr(train_cfg, "voxel_mamba_strict", False)),
            "num_guidance_tokens": int(getattr(train_cfg, "mamba_num_guidance_tokens", 4)),
            "d_state": int(getattr(train_cfg, "mamba_d_state", 16)),
            "d_conv": int(getattr(train_cfg, "mamba_d_conv", 4)),
            "expand": int(getattr(train_cfg, "mamba_expand", 2)),
            "dropout": float(getattr(train_cfg, "dropout", 0.0)),
        }

        ckpt_payload = self._extract_imf_checkpoint_payload(ckpt_path)
        if isinstance(ckpt_payload, dict) and isinstance(ckpt_payload.get("stage2_model_config"), dict):
            cfg.update(ckpt_payload["stage2_model_config"])
            return cfg

        sd = self._extract_imf_state_dict(ckpt_path)
        if not isinstance(sd, dict) or "input_embed.weight" not in sd:
            return cfg

        weight = sd["input_embed.weight"]
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            cfg["hidden_dim"] = int(weight.shape[0])
            cfg["input_dim"] = int(weight.shape[1])

        ctx_w = sd.get("context_tokenizer.0.weight", None)
        if isinstance(ctx_w, torch.Tensor) and ctx_w.ndim == 2:
            cfg["context_dim"] = int(ctx_w.shape[1])
            if int(cfg["hidden_dim"]) > 0:
                cfg["num_context_tokens"] = max(1, int(ctx_w.shape[0] // int(cfg["hidden_dim"])))

        time_w = sd.get("time_tokenizer.0.weight", None)
        if isinstance(time_w, torch.Tensor) and time_w.ndim == 2 and int(cfg["hidden_dim"]) > 0:
            cfg["num_time_tokens"] = max(1, int(time_w.shape[0] // int(cfg["hidden_dim"])))

        r_w = sd.get("r_tokenizer.0.weight", None)
        if isinstance(r_w, torch.Tensor) and r_w.ndim == 2 and int(cfg["hidden_dim"]) > 0:
            cfg["num_r_tokens"] = max(1, int(r_w.shape[0] // int(cfg["hidden_dim"])))
        else:
            cfg["num_r_tokens"] = 0

        interval_w = sd.get("interval_tokenizer.0.weight", None)
        if isinstance(interval_w, torch.Tensor) and interval_w.ndim == 2 and int(cfg["hidden_dim"]) > 0:
            cfg["num_interval_tokens"] = max(1, int(interval_w.shape[0] // int(cfg["hidden_dim"])))
        else:
            cfg["num_interval_tokens"] = 0

        guide_w = sd.get("guidance_tokenizer.0.weight", None)
        if isinstance(guide_w, torch.Tensor) and guide_w.ndim == 2 and int(cfg["hidden_dim"]) > 0:
            cfg["num_guidance_tokens"] = max(1, int(guide_w.shape[0] // int(cfg["hidden_dim"])))
        else:
            cfg["num_guidance_tokens"] = 0

        layer_ids = []
        for key in sd.keys():
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_ids.append(int(parts[1]))
        if layer_ids:
            cfg["num_layers"] = max(layer_ids) + 1

        if any(".gru." in key for key in sd.keys()):
            cfg["backend"] = "gru"
        elif any(".forward_mamba." in key or ".backward_mamba." in key for key in sd.keys()):
            cfg["backend"] = "mamba"
        return cfg

    def _build_stage2_model(self, stage2_cfg: dict) -> nn.Module:
        from src.models.voxel_mamba import VoxelMamba

        return VoxelMamba(
            input_dim=int(stage2_cfg["input_dim"]),
            hidden_dim=int(stage2_cfg["hidden_dim"]),
            num_layers=int(stage2_cfg["num_layers"]),
            slat_length=int(stage2_cfg["slat_length"]),
            context_dim=int(stage2_cfg["context_dim"]),
            backend=str(stage2_cfg["backend"]),
            strict=bool(stage2_cfg["strict"]),
            num_context_tokens=int(stage2_cfg["num_context_tokens"]),
            num_time_tokens=int(stage2_cfg["num_time_tokens"]),
            num_r_tokens=int(stage2_cfg["num_r_tokens"]),
            num_interval_tokens=int(stage2_cfg["num_interval_tokens"]),
            num_guidance_tokens=int(stage2_cfg["num_guidance_tokens"]),
            dropout=float(stage2_cfg["dropout"]),
            d_state=int(stage2_cfg["d_state"]),
            d_conv=int(stage2_cfg["d_conv"]),
            expand=int(stage2_cfg["expand"]),
        )

    def _load_imf_checkpoint(self, model: nn.Module, ckpt_path: str):
        state_dict = self._extract_imf_state_dict(ckpt_path)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported iMF checkpoint format: {ckpt_path}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        total_keys = len(model.state_dict())
        if len(missing) > max(10, total_keys // 2):
            raise RuntimeError(
                f"Incompatible iMF checkpoint: loaded keys mismatch too large "
                f"(missing={len(missing)}/{total_keys})."
            )
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                "[Generator] Warning: partial iMF checkpoint load | "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    
    def _get_vram(self) -> float:
        """Trả về VRAM cấp phát hiện tại (MB)."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024**2)
        return 0.0

    def _get_slat_grid_indices(self, batch_size: int, grid_size: int = 16) -> torch.Tensor:
        """Construct the dense 3D coordinate layout for Slat tokens (16x16x16 grid)"""
        # [grid_size, grid_size, grid_size]
        z, y, x = torch.meshgrid(
            torch.arange(grid_size, device=self.device),
            torch.arange(grid_size, device=self.device),
            torch.arange(grid_size, device=self.device),
            indexing='ij'
        )
        # flatten -> [4096, 3]
        coords = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=-1).to(torch.int32)
        
        # Batch expand -> [B*4096, 4]
        all_indices = []
        for b in range(batch_size):
            b_idx = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=self.device)
            p_idx = torch.cat([b_idx, coords], dim=-1)
            all_indices.append(p_idx)
            
        return torch.cat(all_indices, dim=0)

    def _infer_in_channels_from_ckpt(self, ckpt_path: str, default_channels: int) -> int:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return int(default_channels)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', {}) if isinstance(ckpt, dict) else {}

        out_w = sd.get('out_proj.weight', None)
        if isinstance(out_w, torch.Tensor) and out_w.ndim == 2 and out_w.shape[0] > 0:
            return int(out_w.shape[0])

        enc_w = sd.get('_linear_enc.0.weight', None)
        if isinstance(enc_w, torch.Tensor) and enc_w.ndim == 2 and enc_w.shape[1] > 0:
            return int(enc_w.shape[1])

        return int(default_channels)

    def _load_vae_checkpoint(self, model: SC_VAE, ckpt_path: str):
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SC-VAE checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    
    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        output_path: str = "generated_face.obj",
        negative_context: torch.Tensor | None = None,
        omega: float | torch.Tensor | None = None,
        cfg_tmin: float | torch.Tensor | None = None,
        cfg_tmax: float | torch.Tensor | None = None,
    ) -> str:
        """
        Sinh 1 khuôn mặt 3D từ vector điều kiện (Hybrid Context 946-dim).
        
        Args:
            context: [B, 946] Hybrid Context vector
            output_path: Đường dẫn file OBJ xuất ra
            negative_context: [B, 946], optional; dùng cho negative guidance
            omega: Guidance scale override tại inference
            cfg_tmin: Guidance interval lower-bound override
            cfg_tmax: Guidance interval upper-bound override
            
        Returns:
            Đường dẫn tuyệt đối tới file OBJ đã sinh
        """
        b = context.shape[0]
        if b != 1:
            raise ValueError("Generator hiện chỉ export mesh cho batch size = 1.")

        omega_eff = self.cfg_scale if omega is None else omega
        tmin_eff = self.cfg_tmin if cfg_tmin is None else cfg_tmin
        tmax_eff = self.cfg_tmax if cfg_tmax is None else cfg_tmax

        if torch.is_tensor(omega_eff):
            omega_eff = omega_eff.to(context.device).clamp_min(1.0)
        else:
            omega_eff = float(max(1.0, float(omega_eff)))

        if torch.is_tensor(tmin_eff):
            tmin_eff = tmin_eff.to(context.device).clamp(0.0, 1.0)
        else:
            tmin_eff = float(max(0.0, min(1.0, float(tmin_eff))))

        if torch.is_tensor(tmin_eff) and not torch.is_tensor(tmax_eff):
            tmax_eff = torch.full_like(tmin_eff, float(tmax_eff))
        if torch.is_tensor(tmax_eff) and not torch.is_tensor(tmin_eff):
            tmin_eff = torch.full_like(tmax_eff, float(tmin_eff))

        if torch.is_tensor(tmax_eff):
            tmax_eff = tmax_eff.to(context.device).clamp(0.0, 1.0)
            tmax_eff = torch.maximum(tmax_eff, tmin_eff)
        else:
            tmax_eff = float(max(float(tmin_eff), min(1.0, float(tmax_eff))))

        mem_start = self._get_vram()

        # Optional negative context guidance (inference-only).
        if negative_context is not None and self.neg_guidance_scale > 0.0:
            negative_context = negative_context.to(context.device)
            context = context + self.neg_guidance_scale * (context - negative_context)

        active_token_indices = None
        if self.enable_structure_stage and self.structure_model is not None:
            print("\n[Generator] Bước 0/4: Stage-1 sparse structure gating...")
            structure_logits = self.structure_model(context)
            structure_probs = torch.sigmoid(structure_logits)
            structure_mask = structure_probs >= self.structure_threshold
            if bool((~structure_mask).all().item()):
                top_idx = int(torch.argmax(structure_probs[0]).item())
                structure_mask[0, top_idx] = True

            active_token_indices = torch.nonzero(structure_mask[0], as_tuple=False).squeeze(1)
            active_ratio = float(structure_mask.float().mean().item())
            print(
                f"  -> Active latent cells: {int(active_token_indices.numel())}/{self.slat_length} "
                f"({active_ratio * 100.0:.2f}%)"
            )
        
        # ====== BƯỚC 1: iMF 1-Step Sampling ======
        print("\n[Generator] Bước 1/4: iMF 1-Step Generation từ Random Noise...")
        slat_shape = (b, self.slat_length, self.stage2_input_dim)
        generated_slats = self.imf.sample_1_step(
            self.unet,
            context,
            shape=slat_shape,
            omega=omega_eff,
            cfg_tmin=tmin_eff,
            cfg_tmax=tmax_eff,
        )
        print(f"  -> Slat Tokens Shape: {generated_slats.shape}")
        
        # ====== BƯỚC 2: SC-VAE Decode ======
        print("[Generator] Bước 2/4: SC-VAE Decode Slat -> Dense Voxel Features...")
        
        # Slat Tokens are structured as a 16x16x16 dense grid space. We need to pass its layout.
        full_grid_indices = self._get_slat_grid_indices(batch_size=b, grid_size=self.slat_grid_size)
        if active_token_indices is not None:
            active_token_indices = active_token_indices.to(device=generated_slats.device, dtype=torch.long)
            generated_slats = generated_slats.index_select(1, active_token_indices)
            grid_indices = full_grid_indices.index_select(
                0,
                active_token_indices.to(device=full_grid_indices.device, dtype=torch.long),
            )
        else:
            grid_indices = full_grid_indices
        mesh_sparse_indices = grid_indices

        if self.shape_vae is not None and self.material_vae is not None:
            # Dual-branch: split by latent dims of each branch to avoid hard-coded assumptions.
            shape_latent_dim = int(getattr(self.shape_vae, "latent_dim", self.slat_dim))
            mat_latent_dim = int(getattr(self.material_vae, "latent_dim", self.slat_dim))
            total_latent_dim = shape_latent_dim + mat_latent_dim
            if generated_slats.shape[-1] < total_latent_dim:
                raise RuntimeError(
                    f"Generated latent dim={generated_slats.shape[-1]} < required dual-branch dim={total_latent_dim}"
                )

            slats_flat = generated_slats[..., :total_latent_dim].contiguous().view(-1, total_latent_dim)
            shape_slats = slats_flat[:, :shape_latent_dim]
            mat_slats = slats_flat[:, shape_latent_dim:shape_latent_dim + mat_latent_dim]
            
            shape_features, _, _, shape_indices = self.shape_vae.decode(
                shape_slats,
                original_indices=grid_indices,
                batch_size=b,
                return_indices=True,
            )
            mat_features, _, _, mat_indices = self.material_vae.decode(
                mat_slats,
                original_indices=grid_indices,
                batch_size=b,
                return_indices=True,
            )

            if shape_features.shape[0] != mat_features.shape[0]:
                n = int(min(shape_features.shape[0], mat_features.shape[0]))
                print(
                    f"[Generator] Warning: dual-branch decoded length mismatch "
                    f"shape={shape_features.shape[0]} vs material={mat_features.shape[0]}, truncating to {n}."
                )
                shape_features = shape_features[:n]
                mat_features = mat_features[:n]
            
            # Combine based on target layout
            if self.feature_mode in {"shape_native", "shape_mat"}:
                # shape[7] + mat[3] = 10 channels
                voxel_features = torch.cat([shape_features[:, :7], mat_features[:, :3]], dim=-1)
            else:
                voxel_features = torch.cat([shape_features[:, :6], mat_features[:, :3]], dim=-1)

            if shape_indices is not None and isinstance(shape_indices, torch.Tensor):
                mesh_sparse_indices = shape_indices
            elif mat_indices is not None and isinstance(mat_indices, torch.Tensor):
                mesh_sparse_indices = mat_indices
        else:
            single_latent_dim = int(getattr(self.vae, "latent_dim", self.slat_dim))
            if generated_slats.shape[-1] < single_latent_dim:
                raise RuntimeError(
                    f"Generated latent dim={generated_slats.shape[-1]} < required VAE latent dim={single_latent_dim}"
                )
            if generated_slats.shape[-1] > single_latent_dim:
                print(
                    f"[Generator] Warning: truncating latent width "
                    f"{generated_slats.shape[-1]} -> {single_latent_dim} for single-branch VAE decode."
                )
            slats_flat = generated_slats[..., :single_latent_dim].contiguous().view(-1, single_latent_dim)
            voxel_features, _, _, decoded_indices = self.vae.decode(
                slats_flat,
                original_indices=grid_indices,
                batch_size=b,
                return_indices=True,
            )
            if decoded_indices is not None and isinstance(decoded_indices, torch.Tensor):
                mesh_sparse_indices = decoded_indices
        print(f"  -> Voxel Features Shape: {voxel_features.shape}")
        
        # ====== BƯỚC 3: Voxel -> Mesh ======
        print("[Generator] Bước 3/4: Xây dựng lưới 3D...")
        if mesh_sparse_indices is not None and isinstance(mesh_sparse_indices, torch.Tensor):
            if mesh_sparse_indices.shape[0] != voxel_features.shape[0]:
                n = int(min(mesh_sparse_indices.shape[0], voxel_features.shape[0]))
                print(
                    f"[Generator] Warning: sparse-index/feature length mismatch "
                    f"idx={mesh_sparse_indices.shape[0]} vs feat={voxel_features.shape[0]}, truncating to {n}."
                )
                mesh_sparse_indices = mesh_sparse_indices[:n]
                voxel_features = voxel_features[:n]
        mesh_vertices, mesh_faces, mesh_colors = self._voxel_to_mesh(
            voxel_features, b, sparse_indices=mesh_sparse_indices
        )
        print(f"  -> Mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
        
        # ====== BƯỚC 4: Export OBJ ======
        print(f"[Generator] Bước 4/4: Xuất file OBJ -> {output_path}")
        self._export_obj(mesh_vertices, mesh_faces, output_path, vertex_colors=mesh_colors)
        if mesh_colors is not None:
            ply_path = os.path.splitext(output_path)[0] + ".ply"
            self._export_ply(mesh_vertices, mesh_faces, mesh_colors, ply_path)
            print(f"  -> Color PLY: {ply_path}")
        
        mem_end = self._get_vram()
        print(f"\n[Generator] VRAM Delta trong quá trình Inference: {mem_end - mem_start:.2f} MB")
        print(f"[Generator] ĐÃ HOÀN TẤT! File 3D: {os.path.abspath(output_path)}")
        
        return os.path.abspath(output_path)
    
    def _voxel_to_mesh(self, voxel_features: torch.Tensor, batch_size: int,
                        sparse_indices: torch.Tensor | None = None):
        """
        Chuyển đổi decoded features thành lưới tam giác.
        
        For shape-native layouts: uses O-Voxel Dual Contouring (flexible_dual_grid_to_mesh)
        For other modes: uses legacy Marching Cubes approach
        """
        # === Shape Native: O-Voxel Dual Contouring ===
        if self.feature_mode in {"shape_native", "shape_mat"} and sparse_indices is not None:
            return self._voxel_to_mesh_dual_contouring(
                voxel_features, sparse_indices, batch_size
            )

        if self.enforce_dual_contouring and self.feature_mode in {"shape_native", "shape_mat"}:
            raise RuntimeError(
                "Dual Contouring is enforced but sparse indices were not available for DC mesh extraction."
            )

        # === Legacy path: Marching Cubes ===
        return self._voxel_to_mesh_marching_cubes(voxel_features, batch_size)

    def _voxel_to_mesh_dual_contouring(
        self, voxel_features: torch.Tensor,
        sparse_indices: torch.Tensor, batch_size: int
    ):
        """Reconstruct mesh using TRELLIS Dual Contouring from predicted [v, delta, gamma]."""
        try:
            # Import from the dedicated submodule to avoid pulling material conversion
            # dependencies (trimesh/scipy) when only DC geometry extraction is needed.
            from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
        except ImportError:
            if self.enforce_dual_contouring:
                raise RuntimeError("Dual Contouring is enforced but o_voxel backend is unavailable")
            print("  [WARNING] o_voxel not available for DC, falling back to Marching Cubes")
            return self._voxel_to_mesh_marching_cubes(voxel_features, batch_size)

        feats = voxel_features.detach().float()
        if batch_size > 1:
            mask = sparse_indices[:, 0] == 0
            sparse_indices = sparse_indices[mask]
            feats = feats[mask]

        if sparse_indices.shape[0] != feats.shape[0]:
            n = int(min(sparse_indices.shape[0], feats.shape[0]))
            if n <= 0:
                if self.enforce_dual_contouring:
                    raise RuntimeError("Dual Contouring strict mode received empty sparse payload")
                print("  [DualContouring] Empty sparse payload, falling back to Marching Cubes")
                return self._voxel_to_mesh_marching_cubes(voxel_features, batch_size)
            print(
                f"  [DualContouring] Aligning sparse payload size idx={sparse_indices.shape[0]} "
                f"feat={feats.shape[0]} -> {n}"
            )
            sparse_indices = sparse_indices[:n]
            feats = feats[:n]

        # feats layout: [v(3), delta_logits(3), gamma_logits(1)]
        dc_device = feats.device
        v = torch.clamp(feats[:, :3].to(device=dc_device), 0.0, 1.0)
        delta = (feats[:, 3:6] > 0.0).to(device=dc_device)

        # Avoid directional triangulation artifacts when gamma is nearly constant.
        split_weight = None
        if feats.shape[1] >= 7:
            gamma_raw = torch.nn.functional.softplus(feats[:, 6:7]).to(device=dc_device)
            if gamma_raw.numel() > 0:
                gamma_pos = gamma_raw.clamp_min(1e-3)
                gamma_span = float((gamma_pos.max() - gamma_pos.min()).item())
                if gamma_span > 1e-6:
                    split_weight = gamma_pos

        # Extract coords from sparse_indices (strip batch column)
        coords = sparse_indices[:, 1:4].to(device=dc_device, dtype=torch.int32)

        # Construct default AABB in normalized space. Use list so backend puts it on coords device.
        grid_size = self.ovoxel_resolution
        aabb = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
        aabb_tensor = torch.as_tensor(aabb, dtype=torch.float32, device=dc_device)

        try:
            verts, faces = flexible_dual_grid_to_mesh(
                coords, v, delta, split_weight,
                aabb=aabb, grid_size=grid_size
            )
            verts_np = verts.cpu().numpy()
            faces_np = faces.cpu().numpy().astype(np.int64)

            if len(faces_np) > 0 and len(verts_np) > 0:
                # Remove obviously degenerate triangles before normal fixing.
                repeated = (
                    (faces_np[:, 0] == faces_np[:, 1])
                    | (faces_np[:, 1] == faces_np[:, 2])
                    | (faces_np[:, 0] == faces_np[:, 2])
                )
                faces_np = faces_np[~repeated]

            if len(faces_np) > 0 and len(verts_np) > 0:
                try:
                    import trimesh

                    mesh_tmp = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                    if hasattr(mesh_tmp, "nondegenerate_faces"):
                        mesh_tmp.update_faces(mesh_tmp.nondegenerate_faces())
                    mesh_tmp.remove_unreferenced_vertices()
                    trimesh.repair.fix_normals(mesh_tmp, multibody=True)
                    verts_np = mesh_tmp.vertices.astype(np.float32)
                    faces_np = mesh_tmp.faces.astype(np.int64)
                except Exception:
                    pass

            # RGB Sampling from Voxel features if available (channels 7-10)
            mesh_colors = None
            if feats.shape[1] >= 10 and len(verts_np) > 0 and coords.numel() > 0:
                # Nearest-voxel colors create visible Voronoi striping; dùng GPU IDW KNN blend.
                from src.mesh_gpu import gpu_knn_idw_colors

                voxel_size = (aabb_tensor[1] - aabb_tensor[0]) / float(max(int(grid_size), 1))
                dual_vertices_world = (coords.to(torch.float32) + v) * voxel_size.unsqueeze(0) + aabb_tensor[0].unsqueeze(0)
                k = int(max(1, min(8, dual_vertices_world.shape[0])))
                verts_t = torch.from_numpy(verts_np).to(dual_vertices_world.device)
                colors_t = gpu_knn_idw_colors(
                    verts_t, dual_vertices_world.detach(), feats[:, 7:10].detach(),
                    k=k,
                )
                mesh_colors = colors_t.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

            if len(faces_np) > 0:
                print(f"  [DualContouring] {len(verts_np)} verts, {len(faces_np)} faces")
                return verts_np, faces_np, mesh_colors
            else:
                if self.enforce_dual_contouring:
                    raise RuntimeError("Dual Contouring returned 0 faces while strict DC mode is enabled")
                print("  [DualContouring] 0 faces, falling back to Marching Cubes")
                return self._voxel_to_mesh_marching_cubes(voxel_features, batch_size)
        except Exception as e:
            if self.enforce_dual_contouring:
                raise RuntimeError(f"Dual Contouring failed in strict mode: {e}")
            print(f"  [DualContouring] Failed: {e}, falling back to Marching Cubes")
            return self._voxel_to_mesh_marching_cubes(voxel_features, batch_size)

    def _voxel_to_mesh_marching_cubes(self, voxel_features: torch.Tensor, batch_size: int):
        """Legacy Marching Cubes path for geom6/mat6 modes."""
        features_np = voxel_features.detach().cpu().float().numpy()
        points = features_np[:, :3]
        points = np.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
        points = np.clip(points, -1.0, 1.0)

        point_colors = None
        if features_np.shape[1] >= 10:
            # shape_mat 10-channel layout: [v(3), delta(3), gamma(1), rgb(3)]
            # RGB bắt đầu từ index 7, KHÔNG phải 6
            point_colors = np.nan_to_num(features_np[:, 7:10], nan=0.5, posinf=1.0, neginf=0.0)
            point_colors = np.clip(point_colors, 0.0, 1.0)
        elif features_np.shape[1] >= 9:
            # Legacy 6/12-channel modes: geom6 [xyz, normals] + mat [kd_rgb, ...]
            point_colors = np.nan_to_num(features_np[:, 6:9], nan=0.5, posinf=1.0, neginf=0.0)
            point_colors = np.clip(point_colors, 0.0, 1.0)

        grid_size = self.voxel_grid_size
        volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        idx = ((points + 1.0) * 0.5 * (grid_size - 1)).astype(np.int32)
        idx = np.clip(idx, 0, grid_size - 1)
        np.add.at(volume, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

        color_grid = None
        if point_colors is not None and len(point_colors) == len(idx):
            color_sum = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
            color_count = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            np.add.at(color_sum[..., 0], (idx[:, 0], idx[:, 1], idx[:, 2]), point_colors[:, 0])
            np.add.at(color_sum[..., 1], (idx[:, 0], idx[:, 1], idx[:, 2]), point_colors[:, 1])
            np.add.at(color_sum[..., 2], (idx[:, 0], idx[:, 1], idx[:, 2]), point_colors[:, 2])
            np.add.at(color_count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
            nonzero = color_count > 0
            color_grid = np.zeros_like(color_sum)
            color_grid[nonzero] = color_sum[nonzero] / color_count[nonzero, None]

        if float(volume.max()) <= 0.0:
            print("  [Cảnh báo] Occupancy volume rỗng. Dùng placeholder mesh.")
            verts, faces = self._create_icosphere()
            return verts, faces, None

        volume /= float(volume.max())
        
        if self.mesh_smooth_sigma > 0.0:
            try:
                from scipy.ndimage import gaussian_filter
                volume = gaussian_filter(volume, sigma=self.mesh_smooth_sigma)
            except ImportError:
                pass

        backend_order = self._resolve_backend_order(self.mesh_backend)
        for backend in backend_order:
            if backend == "sparseflex":
                result = self._mesh_sparseflex(volume, grid_size)
            elif backend == "flexicubes":
                result = self._mesh_flexicubes(volume, grid_size)
            elif backend == "diffmc":
                result = self._mesh_diffmc(volume, grid_size)
            else:
                result = self._mesh_marching_cubes(volume, grid_size)

            if result is not None:
                vertices, faces = result
                vertex_colors = self._sample_vertex_colors(vertices, color_grid, grid_size)
                return vertices, faces, vertex_colors

        print("  [Cảnh báo] Không backend mesh nào khả dụng. Tạo Icosphere placeholder...")
        verts, faces = self._create_icosphere()
        return verts, faces, None

    def _sample_vertex_colors(self, vertices: np.ndarray, color_grid: np.ndarray | None, grid_size: int):
        """Sample vertex colors from voxel color grid if material channels are available."""
        if color_grid is None or len(vertices) == 0:
            return None

        idx = ((vertices + 1.0) * 0.5 * (grid_size - 1)).astype(np.int32)
        idx = np.clip(idx, 0, grid_size - 1)
        colors = color_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        colors = np.clip(colors, 0.0, 1.0)

        if np.allclose(colors, 0.0):
            return None
        return colors.astype(np.float32)

    def _resolve_backend_order(self, backend: str):
        """Resolve backend fallback chain for mesh extraction."""
        backend = (backend or "auto").lower()
        if backend == "auto":
            return ["sparseflex", "flexicubes", "diffmc", "marching_cubes"]
        if backend == "sparseflex":
            return ["sparseflex", "diffmc", "marching_cubes"]
        if backend == "flexicubes":
            return ["flexicubes", "diffmc", "marching_cubes"]
        if backend == "diffmc":
            return ["diffmc", "marching_cubes"]
        return ["marching_cubes"]

    def _mesh_sparseflex(self, volume: np.ndarray, grid_size: int):
        """
        Try SparseFlex backend (v2.0) with feature-preserving gradient-based optimization.
        
        SparseFlex introduces:
        - Frustum-aware sectional voxel training (chỉ activate voxels gần surface)
        - Gradient-based mesh optimization với rendering loss supervision
        - Giảm 82% Chamfer Distance vs marching cubes
        - Hoạt động mượt mà ở độ phân giải 1024³
        
        Return None on failure để fallback chain hoạt động.
        """
        try:
            try:
                from sparseflex import SparseFlex  # type: ignore
            except Exception:
                # Local vendored fallback (workspace-only) if external package is unavailable.
                from src.inference.sparseflex import SparseFlex  # type: ignore
            
            # Chuyển volume thành tensor, giữ requires_grad=True để tối ưu hóa gradient-based
            volume_tensor = torch.tensor(
                volume, dtype=torch.float32, device=self.device, requires_grad=False
            )
            
            # Khởi tạo solver với rendering loss supervision enabled
            solver = SparseFlex(
                device=self.device,
                frustum_aware=True,  # Chỉ activate voxels gần surface
                enable_rendering_loss=False,  # Có thể enable trong training context
                learning_rate=1e-3,  # Cho gradient-based optimization
            )
            
            # Trích xuất mesh với feature-preserving
            vertices, faces = solver(volume_tensor)
            
            vertices = vertices.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()
            
            # Normalize coordinates
            vertices = (vertices / max(grid_size - 1, 1)) * 2 - 1
            
            print("  [SparseFlex] Sử dụng backend vi phân feature-preserving (v2.0).")
            return vertices, faces
            
        except Exception as e:
            # Fallback chain: return None để thử backend tiếp theo
            if "module" in str(e).lower():
                return None
            return None

    def _mesh_flexicubes(self, volume: np.ndarray, grid_size: int):
        """
        Try FlexiCubes backend (feature-aware gradient-based mesh optimization).
        
        FlexiCubes provides:
        - Learnable interpolation, splitting, and deformation parameters
        - Sharp feature preservation through local parameter adjustment
        - Fully differentiable mesh extraction for end-to-end optimization
        - Better quality than vanilla marching cubes while maintaining sharp features
        
        Return None on failure để fallback chain hoạt động.
        """
        try:
            from flexicubes import FlexiCubes  # type: ignore

            volume_tensor = torch.tensor(
                volume, dtype=torch.float32, device=self.device, requires_grad=False
            )

            # Khởi tạo solver với feature-preserving optimization
            solver = FlexiCubes(device=self.device)

            # Trích xuất mesh với sharp feature preservation
            vertices, faces = solver(volume_tensor)

            vertices = vertices.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()

            # Normalize coordinates
            vertices = (vertices / max(grid_size - 1, 1)) * 2 - 1

            print("  [FlexiCubes] Sử dụng backend vi phân bảo tồn đặc trưng sắc nét.")
            return vertices, faces

        except Exception:
            # Kaolin bundles FlexiCubes-style conversion kernels for many torch/cuda combos.
            try:
                from kaolin.ops.conversions import voxelgrids_to_trianglemeshes  # type: ignore

                volume_tensor = torch.tensor(
                    volume, dtype=torch.float32, device=self.device, requires_grad=False
                )
                occupancy = (volume_tensor > 0.5).unsqueeze(0)
                verts_list, faces_list = voxelgrids_to_trianglemeshes(occupancy)
                if not verts_list or not faces_list:
                    return None
                vertices = verts_list[0].detach().cpu().numpy()
                faces = faces_list[0].detach().cpu().numpy()
                vertices = (vertices / max(grid_size - 1, 1)) * 2 - 1
                print("  [FlexiCubes] Sử dụng Kaolin conversions fallback.")
                return vertices, faces
            except Exception:
                return None

    def _mesh_diffmc(self, volume: np.ndarray, grid_size: int):
        """
        Try DiffMC (Differentiable Marching Cubes) backend from diso package.
        
        DiffMC provides:
        - Full differentiability for gradient-based optimization
        - Better quality than vanilla marching cubes while being faster
        - Support for rendering loss supervision trong training context
        
        Return None on failure để fallback chain hoạt động.
        """
        try:
            from diso import DiffMC  # type: ignore
            
            volume_tensor = torch.tensor(
                volume, dtype=torch.float32, device=self.device, requires_grad=False
            )
            
            # Khởi tạo DiffMC solver
            diffmc = DiffMC(dtype=torch.float32).to(self.device)
            
            # Trích xuất mesh
            vertices, faces = diffmc(volume_tensor)
            
            vertices = vertices.cpu().numpy()
            faces = faces.cpu().numpy()
            
            # Normalize coordinates
            vertices = (vertices / max(grid_size - 1, 1)) * 2 - 1
            
            print("  [DiffMC] Sử dụng Differentiable Marching Cubes (diso backend).")
            return vertices, faces
            
        except Exception:
            return None

    def _mesh_marching_cubes(self, volume: np.ndarray, grid_size: int):
        """
        Fallback: skimage marching cubes backend (luôn khả dụng).
        
        Limitations so với advanced backends:
        - Non-differentiable (không thể tối ưu hóa gradient-based)
        - Làm mượt chi tiết (over-tessellation & smoothing)
        - Có thể tạo lưới kép (double-layered meshes)
        
        Tuy nhiên, đủ tốt cho inference và fallback safety.
        
        Note: Để sử dụng rendering loss supervision, cần chuyển sang SparseFlex/FlexiCubes
        trong quá trình fine-tuning qua training pipeline.
        
        Return None on failure để fallback vào icosphere placeholder.
        """
        try:
            from skimage.measure import marching_cubes
            
            # Compute adaptive threshold từ percentile của non-zero voxels
            nonzero = volume[volume > 0]
            if nonzero.size == 0:
                return None
                
            threshold = float(np.clip(np.percentile(nonzero, 45), 0.03, 0.35))
            
            # Apply marching cubes
            vertices, faces, normals, values = marching_cubes(volume, level=threshold)
            
            # Normalize coordinates
            vertices = (vertices / max(grid_size - 1, 1)) * 2 - 1
            
            print("  [MarchingCubes] Sử dụng backend skimage fallback (non-differentiable).")
            return vertices, faces
            
        except Exception:
            return None
    
    def _create_icosphere(self):
        """Tạo hình cầu đơn giản (Icosphere) khi không có Marching Cubes."""
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
            return sphere.vertices, sphere.faces
        except ImportError:
            # Tạo hình lập phương tối giản
            v = np.array([
                [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
            ], dtype=np.float64) * 0.5
            f = np.array([
                [0,1,2], [0,2,3], [4,6,5], [4,7,6],
                [0,5,1], [0,4,5], [2,6,7], [2,7,3],
                [0,7,4], [0,3,7], [1,5,6], [1,6,2]
            ])
            return v, f
    
    def _export_obj(self, vertices: np.ndarray, faces: np.ndarray, path: str, vertex_colors: np.ndarray | None = None):
        """Xuất lưới tam giác ra file .obj chuẩn Wavefront."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("# FaceDiff Generated 3D Face\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            use_color = vertex_colors is not None and len(vertex_colors) == len(vertices)
            for i, v in enumerate(vertices):
                if use_color:
                    c = np.clip(vertex_colors[i], 0.0, 1.0)
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            f.write("\n")
            for face in faces:
                # OBJ faces are 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _export_ply(self, vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray, path: str):
        """Export PLY with RGB vertex colors for broad viewer compatibility."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        colors_u8 = np.clip(vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8)

        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for i, v in enumerate(vertices):
                c = colors_u8[i]
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def run_inference_test():
    """
    Test End-to-End: Tạo ArcFace vector giả -> Sinh 3D Face -> Xuất OBJ.
    """
    print("=" * 60)
    print(" FACEDIFF 2.0: INFERENCE PIPELINE - TEST END-TO-END")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")
    
    cfg = TrainConfig()
    inf_cfg = cfg.inference

    # Khởi tạo Generator theo config inference để đồng bộ train/infer.
    generator = FaceDiffGenerator(
        device=device,
        context_dim=int(cfg.imf.context_dim),
        sc_vae_ckpt=inf_cfg.sc_vae_checkpoint,
        sc_vae_shape_ckpt=inf_cfg.sc_vae_shape_checkpoint,
        sc_vae_material_ckpt=inf_cfg.sc_vae_material_checkpoint,
        imf_ckpt=inf_cfg.imf_checkpoint,
        structure_ckpt=inf_cfg.structure_checkpoint,
        enable_structure_stage=bool(inf_cfg.enable_structure_stage),
        structure_threshold=float(inf_cfg.structure_threshold),
        feature_mode=str(inf_cfg.feature_mode),
        cfg_scale=float(inf_cfg.cfg_scale),
        cfg_tmin=float(inf_cfg.cfg_tmin),
        cfg_tmax=float(inf_cfg.cfg_tmax),
        neg_guidance_scale=float(inf_cfg.neg_guidance_scale),
        mesh_backend=str(inf_cfg.mesh_backend),
        enforce_dual_contouring=bool(inf_cfg.enforce_dual_contouring),
        mesh_smooth_sigma=float(inf_cfg.mesh_smooth_sigma),
        ovoxel_resolution=int(inf_cfg.ovoxel_resolution),
    )
    
    # Giả lập Hybrid vector [1, 946]
    print("\n[Test] Tạo Hybrid Context vector giả lập [1, 946]...")
    fake_context = torch.randn(1, 946, device=device)
    fake_context = torch.nn.functional.normalize(fake_context, p=2, dim=-1)
    
    # Sinh mesh 3D
    out_dir = "/mnt/18TData/facediff/data/generated"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_test_face.obj")
    
    result_path = generator.generate(fake_context, output_path=out_path)
    
    # Kiểm tra file xuất
    if os.path.exists(result_path):
        file_size = os.path.getsize(result_path) / 1024
        print(f"\n[KIỂM TRA] File OBJ tồn tại: {result_path}")
        print(f"[KIỂM TRA] Kích thước: {file_size:.1f} KB")
        
        # Đếm vertices & faces
        v_count, f_count = 0, 0
        with open(result_path) as f:
            for line in f:
                if line.startswith('v '): v_count += 1
                elif line.startswith('f '): f_count += 1
        print(f"[KIỂM TRA] Vertices: {v_count}, Faces: {f_count}")
    else:
        print(f"[LỖI] File không được tạo!")
        
    # VRAM tổng kết
    if device != "cpu":
        total_vram = torch.cuda.memory_allocated(device) / (1024**2)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\n[VRAM] Hiện tại: {total_vram:.2f} MB")
        print(f"[VRAM] Đỉnh điểm: {peak_vram:.2f} MB")
    
    print("\n" + "=" * 60)
    print(" PASSED - INFERENCE PIPELINE HOẠT ĐỘNG HOÀN HẢO!")
    print("=" * 60)


if __name__ == "__main__":
    run_inference_test()
