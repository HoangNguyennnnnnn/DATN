import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import spconv.pytorch as spconv
from spconv.pytorch import ConvAlgo
# Hard requirement: spconv is mandatory for SC-VAE
# No fallback to mock layers - this ensures quality is never silently degraded
# Use Native algo to avoid ConvTunerSimple autotuning failures on certain voxel shapes
_ALGO = ConvAlgo.Native

# Export for backward compatibility (always True now)
SPCONV_AVAILABLE = True


# TRELLIS.2 official spec: SC-VAE output activations.
# Reference: trellis2/models/sc_vaes/fdg_vae.py FlexiDualGridVaeDecoder.forward()
#   vertices       = (1 + 2*voxel_margin) * sigmoid(h[..., 0:3]) - voxel_margin
#   intersected    = h[..., 3:6] > 0  (raw logit threshold at inference)
#   quad_lerp      = softplus(h[..., 6:7])
# voxel_margin=0.5 lets dual vertex extend slightly beyond its voxel cell,
# improving boundary accuracy of the dual contouring extraction.
TRELLIS2_VOXEL_MARGIN: float = 0.5


def apply_dv_activation(h_dv: torch.Tensor, voxel_margin: float = TRELLIS2_VOXEL_MARGIN) -> torch.Tensor:
    """Map raw dual-vertex logits to [-margin, 1+margin] (TRELLIS.2 Eq. Act-1)."""
    return (1.0 + 2.0 * voxel_margin) * torch.sigmoid(h_dv) - voxel_margin


def apply_intersection_activation(h_flag: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Sigmoid in train (differentiable), hard threshold at logit>0 in eval (TRELLIS.2 Eq. Act-2)."""
    if training:
        return torch.sigmoid(h_flag)
    return (h_flag > 0).to(h_flag.dtype)


def apply_gamma_activation(h_gamma: torch.Tensor) -> torch.Tensor:
    """Softplus enforces strictly positive split-weight (TRELLIS.2 Eq. Act-3)."""
    return F.softplus(h_gamma)


def apply_shape_mat_output_activations(
    out_proj: torch.Tensor,
    training: bool = True,
    voxel_margin: float = TRELLIS2_VOXEL_MARGIN,
    rgb_clamp: bool = True,
) -> torch.Tensor:
    """Apply paper-spec activations to a 10-channel `shape_mat` output tensor.

    Channels: [dv(3), delta(3), gamma(1), rgb(3)].
    The returned tensor has the same shape but post-activation values:
      - dv:    (1+2m)·sigmoid(h)-m  ∈ [-m, 1+m]
      - delta: sigmoid in training, threshold>0 in eval
      - gamma: softplus(h) > 0
      - rgb:   clamp(h, 0, 1) when rgb_clamp=True (Albedo lives in [0,1])
    """
    if out_proj.shape[-1] < 7:
        # Fallback: at least produce dv activation if available
        if out_proj.shape[-1] >= 3:
            return torch.cat([
                apply_dv_activation(out_proj[..., 0:3], voxel_margin),
                out_proj[..., 3:],
            ], dim=-1)
        return out_proj
    parts = [
        apply_dv_activation(out_proj[..., 0:3], voxel_margin),
        apply_intersection_activation(out_proj[..., 3:6], training=training),
        apply_gamma_activation(out_proj[..., 6:7]),
    ]
    if out_proj.shape[-1] >= 10:
        rgb = out_proj[..., 7:10]
        parts.append(rgb.clamp(0.0, 1.0) if rgb_clamp else torch.sigmoid(rgb))
    if out_proj.shape[-1] > 10:
        parts.append(out_proj[..., 10:])
    return torch.cat(parts, dim=-1)


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise the parameters of a module (TRELLIS.2 utils.zero_module)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# Use a fixed hash base to avoid per-forward CPU sync via dynamic max().item().
_SPARSE_HASH_BASE = 32768


def _replace_sparse_feature(x, new_features: torch.Tensor):
    """Compatibility helper for spconv versions with/without replace_feature."""
    if hasattr(x, "replace_feature"):
        return x.replace_feature(new_features)
    x.features = new_features
    return x


def _hash_indices(indices: torch.Tensor, base: int) -> torch.Tensor:
    """Hash [N,4] sparse indices to int64 keys for fast parent/child lookup."""
    idx = indices.to(dtype=torch.int64)
    return (((idx[:, 0] * base + idx[:, 1]) * base + idx[:, 2]) * base + idx[:, 3])


def _lookup_features_by_keys(
    query_keys: torch.Tensor,
    source_keys: torch.Tensor,
    source_features: torch.Tensor,
):
    """Lookup source features by hashed keys using sorted search."""
    if source_keys.numel() == 0:
        zeros = source_features.new_zeros((query_keys.shape[0], source_features.shape[1]))
        mask = torch.zeros((query_keys.shape[0],), dtype=torch.bool, device=query_keys.device)
        return zeros, mask

    sorted_keys, order = torch.sort(source_keys)
    sorted_feats = source_features[order]

    pos = torch.searchsorted(sorted_keys, query_keys)
    safe_pos = torch.clamp(pos, 0, max(sorted_keys.shape[0] - 1, 0))
    valid = (pos < sorted_keys.shape[0]) & (sorted_keys[safe_pos] == query_keys)

    out = source_features.new_zeros((query_keys.shape[0], source_features.shape[1]))
    if valid.any():
        out[valid] = sorted_feats[safe_pos[valid]]
    return out, valid


def _avg_groups_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Non-parametric grouped averaging to target channel width."""
    if x.shape[1] == target_channels:
        return x
    if target_channels <= 0:
        return x[:, :0]
    c = x.shape[1]
    if c % target_channels == 0:
        g = c // target_channels
        return x.view(x.shape[0], target_channels, g).mean(dim=2)
    return F.adaptive_avg_pool1d(x.unsqueeze(1), target_channels).squeeze(1)


def _dup_groups_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Non-parametric channel duplication to target channel width."""
    c = x.shape[1]
    if c == target_channels:
        return x
    if c <= 0:
        return x.new_zeros((x.shape[0], target_channels))
    if c < target_channels:
        rep = (target_channels + c - 1) // c
        y = x.repeat_interleave(rep, dim=1)
        return y[:, :target_channels]
    return F.adaptive_avg_pool1d(x.unsqueeze(1), target_channels).squeeze(1)


def _build_sparse_down_shortcut(
    fine_indices: torch.Tensor,
    fine_features: torch.Tensor,
    coarse_indices: torch.Tensor,
    target_channels: int,
) -> torch.Tensor:
    """Approximate Eq.(4): stack 8 children then avg_groups to coarse channels."""
    if coarse_indices.numel() == 0:
        return fine_features.new_zeros((0, target_channels))

    source_keys = _hash_indices(fine_indices, _SPARSE_HASH_BASE)

    offsets = fine_indices.new_tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=torch.int64,
    )
    coarse_xyz = coarse_indices[:, 1:].to(torch.int64)
    child_xyz = coarse_xyz.unsqueeze(1) * 2 + offsets.unsqueeze(0)
    child_b = coarse_indices[:, 0:1].to(torch.int64).unsqueeze(1).expand(-1, 8, -1)
    child_idx = torch.cat([child_b, child_xyz], dim=2).view(-1, 4)
    child_keys = _hash_indices(child_idx, _SPARSE_HASH_BASE)

    child_feats, _ = _lookup_features_by_keys(child_keys, source_keys, fine_features)
    child_feats = child_feats.view(coarse_indices.shape[0], 8 * fine_features.shape[1])
    return _avg_groups_channels(child_feats, target_channels)


def _build_sparse_up_shortcut(
    coarse_indices: torch.Tensor,
    coarse_features: torch.Tensor,
    fine_indices: torch.Tensor,
    target_channels: int,
) -> torch.Tensor:
    """Approximate Eq.(5): unstack parent channels by child id then duplicate groups."""
    if fine_indices.numel() == 0:
        return coarse_features.new_zeros((0, target_channels))

    source_keys = _hash_indices(coarse_indices, _SPARSE_HASH_BASE)
    parent_idx = torch.cat([
        fine_indices[:, 0:1],
        (fine_indices[:, 1:] // 2),
    ], dim=1)
    parent_keys = _hash_indices(parent_idx, _SPARSE_HASH_BASE)
    parent_feats, _ = _lookup_features_by_keys(parent_keys, source_keys, coarse_features)

    if parent_feats.shape[1] >= 8 and parent_feats.shape[1] % 8 == 0:
        seg = parent_feats.shape[1] // 8
        child_id = (fine_indices[:, 1] % 2) * 4 + (fine_indices[:, 2] % 2) * 2 + (fine_indices[:, 3] % 2)
        parent_reshaped = parent_feats.view(parent_feats.shape[0], 8, seg)
        picked = parent_reshaped[torch.arange(parent_feats.shape[0], device=parent_feats.device), child_id.long()]
    else:
        picked = parent_feats

    return _dup_groups_channels(picked, target_channels)


def _apply_child_pruning(
    x_up,
    parent_indices: torch.Tensor,
    parent_rho_logits: torch.Tensor,
    threshold: float,
):
    """Prune child voxels using rho logits predicted on parent voxels.

    This implements the Early-Pruning Upsampler from TRELLIS.2 paper Section 3.2.1.
    Keeps indices only for children whose predicted occupancy exceeds the threshold.
    
    Note: No fallback - if pruning fails, we raise an error to prevent silent degradation.
    """
    if parent_indices is None or parent_indices.numel() == 0:
        return x_up
    if parent_rho_logits is None or parent_rho_logits.numel() == 0:
        raise ValueError("Early-pruning requires parent rho logits, got None or empty")

    child_indices = x_up.indices
    if child_indices.numel() == 0:
        return x_up

    parent_keys = _hash_indices(parent_indices, _SPARSE_HASH_BASE)
    child_parent = torch.cat([
        child_indices[:, 0:1],
        (child_indices[:, 1:] // 2),
    ], dim=1)
    child_parent_keys = _hash_indices(child_parent, _SPARSE_HASH_BASE)
    parent_logits, valid = _lookup_features_by_keys(child_parent_keys, parent_keys, parent_rho_logits)
    if parent_logits.numel() == 0:
        raise RuntimeError("Early-pruning: Failed to lookup parent logits for child voxels")

    child_id = (
        (child_indices[:, 1] & 1) * 4
        + (child_indices[:, 2] & 1) * 2
        + (child_indices[:, 3] & 1)
    ).to(torch.long)
    child_prob = torch.sigmoid(parent_logits).gather(1, child_id.unsqueeze(1)).squeeze(1)
    keep = (child_prob >= float(threshold)) & valid
    if int(keep.sum().item()) == 0:
        raise RuntimeError("Early-pruning: All child voxels would be pruned (threshold too high or bad predictions)")

    new_features = x_up.features[keep]
    new_indices = child_indices[keep]
    pruned = spconv.SparseConvTensor(
        new_features,
        new_indices,
        x_up.spatial_shape,
        x_up.batch_size,
    )
    pruned.indice_dict = dict(getattr(x_up, "indice_dict", {}))
    pruned.grid = getattr(x_up, "grid", None)
    return pruned

def _prune_by_target(x_up, target_indices: torch.Tensor):
    """
    Teacher-Forcing Pruning: During training, prune the upsampled voxels
    so they perfectly match the Ground Truth spatial topology.
    
    Paper Section 3.2.1: "During training, we use teacher-forcing with ground-truth
    occupancy masks for stable gradients."
    """
    if target_indices is None or target_indices.numel() == 0:
        raise ValueError("Teacher-forcing pruning requires target_indices, got None or empty")
    if not hasattr(x_up, "indices") or not hasattr(x_up, "features"):
        raise TypeError(f"Expected SparseConvTensor, got {type(x_up)}")
    if x_up.indices.numel() == 0:
        return x_up

    x_up_keys = _hash_indices(x_up.indices, _SPARSE_HASH_BASE)
    tgt_keys = _hash_indices(target_indices, _SPARSE_HASH_BASE)

    sorted_tgt_keys, _ = torch.sort(tgt_keys)
    pos = torch.searchsorted(sorted_tgt_keys, x_up_keys)
    safe_pos = torch.clamp(pos, 0, max(sorted_tgt_keys.shape[0] - 1, 0))
    valid = (pos < sorted_tgt_keys.shape[0]) & (sorted_tgt_keys[safe_pos] == x_up_keys)
    
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        raise RuntimeError("Teacher-forcing: No valid voxels matched target topology (all pruned)")

    new_features = x_up.features[valid]
    new_indices = x_up.indices[valid]
    
    pruned = spconv.SparseConvTensor(
        new_features,
        new_indices,
        x_up.spatial_shape,
        x_up.batch_size,
    )
    pruned.indice_dict = dict(getattr(x_up, "indice_dict", {}))
    pruned.grid = getattr(x_up, "grid", None)
    return pruned


def _build_child_mask_targets(coarse_indices: torch.Tensor, fine_indices: torch.Tensor) -> torch.Tensor:
    """Build 8-way child occupancy mask per coarse sparse voxel.

    Both indices are expected in [N, 4] format: [batch, x, y, z] or [batch, z, y, x].
    We only require integer parent-child relation via //2 and %2 per spatial axis.
    """
    device = coarse_indices.device
    num_coarse = int(coarse_indices.shape[0])
    targets = torch.zeros((num_coarse, 8), device=device, dtype=torch.float32)
    if num_coarse == 0 or fine_indices is None or fine_indices.numel() == 0:
        return targets

    coarse_keys = _hash_indices(coarse_indices, _SPARSE_HASH_BASE)
    sorted_keys, order = torch.sort(coarse_keys)

    fine_parent = torch.cat([
        fine_indices[:, 0:1],
        (fine_indices[:, 1:] // 2),
    ], dim=1)
    fine_parent_keys = _hash_indices(fine_parent, _SPARSE_HASH_BASE)

    pos = torch.searchsorted(sorted_keys, fine_parent_keys)
    safe_pos = torch.clamp(pos, 0, max(sorted_keys.shape[0] - 1, 0))
    valid = (pos < sorted_keys.shape[0]) & (sorted_keys[safe_pos] == fine_parent_keys)
    if not valid.any():
        return targets

    rows = order[safe_pos[valid]].to(torch.long)
    child_id = (
        (fine_indices[valid, 1] & 1) * 4
        + (fine_indices[valid, 2] & 1) * 2
        + (fine_indices[valid, 3] & 1)
    ).to(torch.long)
    targets[rows, child_id] = 1.0

    return targets


def _infer_sparse_spatial_shape(indices: torch.Tensor, default_size: int = 16):
    """Infer sparse spatial shape [D, H, W] from [N,4] indices."""
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0 or indices.ndim != 2 or indices.shape[1] < 4:
        s = int(max(2, default_size))
        return [s, s, s]
    max_xyz = indices[:, 1:4].to(torch.int64).amax(dim=0)
    spatial = [int(max(2, int(v.item()) + 1)) for v in max_xyz]
    return spatial


class SparseResMLPBlock(nn.Module):
    """ConvNeXt-style sparse residual block: one sparse conv + point-wise MLP.

    Mirrors TRELLIS.2 ``SparseConvNeXtBlock3d`` (trellis2/models/sc_vaes/sparse_unet_vae.py).
    Uses FP32-cast LayerNorm (LayerNorm32) for AMP numerical stability and zero-inits
    the final point-wise projection so the residual branch is identity at init.

    Note: parameter shapes/names are unchanged vs. the prior version, so checkpoints
    such as ``sc_vae_shape/epoch_390.pt`` load without modification (LayerNorm32 is a
    plain ``nn.LayerNorm`` subclass with the same ``weight``/``bias`` parameters).
    Zero-init only matters when training from scratch; when resuming, the loaded
    weights overwrite the zero initialisation immediately.
    """

    def __init__(self, channels: int, mlp_ratio: int = 4, key_id: str = "0"):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            indice_key=f"subm_res_{key_id}",
            algo=_ALGO,
        )
        # TRELLIS.2 uses LayerNorm32 (FP32 cast) for AMP stability. Falls back to
        # plain nn.LayerNorm if the project-wide module is unavailable.
        try:
            from src.modules.norm import LayerNorm32 as _LN32
            self.norm = _LN32(channels)
        except Exception:
            self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, hidden)
        self.act = nn.SiLU()
        # zero-init final projection: residual branch outputs 0 at init -> stable optimisation
        self.pw2 = _zero_module(nn.Linear(hidden, channels))

    def forward(self, x):
        residual = x.features
        y = self.conv(x)
        # LayerNorm32 wrapper accepts SparseConvTensor (it would call .replace()),
        # but here we already have raw features so call the underlying nn.LayerNorm path.
        f = y.features
        if hasattr(self.norm, "_parameters") and isinstance(self.norm, nn.LayerNorm):
            # nn.LayerNorm path: keep dtype consistent with input for spconv kernels
            f = self.norm(f.float()).to(f.dtype) if f.dtype != torch.float32 else self.norm(f)
        else:
            f = self.norm(f)
        f = self.act(self.pw1(f))
        f = self.pw2(f)
        f = f + residual
        return _replace_sparse_feature(y, f)

class SparseEncoderBlock(nn.Module):
    """ Downsampling block for Sparse Voxel grid with Residual Autoencoding.
    
    Paper Section 3.2.1: "We adapt the Residual Autoencoding principle from DC-AE 
    to sparse voxel data by introducing non-parametric residual shortcuts within 
    downsampling and upsampling blocks."
    """
    def __init__(self, in_c: int, out_c: int, key_id: str = "0", num_res_blocks: int = 2):
        super().__init__()
        self.out_c = out_c
        self.proj = spconv.SubMConv3d(in_c, out_c, kernel_size=1, indice_key=f"proj_{key_id}", algo=_ALGO)
        # Multiple residual blocks as per paper: "multiple residual blocks"
        self.res_blocks = nn.ModuleList([
            SparseResMLPBlock(out_c, mlp_ratio=4, key_id=f"enc_{key_id}_{i}")
            for i in range(num_res_blocks)
        ])
        self.down = spconv.SparseConv3d(out_c, out_c, kernel_size=2, stride=2, indice_key=f"down_{key_id}", algo=_ALGO)
            
    def forward(self, x):
        x = self.proj(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x_down = self.down(x)

        # Residual Autoencoding shortcut (Eq. 4 in paper)
        shortcut = _build_sparse_down_shortcut(
            fine_indices=x.indices,
            fine_features=x.features,
            coarse_indices=x_down.indices,
            target_channels=self.out_c,
        )
        x_down = _replace_sparse_feature(x_down, x_down.features + shortcut)
        return x_down

class SparseDecoderBlock(nn.Module):
    """ Upsampling block for Sparse Voxel grid with Early-Pruning.
    
    Paper Section 3.2.1: "We employ an early-pruning mechanism for the upsampler. 
    Before each upsampling step, the module predicts a binary mask specifying 
    the active child voxels of each parent node."
    """
    def __init__(self, in_c: int, out_c: int, key_id: str = "0", rho_prune_threshold: float = 0.5, num_res_blocks: int = 2):
        super().__init__()
        self.out_c = out_c
        self.rho_prune_threshold = float(rho_prune_threshold)
        # Predict 8-way child occupancy (rho mask)
        self.rho_head = nn.Linear(in_c, 8)
        # Use SparseConvTranspose3d for Generative Upsampling (kernel=2, stride=2 expands 1 voxel -> 8 voxels)
        self.up = spconv.SparseConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, indice_key=f"up_{key_id}", algo=_ALGO)
        # Multiple residual blocks as per paper
        self.res_blocks = nn.ModuleList([
            SparseResMLPBlock(out_c, mlp_ratio=4, key_id=f"dec_{key_id}_{i}")
            for i in range(num_res_blocks)
        ])
            
    def forward(self, x, target_fine_indices: Optional[torch.Tensor] = None):
        rho_logits = self.rho_head(x.features)
        rho_targets = None
        if target_fine_indices is not None and hasattr(x, "indices"):
            rho_targets = _build_child_mask_targets(x.indices, target_fine_indices)

        # Optional lightweight parent feature gating
        gate = torch.sigmoid(rho_logits).amax(dim=1, keepdim=True)
        x = _replace_sparse_feature(x, x.features * gate)
        parent_indices = x.indices
        parent_features = x.features
        
        # Generative Upsample (Allocates non-overlapping children for all parents)
        x_up = self.up(x)

        # Residual Autoencoding shortcut (Eq. 5 in paper)
        shortcut = _build_sparse_up_shortcut(
            coarse_indices=parent_indices,
            coarse_features=parent_features,
            fine_indices=x_up.indices,
            target_channels=self.out_c,
        )
        x_up = _replace_sparse_feature(x_up, x_up.features + shortcut)

        # Early Pruning Upsampler Logic
        if target_fine_indices is not None:
            # Training Phase: Teacher-force the output to perfectly match Ground Truth topology.
            x_up = _prune_by_target(x_up, target_fine_indices)
        else:
            # Inference Phase: Autonomous pruning using predicted rho logits.
            x_up = _apply_child_pruning(
                x_up,
                parent_indices=parent_indices,
                parent_rho_logits=rho_logits,
                threshold=self.rho_prune_threshold,
            )

        for res_block in self.res_blocks:
            x_up = res_block(x_up)
        return x_up, rho_logits, rho_targets

class SC_VAE(nn.Module):
    """
    Sparse Convolutional Variational Autoencoder (SC-VAE).
    Compresses high-resolution (512^3) O-Voxels into a compacted "Slat" token representation (~9.6K tokens).
    Works for both Shape (Geometry) or Texture (Materials), depending on `in_channels`.
    
    Scaled for RTX 4090 (24GB VRAM) with face portrait dataset.
    Reference: TRELLIS.2 paper Section 3.2 - "Sparse Compression VAE"
    """
    def __init__(
        self,
        in_channels: int = 7,
        latent_dim: int = 32,
        device: str = "cuda:0",
        rho_prune_threshold: float = 0.5,
        encoder_dims: list = None,
        num_res_blocks: int = 2,
        voxel_margin: float = TRELLIS2_VOXEL_MARGIN,
        apply_output_activations: bool = False,
        pre_latent_norm: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.rho_prune_threshold = float(rho_prune_threshold)
        # TRELLIS.2 dual-vertex margin: dv ∈ [-m, 1+m] after activation.
        # Stored on the module so callers can read it for downstream coord arithmetic.
        self.voxel_margin = float(voxel_margin)
        # When True, ``forward()`` applies paper-spec activations to ``recon`` before
        # returning. Default False keeps backward-compatibility (existing loss apply
        # activations themselves via apply_shape_mat_output_activations() / BCEWithLogits).
        self.apply_output_activations = bool(apply_output_activations)
        # TRELLIS.2 places a non-affine LayerNorm right before to_mu/to_logvar
        # (sparse_unet_vae.py SparseUnetVaeEncoder.forward, last F.layer_norm call).
        # It carries no learnable parameters so it never affects checkpoint state_dict.
        self.pre_latent_norm = bool(pre_latent_norm)

        # Scaled architecture for RTX 4090 (face portraits don't need full 800M params)
        # Default: 64 -> 128 -> 256 -> 512 (more capacity than original 32->64->128->256)
        if encoder_dims is None:
            encoder_dims = [64, 128, 256, 512]
        self.encoder_dims = encoder_dims
        
        # Encoder (U-Net styled downsampling to token bottleneck)
        self.enc1 = SparseEncoderBlock(in_channels, encoder_dims[0], key_id="enc1", num_res_blocks=num_res_blocks)
        self.enc2 = SparseEncoderBlock(encoder_dims[0], encoder_dims[1], key_id="enc2", num_res_blocks=num_res_blocks)
        self.enc3 = SparseEncoderBlock(encoder_dims[1], encoder_dims[2], key_id="enc3", num_res_blocks=num_res_blocks)
        self.enc4 = SparseEncoderBlock(encoder_dims[2], encoder_dims[3], key_id="enc4", num_res_blocks=num_res_blocks)
        
        # Latent projection (Mean and LogVar)
        self.to_mu = nn.Linear(encoder_dims[3], latent_dim)
        self.to_logvar = nn.Linear(encoder_dims[3], latent_dim)
        
        # Decoder (Upsampling from Latent back to high-res voxels)
        self.dec_proj = nn.Linear(latent_dim, encoder_dims[3])
        self.dec4 = SparseDecoderBlock(encoder_dims[3], encoder_dims[2], key_id="dec4", 
                                       rho_prune_threshold=self.rho_prune_threshold, num_res_blocks=num_res_blocks)
        self.dec3 = SparseDecoderBlock(encoder_dims[2], encoder_dims[1], key_id="dec3", 
                                       rho_prune_threshold=self.rho_prune_threshold, num_res_blocks=num_res_blocks)
        self.dec2 = SparseDecoderBlock(encoder_dims[1], encoder_dims[0], key_id="dec2", 
                                       rho_prune_threshold=self.rho_prune_threshold, num_res_blocks=num_res_blocks)
        self.dec1 = SparseDecoderBlock(encoder_dims[0], encoder_dims[0], key_id="dec1", 
                                       rho_prune_threshold=self.rho_prune_threshold, num_res_blocks=num_res_blocks)
        
        # Final output projection depending on target (Shape logits or Texture regression)
        self.out_proj = nn.Linear(encoder_dims[0], in_channels)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sparse voxels to latent distribution.
        
        Args:
            x: spconv.SparseConvTensor with shape [N, C] features
            
        Returns:
            mu, logvar: Tensors of shape [N, latent_dim]
        """
        if not hasattr(x, 'features'):
            raise TypeError(f"SC-VAE encode expects SparseConvTensor, got {type(x)}")
        
        # Sparse encode path (4-level hierarchical downsampling)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        feats = x.features

        # TRELLIS.2: non-affine LayerNorm before to_mu/to_logvar for stable posterior.
        # F.layer_norm(..., normalized_shape=feats.shape[-1:]) has zero parameters →
        # state_dict is unchanged, so resuming an old checkpoint is safe.
        if self.pre_latent_norm:
            feats_fp32 = feats.float()
            feats_fp32 = F.layer_norm(feats_fp32, (feats_fp32.shape[-1],))
            feats = feats_fp32.to(feats.dtype)

        mu = self.to_mu(feats)
        logvar = self.to_logvar(feats)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        original_indices: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        sparse_template=None,
        sparse_pyramid=None,
        return_indices: bool = False,
    ):
        """
        Decode Slat tokens back to voxel features.
        - Training: Use sparse_pyramid for teacher-forcing topology constraints.
        - Inference: Autonomously upsample and predict sparsity map.
        
        Args:
            z: Latent tensor [N, latent_dim]
            original_indices: Target fine indices for final level (training) or root indices (inference)
            batch_size: Batch size for inference bootstrapping
            sparse_template: Template sparse tensor for feature replacement
            sparse_pyramid: List of 4 sparse tensors from encoder for teacher forcing
            return_indices: Whether to return output indices
        """
        feats = self.dec_proj(z)
        rho_logits_list = []
        rho_targets_list = []
        out_indices = None
        
        if sparse_template is not None:
            # We have the encoded sparse topology (training mode with teacher forcing)
            x_sparse = _replace_sparse_feature(sparse_template, feats)
            
            # If we are training, use sparse_pyramid. Otherwise, leave None for generative pruning.
            if sparse_pyramid is not None and len(sparse_pyramid) == 4:
                x, rho4, tgt4 = self.dec4(x_sparse, target_fine_indices=sparse_pyramid[2].indices)
                x, rho3, tgt3 = self.dec3(x, target_fine_indices=sparse_pyramid[1].indices)
                x, rho2, tgt2 = self.dec2(x, target_fine_indices=sparse_pyramid[0].indices)
                x, rho1, tgt1 = self.dec1(x, target_fine_indices=original_indices)
                rho_logits_list = [rho4, rho3, rho2, rho1]
                rho_targets_list = [tgt4, tgt3, tgt2, tgt1]
            else:
                # Inference mode with sparse template (hierarchical topology known)
                x, _, _ = self.dec4(x_sparse)
                x, _, _ = self.dec3(x)
                x, _, _ = self.dec2(x)
                x, _, _ = self.dec1(x)
            
            out_feats = x.features
            out_indices = x.indices
        
        elif original_indices is not None:
            # Pure Inference generation from Slat tokens WITHOUT known hierarchy.
            # `original_indices` here would be the Slat 4096 tokens indices.
            # Bootstrapping the root 4096 indices from U-DiT (16x16x16 grid).
            spatial_shape = _infer_sparse_spatial_shape(original_indices, default_size=16)
            x_sparse = spconv.SparseConvTensor(feats, original_indices, spatial_shape, batch_size)
            x, _, _ = self.dec4(x_sparse)
            x, _, _ = self.dec3(x)
            x, _, _ = self.dec2(x)
            x, _, _ = self.dec1(x)
            out_feats = x.features
            out_indices = x.indices
        else:    
            raise ValueError("Decoder needs either sparse_template (training) or original_indices (inference)!")

        out_proj = self.out_proj(out_feats)
        if return_indices:
            return out_proj, rho_logits_list, rho_targets_list, out_indices
        return out_proj, rho_logits_list, rho_targets_list

    def forward(self, x):
        """
        Forward pass through SC-VAE.
        
        Input: spconv.SparseConvTensor with shape [N, in_channels]
        Returns: 
            recon: Reconstructed features [N, in_channels]
            mu, logvar: Latent distribution parameters [N, latent_dim]
            rho_logits_list: List of 4 rho predictions for early-pruning supervision
            rho_targets_list: List of 4 rho targets (training only)
        """
        if not hasattr(x, 'features'):
            raise TypeError(f"SC-VAE forward expects SparseConvTensor, got {type(x)}")

        # Keep sparse context from encoder so SparseConvTranspose can reuse indice_dict.
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        feats = x4.features

        # TRELLIS.2: non-affine LayerNorm before to_mu/to_logvar (zero-param, ckpt-safe).
        if self.pre_latent_norm:
            feats_fp32 = feats.float()
            feats_fp32 = F.layer_norm(feats_fp32, (feats_fp32.shape[-1],))
            feats = feats_fp32.to(feats.dtype)

        mu = self.to_mu(feats)
        logvar = self.to_logvar(feats)
        z = self.reparameterize(mu, logvar)
        recon, rho_logits_list, rho_targets_list, out_indices = self.decode(
            z,
            original_indices=x.indices,
            sparse_template=x4,
            sparse_pyramid=[x1, x2, x3, x4],
            return_indices=True,
        )

        if self.apply_output_activations:
            recon = apply_shape_mat_output_activations(
                recon, training=self.training, voxel_margin=self.voxel_margin,
            )

        return recon, mu, logvar, rho_logits_list, rho_targets_list, out_indices
