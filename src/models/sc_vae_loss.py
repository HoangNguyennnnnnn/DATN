import torch
import torch.nn as nn
import torch.nn.functional as F

class SCVAELoss(nn.Module):
    """
    Hàm Loss tổng hợp dùng để huấn luyện mạng SC-VAE nén Voxel 3D.
    Bao gồm:
    1. Reconstruction Loss: MSE (cho features như màu, roughness) hoặc BCE (cho mask/điểm).
    2. KL Divergence Loss: Ép phân phối của Slat Token về dạng Gaussian chuẩn (0, 1) để cho phép U-Net sinh mẫu mượt mà.
    """
    def __init__(self, kl_weight: float = 1e-4, use_bce_for_geom: bool = False, rho_loss_weight: float = 1.0):
        """
        Args:
            kl_weight (float): Trọng số cân bằng KL Divergence. Quá cao sẽ làm mờ texture, quá thấp sẽ khiến sinh mẫu Flow Matching khó hội tụ.
            use_bce_for_geom (bool): Nếu True, sử dụng Binary Cross Entropy cho Geometry logic thay vì MSE.
            rho_loss_weight (float): Trọng số cho Focal Loss của Early-Pruning (rho mask).
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.use_bce_for_geom = use_bce_for_geom
        self.rho_loss_weight = rho_loss_weight

    def _shape_recon_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor, feature_mode: str) -> torch.Tensor:
        """Geometry-centric reconstruction terms for shape branches."""
        if self.use_bce_for_geom:
            # BCE path for occupancy-like / binary geometry targets (`use_bce_for_geom=True`).
            return F.binary_cross_entropy_with_logits(recon_x, target_x, reduction='mean')

        # Shape proxy: prioritize xyz, then normals/remaining channels.
        if recon_x.shape[1] < 3 or target_x.shape[1] < 3:
            return F.mse_loss(recon_x, target_x, reduction='mean')

        loss_xyz = F.mse_loss(recon_x[:, :3], target_x[:, :3], reduction='mean')
        if recon_x.shape[1] >= 6 and target_x.shape[1] >= 6:
            loss_rest = F.mse_loss(recon_x[:, 3:], target_x[:, 3:], reduction='mean')
            return 0.7 * loss_xyz + 0.3 * loss_rest
        return loss_xyz

    def _shape_native_recon_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        """Approximate paper stage-1 geometry terms for native dual-grid features.

        We supervise:
        - dual vertex v (first 3 dims) with MSE
        - dual face flags delta (next up to 3 dims) with BCE
        - splitting weight gamma (last dim when available) with positive-regression loss
        """
        if recon_x.shape[1] < 3 or target_x.shape[1] < 3:
            return F.mse_loss(recon_x, target_x, reduction='mean')

        loss_v = F.mse_loss(recon_x[:, :3], target_x[:, :3], reduction='mean')
        if recon_x.shape[1] >= 6 and target_x.shape[1] >= 6:
            # If channels 3:6 are real-valued (e.g., normals in [-1,1]),
            # BCE is invalid and destabilizes optimization. Fall back to MSE.
            flags_src = target_x[:, 3:6]
            is_binary_like = bool(
                torch.all((flags_src >= 0.0) & (flags_src <= 1.0)).item()
            )
            if is_binary_like:
                target_flags = flags_src.clamp(0.0, 1.0)
                loss_delta = F.binary_cross_entropy_with_logits(
                    recon_x[:, 3:6],
                    target_flags,
                    reduction='mean',
                )
            else:
                loss_delta = F.mse_loss(recon_x[:, 3:6], flags_src, reduction='mean')
            loss_total = loss_v + loss_delta

            if recon_x.shape[1] >= 7 and target_x.shape[1] >= 7:
                # gamma is expected positive; enforce positivity on prediction side.
                pred_gamma = F.softplus(recon_x[:, 6:7])
                tgt_gamma = torch.clamp(target_x[:, 6:7], min=1e-3)
                loss_gamma = F.smooth_l1_loss(pred_gamma, tgt_gamma, reduction='mean')
                loss_total = loss_total + loss_gamma
            return loss_total
        return loss_v

    def _material_recon_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        """Material-centric reconstruction term (paper uses L1 for material attributes)."""
        return F.l1_loss(recon_x, target_x, reduction='mean')

    def _shape_mat_recon_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        """Per-channel loss cho 10-channel shape_mat [dv(3), delta(3), gamma(1), rgb(3)].

        Loss weights — đối chiếu trực tiếp với TRELLIS.2
        ``configs/scvae/shape_vae_next_dc_f16c32_fp16.json``:
        - dv (channels 0:3): MSE × 0.01 = ``lambda_vertice``.
          Predictions are first mapped through the TRELLIS.2 dv activation
          ``(1+2m)·sigmoid(h)-m`` (Eq. Act-1 with m=0.5), so ``recon_dv`` lives in
          [-0.5, 1.5] just like the GT can after voxel margin extension. The MSE is
          taken on activated values so gradients respect the saturation regions.
        - delta (channels 3:6): BCE × 0.1 = ``lambda_intersected`` — binary flags
          for surface intersection on each axis. Uses BCE-with-logits so we keep
          numerical stability even on raw logits (no double sigmoid).
        - gamma (channel 6): softplus(pred) + smooth_l1, weight 1.0. Softplus
          guarantees positivity for downstream dual contouring split weighting.
        - rgb (channels 7:10): L1 × 1.0 (TRELLIS.2 uses L1 for material attrs).
          We clamp predictions to [0,1] before the L1 because Albedo lives there.
        """
        # dv (0:3): activated → MSE × 0.01.
        # Activate so gradients live on the same scale as targets (≈ [0,1] inside cell).
        from src.models.sc_vae import apply_dv_activation, TRELLIS2_VOXEL_MARGIN
        recon_dv = apply_dv_activation(recon_x[:, 0:3], voxel_margin=TRELLIS2_VOXEL_MARGIN)
        tgt_dv = target_x[:, 0:3].clamp(min=-TRELLIS2_VOXEL_MARGIN, max=1.0 + TRELLIS2_VOXEL_MARGIN)
        loss_dv = F.mse_loss(recon_dv, tgt_dv, reduction='mean')

        # delta (3:6): BCE-with-logits × 0.1 — keep raw logits for numerical stability.
        loss_delta = F.binary_cross_entropy_with_logits(
            recon_x[:, 3:6], target_x[:, 3:6].clamp(0.0, 1.0), reduction='mean')

        # gamma (6:7): softplus(pred) + smooth_l1.
        pred_gamma = F.softplus(recon_x[:, 6:7])
        tgt_gamma = target_x[:, 6:7].clamp(min=1e-3)
        loss_gamma = F.smooth_l1_loss(pred_gamma, tgt_gamma, reduction='mean')

        # rgb (7:10): L1 × 1.0 — clamp predictions to the legal Albedo range.
        recon_rgb = recon_x[:, 7:10].clamp(0.0, 1.0)
        loss_rgb = F.l1_loss(recon_rgb, target_x[:, 7:10], reduction='mean')

        return 0.01 * loss_dv + 0.1 * loss_delta + loss_gamma + loss_rgb

    def _geom_mat_recon_loss(self, recon_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        """Combined supervision for 12-channel branch: geometry + material."""
        if recon_x.shape[1] < 12 or target_x.shape[1] < 12:
            return self._shape_recon_loss(recon_x, target_x, feature_mode="geom6")
        loss_geom = self._shape_recon_loss(recon_x[:, :6], target_x[:, :6], feature_mode="geom6")
        loss_mat = self._material_recon_loss(recon_x[:, 6:12], target_x[:, 6:12])
        return loss_geom + loss_mat

    def forward(
        self, 
        recon_x: torch.Tensor, 
        target_x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        feature_mode: str = "geom6",
        rho_logits_list=None,
        rho_targets_list=None,
    ) -> dict:
        """
        Tính toán Loss dựa trên đầu ra của SC-VAE.
        
        Args:
            recon_x (Tensor): Tensor phục dựng [N, C] hoặc [B, C, X, Y, Z]
            target_x (Tensor): Tensor nhãn gốc (từ O-Voxel Dataset)
            mu (Tensor): Mean từ Encoder [B, Token_Dim]
            logvar (Tensor): Log-variance từ Encoder [B, Token_Dim]
            
        Returns:
            dict: Chứa 'loss' (loss tổng), 'recon_loss', và 'kl_loss'.
        """
        # Reconstruction Loss (mode-aware)
        if feature_mode == "mat6":
            recon_loss = self._material_recon_loss(recon_x, target_x)
        elif feature_mode == "shape_native":
            recon_loss = self._shape_native_recon_loss(recon_x, target_x)
        elif feature_mode == "shape_mat":
            recon_loss = self._shape_mat_recon_loss(recon_x, target_x)
        elif feature_mode == "geom_mat12":
            recon_loss = self._geom_mat_recon_loss(recon_x, target_x)
        else:
            recon_loss = self._shape_recon_loss(recon_x, target_x, feature_mode)
            
        # KL Divergence Loss
        # D_KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Numerical stabilisation: clamp logvar to a safe range before exp() to avoid
        # FP16/BF16 overflow under AMP (BAQ-VAE / TRELLIS.2 standard practice).
        logvar_safe = logvar.clamp(min=-30.0, max=20.0)
        kl_loss = -0.5 * torch.sum(1.0 + logvar_safe - mu.pow(2) - logvar_safe.exp())

        # Normalize per-element to make `kl_weight` invariant to batch / latent dim.
        # Previous code divided by `target_x.shape[0]` (= number of voxels), which
        # silently inflated the effective KL by a factor of `latent_dim` (32) compared
        # to the documented formulation. Dividing by `mu.numel() = N * latent_dim`
        # matches both the report's Eq. (VAE-2) and the TRELLIS.2 default
        # `lambda_kl=1e-6` calibration.
        kl_loss = kl_loss / max(int(mu.numel()), 1)

        rho_loss = recon_loss.new_zeros(())
        if rho_logits_list is not None and rho_targets_list is not None:
            terms = []
            for logits, targets in zip(rho_logits_list, rho_targets_list):
                if logits is None or targets is None:
                    continue
                if logits.numel() == 0 or targets.numel() == 0:
                    continue
                terms.append(F.binary_cross_entropy_with_logits(logits, targets, reduction='mean'))
            if len(terms) > 0:
                rho_loss = torch.stack(terms).mean()

        # Tổng hợp Loss.
        # NOTE: ``rho_loss`` KHÔNG được cộng vào ``loss`` ở đây — `train_sc_vae.py`
        # blend rho-warmup vào tổng (`+ rho_scale * rho_loss_weight * rho_loss`)
        # để có thể tắt/bật theo curriculum mà không cần khởi tạo lại loss_fn.
        # Nếu dùng SCVAELoss bên ngoài training loop, hãy nhớ cộng `rho_loss` thủ công.
        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "rho_loss": rho_loss,
        }

if __name__ == "__main__":
    # UNIT TEST nội bộ theo quy tắc Rule 6 (Bắt buộc chạy thử Code)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SC-VAE Loss Test] Using device: {device}")
    
    criterion = SCVAELoss(kl_weight=1e-4)
    
    # Mock data [10000 points, 6 channels]
    target_data = torch.rand(10000, 6, device=device)
    recon_data = target_data + torch.randn_like(target_data) * 0.1 # Phục dựng sai số nhẹ
    
    mu_mock = torch.zeros(10000, 16, device=device)
    logvar_mock = torch.ones(10000, 16, device=device) * -1
    
    mem_before = torch.cuda.max_memory_allocated(device) if device != "cpu" else 0
    loss_dict = criterion(recon_data, target_data, mu_mock, logvar_mock)
    mem_after = torch.cuda.max_memory_allocated(device) if device != "cpu" else 0
    
    print("\n[Kết quả Loss]")
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"Recon Loss: {loss_dict['recon_loss'].item():.6f}")
    print(f"KL Loss:    {loss_dict['kl_loss'].item():.6f}")
    
    if device != "cpu":
        print(f"[Memory Tracking] Gradient calculation footprint VRAM: {(mem_after - mem_before) / (1024**2):.2f} MB")
