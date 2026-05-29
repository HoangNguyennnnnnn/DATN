"""
Improved Mean Flow (iMF) — Bản triển khai bám sát paper.
=========================================================
Bài báo: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
       (Geng et al., arXiv:2512.02012v1 — Z. Geng, Y. Lu, Z. Wu, E. Shechtman, J. Z. Kolter, K. He)

Khác biệt cốt lõi vs Rectified Flow:
- Mạng nơ-ron dự đoán **vận tốc trung bình** (average velocity) u(z, r, t), KHÔNG phải vận tốc tức thời (instantaneous velocity) v(z, t)
- Đầu vào có 3 tham số thời gian: z_t, r (thời gian bắt đầu), t (thời gian kết thúc)  
- Dùng JVP (Jacobian-Vector Product) để tính du/dt
- Dừng lan truyền ngược (Stop-gradient) trên du/dt để ổn định huấn luyện
- Lấy mẫu các cặp (r, t) với tỉ lệ 50% r≠t
- Suy luận: z_0 = z_1 - u_theta(z_1, r=0, t=1)  (1-step)

VRAM: JVP tạo thêm ~30% overhead so với MSE đơn thuần, nhưng chất lượng cao hơn đáng kể.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Hybrid context layout: ArcFace(512) + FLAME(50) + DINOv2(384)
HYBRID_ARC_DIM = 512
HYBRID_FLAME_DIM = 50
HYBRID_DINO_DIM = 384


def contrastive_target_dim(
    context_dim: int,
    mode: str = "arcface",
    arc_dim: int = HYBRID_ARC_DIM,
    flame_dim: int = HYBRID_FLAME_DIM,
) -> int:
    """Output dim của ctx_classifier theo mode contrastive."""
    m = (mode or "full").strip().lower()
    if m == "arcface":
        return int(arc_dim)
    if m == "flame":
        return int(flame_dim)
    return int(context_dim)


def slice_contrastive_context(
    context: torch.Tensor,
    mode: str = "arcface",
    arc_dim: int = HYBRID_ARC_DIM,
    flame_dim: int = HYBRID_FLAME_DIM,
) -> torch.Tensor:
    """Lấy khối context cho InfoNCE — mặc định chỉ ArcFace (audit: margin identity tốt nhất)."""
    m = (mode or "full").strip().lower()
    if m == "arcface":
        return context[..., :arc_dim]
    if m == "flame":
        return context[..., arc_dim : arc_dim + flame_dim]
    return context


def context_velocity_separation_loss(
    model: nn.Module,
    z_t: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    context: torch.Tensor,
    *,
    contrastive_mode: str = "arcface",
    margin: float = 0.0,
    omega: Optional[torch.Tensor] = None,
    cfg_tmin: Optional[torch.Tensor] = None,
    cfg_tmax: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Penalty >= 0: relu(cos(u(z,t,ctx_a), u(z,t,ctx_b)) - margin)^2 với cùng z_t,t,r.
    Không cộng cos thẳng (tránh total loss âm khi cos < 0).
    """
    b = int(context.shape[0])
    device = z_t.device
    if b < 2:
        z = torch.zeros((), device=device)
        return z, z

    arc = slice_contrastive_context(context, mode=contrastive_mode)
    arc_n = F.normalize(arc, dim=-1)
    sim_arc = (arc_n[0:1] @ arc_n.t()).squeeze(0)
    sim_arc[0] = -2.0
    j = int(sim_arc.argmin().item())
    if j == 0:
        j = 1

    z0 = z_t[0:1]
    t0 = t[0:1]
    r0 = r[0:1]
    if omega is None:
        omega0 = torch.ones_like(t0)
    else:
        omega0 = omega[0:1]
    if cfg_tmin is None:
        tmin0 = torch.zeros_like(t0)
    else:
        tmin0 = cfg_tmin[0:1]
    if cfg_tmax is None:
        tmax0 = torch.ones_like(t0)
    else:
        tmax0 = cfg_tmax[0:1]

    u_a = model(z0, t0, context[0:1], r=r0, omega=omega0, cfg_tmin=tmin0, cfg_tmax=tmax0)
    u_b = model(z0, t0, context[j : j + 1], r=r0, omega=omega0, cfg_tmin=tmin0, cfg_tmax=tmax0)
    cos = F.cosine_similarity(u_a.float().flatten(1), u_b.float().flatten(1), dim=1)
    loss = F.relu(cos - float(margin)).pow(2).mean()
    return loss, cos.mean().detach()


class ImprovedMeanFlow:
    """
    Khung làm việc Improved Mean Flow (iMF).
    
    Tư tưởng cốt lõi: thay vì dự đoán vận tốc tức thời v(z,t) = dz/dt,
    hãy dự đoán **vận tốc trung bình** u(z,r,t) = (1/(t-r)) * ∫[r→t] v(z_s, s) ds
    
    Đồng nhất thức MeanFlow: v = u + (t-r) * du/dt
    
    Hàm hợp V_θ: V = u_θ + (t-r) * stop_grad(du/dt)
    Hàm mất mát: ||V_θ(z_t) - (e - x)||²
    
    Lấy mẫu 1 bước (1-step sampling): z_0 = z_1 - u_θ(z_1, r=0, t=1)
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        ratio_r_neq_t: float = 0.5,
        t_sampler: str = "logit_normal",
        t_loc: float = -0.4,
        t_scale: float = 1.0,
        curriculum_switch_ratio: float = 0.6,
        curriculum_uniform_prob: float = 0.8,
        cfg_omega_min: float = 1.0,
        cfg_omega_max: float = 8.0,
        cfg_omega_power_beta: float = 1.0,
        enable_cfg_interval_conditioning: bool = True,
        adaptive_loss_weighting: bool = True,
        paper_strict_tr: bool = False,
        adaptive_loss_mode: str = "ema",
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
    ):
        """
        Tham số:
            sigma_min: Mức nhiễu tối thiểu
            ratio_r_neq_t: Tỷ lệ các mẫu có r≠t (bài báo dùng 0.5)
            t_sampler: 'uniform', 'logit_normal', hoặc 'curriculum'
            t_loc: Giá trị trung bình (mean) cho bộ lấy mẫu logit-normal
            t_scale: Tỉ lệ (scale) cho bộ lấy mẫu logit-normal
            curriculum_switch_ratio: Điểm chuyển chương trình giảng dạy (0-1 theo epoch tiến trình)
            curriculum_uniform_prob: Xác suất chọn đồng đều ở giai đoạn 2 của tiến trình
            cfg_omega_min: Thang đo điều hướng tối thiểu cho điều kiện hóa CFG linh hoạt
            cfg_omega_max: Thang đo điều hướng tối đa cho điều kiện hóa CFG linh hoạt
            cfg_omega_power_beta: Giá trị beta của hàm lũy thừa trong p(w) ~ w^-beta
            enable_cfg_interval_conditioning: Nếu True, lấy mẫu khoảng CFG [tmin, tmax]
        """
        self.sigma_min = sigma_min
        self.ratio_r_neq_t = ratio_r_neq_t
        self.t_sampler = t_sampler
        self.t_loc = t_loc
        self.t_scale = t_scale
        self.curriculum_switch_ratio = max(0.0, min(1.0, curriculum_switch_ratio))
        self.curriculum_uniform_prob = max(0.0, min(1.0, curriculum_uniform_prob))
        self.cfg_omega_min = float(max(1.0, cfg_omega_min))
        self.cfg_omega_max = float(max(self.cfg_omega_min, cfg_omega_max))
        self.cfg_omega_power_beta = float(max(0.0, cfg_omega_power_beta))
        self.enable_cfg_interval_conditioning = bool(enable_cfg_interval_conditioning)
        self.training_progress = 0.0

        # --- Adaptive loss weighting (MeanFlow paper, iMF Appendix A) ---
        # Maintains running EMA of per-timestep-bin loss to normalize
        # contributions, preventing high-variance bins from dominating.
        self.adaptive_loss_weighting = adaptive_loss_weighting
        self.paper_strict_tr = bool(paper_strict_tr)
        self.adaptive_loss_mode = str(adaptive_loss_mode).strip().lower()
        self.norm_p = float(norm_p)
        self.norm_eps = float(norm_eps)
        self._num_bins = 100  # discretize t into 100 bins
        self._loss_ema_decay = 0.99
        # Running EMA of loss per bin, initialized to 1.0 (neutral weight)
        self._loss_ema = torch.ones(self._num_bins)
        self._loss_counts = torch.zeros(self._num_bins, dtype=torch.long)

    def set_progress(self, progress: float) -> None:
        """Thiết lập tiến trình huấn luyện trong khoảng [0, 1] cho việc lấy mẫu dấu thời gian theo chương trình."""
        self.training_progress = max(0.0, min(1.0, float(progress)))

    def _sample_cfg_omega(
        self,
        batch_size: int,
        device: torch.device,
        omega_min: float,
        omega_max: float,
        beta: float,
    ) -> torch.Tensor:
        """Lấy mẫu thang đo điều hướng omega với p(w) ~ w^-beta trên đoạn [omega_min, omega_max]."""
        if omega_max <= omega_min + 1e-8:
            return torch.full((batch_size,), float(omega_min), device=device)

        u = torch.rand(batch_size, device=device).clamp_(1e-6, 1.0 - 1e-6)
        if abs(beta - 1.0) < 1e-6:
            ratio = omega_max / omega_min
            return omega_min * (ratio ** u)

        one_minus_beta = 1.0 - beta
        lo = omega_min ** one_minus_beta
        hi = omega_max ** one_minus_beta
        return (lo + (hi - lo) * u).clamp_min(1e-12) ** (1.0 / one_minus_beta)

    def _sample_cfg_interval(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lấy mẫu khoảng CFG [tmin, tmax] theo thực tiễn trong phụ lục của iMF."""
        tmin = torch.rand(batch_size, device=device) * 0.5
        tmax = 0.5 + torch.rand(batch_size, device=device) * 0.5
        return tmin, tmax

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Lấy mẫu dấu thời gian t sử dụng chính sách lấy mẫu đã cấu hình."""
        if self.t_sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.t_sampler == "curriculum":
            # Giai đoạn 1: chủ yếu dùng logit-normal để học nhanh cấu trúc thô.
            # Giai đoạn 2: trộn lẫn với uniform để bao phủ tốt hơn các dấu thời gian ở biên.
            if self.training_progress < self.curriculum_switch_ratio:
                u = torch.randn(batch_size, device=device) * self.t_scale + self.t_loc
                t = torch.sigmoid(u)
            else:
                use_uniform = torch.rand(batch_size, device=device) < self.curriculum_uniform_prob
                u = torch.randn(batch_size, device=device) * self.t_scale + self.t_loc
                t_logit = torch.sigmoid(u)
                t_uniform = torch.rand(batch_size, device=device)
                t = torch.where(use_uniform, t_uniform, t_logit)
        else:
            # Mặc định: logit-normal
            u = torch.randn(batch_size, device=device) * self.t_scale + self.t_loc
            t = torch.sigmoid(u)
        return t
    
    def _sample_t_r_imeanflow(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(t,r) như repo imeanflow: logit-normal, r≤t, batch-index 50% r=t."""
        t = self._sample_t(batch_size, device).clamp(self.sigma_min, 1.0 - self.sigma_min)
        r = self._sample_t(batch_size, device).clamp(0.0, 1.0 - self.sigma_min)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        n_fm = int(batch_size * (1.0 - self.ratio_r_neq_t))
        if n_fm > 0:
            fm_mask = torch.arange(batch_size, device=device) < n_fm
            r = torch.where(fm_mask, t, r)
        return t, r

    def _sample_t_r(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (t, r) pairs theo paper convention r ≤ t.

        Paper Geng et al. 2512.02012v1 — Tab. 4 ratio of r≠t = 50%.
        If paper_strict_tr: imeanflow sample_tr (no 15%/5% FaceDiff extras).
        """
        if self.paper_strict_tr:
            return self._sample_t_r_imeanflow(batch_size, device)

        """
        FaceDiff extended sampling (legacy):

        2026-05-21 v3 FIX: Two injection strategies for t≈1 coverage:

        Strategy A — "near-boundary" (15%): Force t = 1-σ with r = t (boundary).
        Teaches model velocity v(z_t, t≈1) = e-x via simple stable MSE.
        For linear flow matching: v = e-x is constant, so this also teaches the
        correct mean velocity u(z, 0, 1) = v = e-x.
        Avoids JVP instability from (t-r)≈1.

        Strategy B — "endpoint JVP" (5%): Force r=0, t=1-σ for JVP branch.
        Teaches u(z, 0, 1) directly via compound function. Small percentage
        limits exposure to JVP noise early in training.

        Total: 50% boundary | 25% random JVP | 15% t≈1 boundary | 5% r=0,t≈1 | 5% r→0 JVP
        """
        t = self._sample_t(batch_size, device).clamp(self.sigma_min, 1.0 - self.sigma_min)

        # r candidates for JVP: r ~ U[0, t]
        u = torch.rand(batch_size, device=device)
        r_candidate = (u * t).clamp(0.0, 1.0 - self.sigma_min)

        # Multi-way split with dice roll
        # FIX 2026-05-22 (Bug A): Gate all special-case masks behind ratio_r_neq_t > 0.
        # Previously endpoint_jvp_mask was HARDCODED at 5% regardless of config → with
        # ratio_r_neq_t=0 (pure boundary mode), JVP path still triggered for 5% of batches,
        # causing unexpected memory fragmentation + OOM.
        if self.ratio_r_neq_t <= 0:
            # Pure boundary mode — no JVP, no near-boundary forcing
            r = t.clone()
            return t, r

        dice = torch.rand(batch_size, device=device)
        # 0.00-0.15: near-boundary (r=t=1-σ, stable MSE at t≈1)
        # 0.15-0.20: endpoint JVP (r=0, t=1-σ, direct mean-velocity learning)
        # 0.20-ratio: random JVP (r<t, standard iMF)
        # ratio-1.00: boundary (r=t, standard v-loss)
        near_boundary_mask = dice < 0.15
        endpoint_jvp_mask = (dice >= 0.15) & (dice < 0.20)
        random_jvp_mask = (dice >= 0.20) & (dice < self.ratio_r_neq_t)
        boundary_mask = dice >= self.ratio_r_neq_t

        # Default: random JVP
        r = r_candidate.clone()

        # Boundary: r = t
        r = torch.where(boundary_mask, t, r)

        # Near-boundary: force t=1-σ AND r=t (enters boundary branch → stable MSE)
        t_near_one = torch.full_like(t, 1.0 - self.sigma_min)
        t = torch.where(near_boundary_mask, t_near_one, t)
        r = torch.where(near_boundary_mask, t_near_one, r)

        # Endpoint JVP: r=0, t=1-σ (enters JVP branch — small % to limit instability)
        r = torch.where(endpoint_jvp_mask, torch.zeros_like(r), r)
        t = torch.where(endpoint_jvp_mask, t_near_one, t)

        return t, r
    
    def _interpolate(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Nội suy (Interpolation) khớp luồng (Flow matching): z_t = (1-t)*x + t*e
        
        Quy ước của bài báo: x = dữ liệu (data), e = nhiễu (noise) (ngược với một số bản triển khai khác)
        z_t đi từ dữ liệu (t=0) → nhiễu (t=1)
        """
        t_exp = t.view(-1, *([1] * (x.ndim - 1)))
        return (1.0 - t_exp) * x + t_exp * e

    def _slat_position_weights(self, x_data: torch.Tensor) -> torch.Tensor:
        """Upweight non-zero slat rows (real voxels); mean weight = 1 per sample."""
        occ = (x_data.norm(dim=-1) > 1e-6).to(dtype=x_data.dtype)
        return occ / occ.mean(dim=-1, keepdim=True).clamp(min=1e-6)

    def _weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_weights: Optional[torch.Tensor],
        position_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE được đánh trọng số theo kênh và/hoặc vị trí token (slat occupancy)."""
        diff2 = (pred - target) ** 2
        if channel_weights is not None:
            w = channel_weights.view(*([1] * (pred.ndim - 1)), -1)
            diff2 = diff2 * w
        if position_weights is not None:
            diff2 = diff2 * position_weights.unsqueeze(-1)
        return diff2.mean()

    def _per_sample_weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_weights: Optional[torch.Tensor],
        position_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample MSE with optional channel and position weighting.

        Returns a tensor shaped [B] so adaptive weighting can be applied
        per sample/timestep instead of collapsing into a single scalar.
        """
        diff2 = (pred - target) ** 2
        if channel_weights is not None:
            w = channel_weights.view(*([1] * (pred.ndim - 1)), -1)
            diff2 = diff2 * w
        if position_weights is not None:
            diff2 = diff2 * position_weights.unsqueeze(-1)
        reduce_dims = tuple(range(1, pred.ndim))
        return diff2.mean(dim=reduce_dims)

    @torch.no_grad()
    def _get_adaptive_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Compute per-sample adaptive weight based on running loss EMA.

        Bins timestep *t* into ``_num_bins`` buckets and returns
        ``1 / max(loss_ema[bin], eps)`` so that timestep ranges with
        historically large loss get down-weighted, stabilising training.

        Reference: Original MeanFlow paper; iMF Appendix A:
        "As adaptive weighting [mf] is used in the MF loss, we also
        apply it to this auxiliary loss."
        """
        if not self.adaptive_loss_weighting:
            return torch.ones(t.shape[0], device=t.device)

        # Move EMA buffer to correct device lazily
        if self._loss_ema.device != t.device:
            self._loss_ema = self._loss_ema.to(t.device)
            self._loss_counts = self._loss_counts.to(t.device)

        bins = (t.clamp(0.0, 1.0 - 1e-6) * self._num_bins).long()
        ema_vals = self._loss_ema[bins].clamp(min=1e-4)
        # Weight = 1/ema → higher weight for low-loss bins (easy), lower for high-loss (hard)
        # This is the "normalizing" variant: each bin contributes roughly equally to total loss
        weights = 1.0 / ema_vals
        # Normalize so mean weight = 1 (keeps effective LR unchanged)
        weights = weights / weights.mean().clamp(min=1e-6)
        return weights

    @torch.no_grad()
    def _update_adaptive_ema(self, t: torch.Tensor, per_sample_loss: torch.Tensor) -> None:
        """Vectorised EMA update of per-timestep-bin loss using scatter ops.

        Trước đây dùng vòng for trên ``bins.unique()`` rất chậm khi batch lớn (mỗi
        bin gây 1 kernel launch). Giờ tính tổng/số đếm theo bin bằng
        ``scatter_add_`` rồi blend EMA một lần — O(B) thay vì O(unique_bins).
        """
        if not self.adaptive_loss_weighting:
            return

        if self._loss_ema.device != t.device:
            self._loss_ema = self._loss_ema.to(t.device)
            self._loss_counts = self._loss_counts.to(t.device)

        bins = (t.clamp(0.0, 1.0 - 1e-6) * self._num_bins).long()

        bin_sum = torch.zeros(self._num_bins, device=t.device, dtype=per_sample_loss.dtype)
        bin_cnt = torch.zeros(self._num_bins, device=t.device, dtype=per_sample_loss.dtype)
        bin_sum.scatter_add_(0, bins, per_sample_loss)
        bin_cnt.scatter_add_(0, bins, torch.ones_like(per_sample_loss))

        present = bin_cnt > 0
        if not bool(present.any().item()):
            return

        bin_avg = torch.where(present, bin_sum / bin_cnt.clamp_min(1.0), self._loss_ema)
        d = self._loss_ema_decay
        # Chỉ cập nhật những bin có sample trong batch này (giữ giá trị cũ cho phần còn lại).
        self._loss_ema = torch.where(
            present,
            d * self._loss_ema + (1.0 - d) * bin_avg.to(self._loss_ema.dtype),
            self._loss_ema,
        )
        self._loss_counts = self._loss_counts + bin_cnt.to(self._loss_counts.dtype)

    def _reduce_adaptive_loss(
        self,
        per_sample: torch.Tensor,
        t: torch.Tensor,
        per_sample_aux: Optional[torch.Tensor] = None,
        aux_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gộp per-sample loss + adaptive (EMA bin hoặc imeanflow loss/(loss+ε)^p)."""
        if not self.adaptive_loss_weighting:
            loss = per_sample.mean()
            if per_sample_aux is not None and aux_weight > 0:
                loss = loss + float(aux_weight) * per_sample_aux.mean()
            return loss, per_sample.new_ones(())

        if self.adaptive_loss_mode == "paper":
            denom = (per_sample.detach() + self.norm_eps).pow(self.norm_p)
            loss = (per_sample / denom).mean()
            scale = denom.mean()
            if per_sample_aux is not None and aux_weight > 0:
                denom_v = (per_sample_aux.detach() + self.norm_eps).pow(self.norm_p)
                loss = loss + float(aux_weight) * (per_sample_aux / denom_v).mean()
            return loss, scale.detach()

        with torch.no_grad():
            self._update_adaptive_ema(t, per_sample.detach())
        w = self._get_adaptive_weights(t).detach()
        loss = (per_sample * w).mean()
        if per_sample_aux is not None and aux_weight > 0:
            loss = loss + float(aux_weight) * (per_sample_aux * w).mean()
        return loss, w.mean()

    def _compute_dudt_jvp(
        self,
        model: nn.Module,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        context: torch.Tensor,
        v_tangent: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính toán du/dt thông qua JVP với tiếp tuyến (dz/dt=v, dr/dt=0, dt/dt=1)."""

        def fn(z_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return model(
                z_in,
                t_in,
                context,
                r=r,
                omega=omega,
                cfg_tmin=cfg_tmin,
                cfg_tmax=cfg_tmax,
            )

        _, dudt = torch.autograd.functional.jvp(
            fn,
            (z_t, t),
            (v_tangent, torch.ones_like(t)),
            create_graph=False,
            strict=False,
        )
        return dudt
    
    def compute_loss(
        self,
        model: nn.Module,
        x_data: torch.Tensor,
        context: torch.Tensor,
        v_head: Optional[nn.Module] = None,
        material_loss_weight: float = 1.0,
        dual_branch: bool = False,
        shape_dim: Optional[int] = None,
        material_condition_source: str = "gt",
        material_condition_dropout: float = 0.0,
        cfg_conditioning: bool = False,
        cfg_omega_min: Optional[float] = None,
        cfg_omega_max: Optional[float] = None,
        cfg_omega_power_beta: Optional[float] = None,
        cfg_interval_conditioning: Optional[bool] = None,
        cfg_context_dropout: float = 0.1,
        v_loss_weight: float = 0.1,
        ctx_classifier: Optional[nn.Module] = None,
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_mode: str = "arcface",
        context_velocity_sep_weight: float = 0.0,
        context_velocity_sep_margin: float = 0.0,
        occupancy_mask: Optional[torch.Tensor] = None,
        empty_weight_floor: float = 0.0,
        return_components: bool = False,
    ):
        """
        Tính toán hàm mất mát huấn luyện iMF (biến thể thực tế bám sát bài báo).

        Giao diện mô hình hiện tại dự đoán u(z_t, t, context, r).
        r được lấy mẫu và truyền vào các lời gọi mô hình, trong khi JVP được tính trên (z_t, t)
        với r được giữ cố định (dr/dt=0).
        
        Tham số:
            model: U-Net/Transformer dự đoán u(z,t,ctx,r).
            x_data: Các token Slat mục tiêu [B, L, D]
            context: Vector điều kiện hóa [B, C] (ArcFace + FLAME)
            v_head: Khối phụ trợ (tùy chọn) để dự đoán vận tốc cận biên (marginal velocity).
            material_loss_weight: Hệ số nhân nhánh vật liệu trong chế độ nhánh kép.
            dual_branch: Xác định các kênh tiềm ẩn là [shape | material].
            shape_dim: Số lượng kênh hình dáng (shape) trên chiều cuối cùng khi dual_branch=True.
            material_condition_source: "gt" hoặc "pred_detached".
            material_condition_dropout: Tỷ lệ giám sát vật liệu được thay thế bằng tự dự đoán độc lập (detached self-prediction) khi nguồn là "pred_detached".
            cfg_conditioning: Nếu True, bật điều kiện hóa CFG linh hoạt trong quá trình huấn luyện.
            cfg_omega_min/cfg_omega_max/cfg_omega_power_beta: Các giá trị ghi đè tùy chọn.
            cfg_interval_conditioning: Lựa chọn ghi đè tùy chọn để lấy mẫu khoảng [tmin, tmax].
            cfg_context_dropout: Tỉ lệ bỏ qua cho nhánh ngữ cảnh điều kiện.
            v_loss_weight: Trọng số auxiliary v-head loss.
            return_components: Nếu True, trả về dictionary với chi tiết từng thành phần vô hướng (scalar breakdown).
        """
        b = x_data.shape[0]
        device = x_data.device

        channel_weights = None
        shape_slice = None
        material_slice = None
        if dual_branch and shape_dim is not None and x_data.ndim >= 2:
            total_dim = int(x_data.shape[-1])
            shape_ch = max(0, min(int(shape_dim), total_dim))
            if shape_ch < total_dim:
                channel_weights = x_data.new_ones((total_dim,))
                channel_weights[shape_ch:] = float(max(0.0, material_loss_weight))
                shape_slice = slice(0, shape_ch)
                material_slice = slice(shape_ch, total_dim)

        # FIX 2026-05-21 (Finding #2): x_data is NORMALIZED slat in training (zero rows
        # become -mean/std, no longer ~0). Use external occupancy_mask computed BEFORE
        # normalization. Fallback to legacy detection if mask not provided.
        if occupancy_mask is not None:
            position_weights = occupancy_mask.to(device=device, dtype=x_data.dtype)
            if position_weights.ndim == 3 and position_weights.shape[-1] == 1:
                position_weights = position_weights.squeeze(-1)
            if position_weights.shape != x_data.shape[:2]:
                raise ValueError(
                    f"occupancy_mask shape {tuple(position_weights.shape)} must match x_data[:2] {tuple(x_data.shape[:2])}"
                )
            # FIX 2026-05-27: empty voxels (occ=0) từng có weight=0 → model KHÔNG học
            # "empty phải ≈ 0" → lúc sampling (không có mask) phun rác vào 79% vùng trống
            # → mesh noise. empty_weight_floor > 0 ép model học cả empty region.
            # blended = occ * 1.0 + (1-occ) * floor; chuẩn hóa mean=1.
            if empty_weight_floor > 0.0:
                occ = (position_weights > 0).to(dtype=x_data.dtype)
                position_weights = occ + (1.0 - occ) * float(empty_weight_floor)
            position_weights = position_weights / position_weights.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            position_weights = self._slat_position_weights(x_data)
        
        # Bước 1: Lấy mẫu nhiễu
        e = torch.randn_like(x_data)
        
        # Bước 2: Lấy mẫu các cặp (t, r)
        t, r = self._sample_t_r(b, device)
        
        # Bước 3: Nội suy z_t = (1-t)*x + t*e
        z_t = self._interpolate(x_data, e, t)

        # Bước 3.5: Điều kiện hóa điều hướng linh hoạt (Bài báo Mục 4.2), nếu được bật.
        cfg_enabled = bool(cfg_conditioning)
        omega_min = float(self.cfg_omega_min if cfg_omega_min is None else max(1.0, cfg_omega_min))
        omega_max = float(self.cfg_omega_max if cfg_omega_max is None else max(omega_min, cfg_omega_max))
        omega_beta = float(self.cfg_omega_power_beta if cfg_omega_power_beta is None else max(0.0, cfg_omega_power_beta))
        interval_enabled = bool(self.enable_cfg_interval_conditioning if cfg_interval_conditioning is None else cfg_interval_conditioning)

        if cfg_enabled:
            omega = self._sample_cfg_omega(b, device, omega_min, omega_max, omega_beta).to(dtype=t.dtype)
            if interval_enabled:
                cfg_tmin, cfg_tmax = self._sample_cfg_interval(b, device)
                in_interval = (t >= cfg_tmin) & (t <= cfg_tmax)
                omega_effective = torch.where(in_interval, omega, torch.ones_like(omega))
            else:
                cfg_tmin = torch.zeros_like(t)
                cfg_tmax = torch.ones_like(t)
                omega_effective = omega
        else:
            omega = torch.ones_like(t)
            cfg_tmin = torch.zeros_like(t)
            cfg_tmax = torch.ones_like(t)
            omega_effective = torch.ones_like(t)

        context_cond = context
        cfg_context_keep_ratio = torch.ones((), device=device)
        context_dropped = None  # [B] bool — official iMF cond_drop on v_g
        # Phase A: dropout ngữ cảnh (ArcFace→0 → null_ctx_tokens) KHÔNG phụ thuộc cfg_enabled.
        # Nếu chỉ bật dropout ở Phase B, backbone không học u_uncond → CFG sụp.
        drop_prob = float(max(0.0, min(1.0, cfg_context_dropout)))
        if drop_prob > 0.0:
            keep_mask = (torch.rand((b, 1), device=device) >= drop_prob).to(dtype=context.dtype)
            context_cond = context * keep_mask
            cfg_context_keep_ratio = keep_mask.mean()
            context_dropped = keep_mask.squeeze(-1) < 0.5
        
        # Bước 4: Mục tiêu vận tốc có điều kiện
        _want_hidden = (
            (v_head is not None or (ctx_classifier is not None and float(contrastive_loss_weight) > 0.0))
            and hasattr(model, 'get_hidden_state')
        )
        
        _fwd_kwargs = dict(
            r=t, omega=omega_effective, cfg_tmin=cfg_tmin, cfg_tmax=cfg_tmax,
        )
        if _want_hidden:
            _fwd_kwargs['return_hidden'] = True
        
        _fwd_out = model(z_t, t, context_cond, **_fwd_kwargs)
        
        if _want_hidden and isinstance(_fwd_out, tuple):
            v_theta, _cached_hidden = _fwd_out
        else:
            v_theta = _fwd_out
            _cached_hidden = None
        
        # Sửa lỗi: Hàm mục tiêu là luồng trung bình e - x_data
        # Không phụ thuộc vào v_theta.detach() để tránh vòng lặp tự dạy luẩn quẩn
        v_target = e - x_data  # [B, L, D]
        
        # Mục tiêu khớp luồng thô (Raw flow matching target) (e - x), độc lập với tăng cường CFG.
        # Được sử dụng cho việc giám sát v-head phụ trợ theo Phụ lục A của bài báo iMF:
        # "Chúng tôi gắn thêm một hàm mất mát phụ trợ ‖v_θ − (e − x)‖² vào head này"
        raw_v_target = e - x_data
        
        # === Phân nhánh: r=t (biên) vs r≠t (JVP) ===
        mask_eq = (r == t)  # [B], True khi r=t
        mask_eq_any = bool(mask_eq.any().item())
        mask_eq_all = bool(mask_eq.all().item())
        
        # Khi r=t: u(z,t,t) ≡ v(z,t) → hàm mất mát v đơn giản (simple v-loss)
        # Khi r≠t: hàm hợp V = u + (t-r)*sg(du/dt)
        
        v_target_loss = v_target
        material_keep_ratio = torch.ones((), device=device)
        if (
            dual_branch
            and shape_slice is not None
            and material_slice is not None
            and str(material_condition_source).lower() == "pred_detached"
        ):
            drop_prob = float(max(0.0, min(1.0, material_condition_dropout)))
            if drop_prob > 0.0:
                keep_mask = (
                    torch.rand((b, 1, 1), device=device) >= drop_prob
                ).to(dtype=v_target.dtype)
                v_target_loss = v_target.clone()
                gt_mat = v_target[..., material_slice]
                pred_mat = v_theta[..., material_slice].detach()
                v_target_loss[..., material_slice] = keep_mask * gt_mat + (1.0 - keep_mask) * pred_mat
                material_keep_ratio = keep_mask.mean()

        loss_shape = torch.zeros((), device=device)
        loss_material = torch.zeros((), device=device)
        if shape_slice is not None and material_slice is not None:
            loss_shape = F.mse_loss(v_theta[..., shape_slice], v_target[..., shape_slice])
            loss_material = F.mse_loss(v_theta[..., material_slice], v_target_loss[..., material_slice])

        loss_boundary = torch.zeros((), device=device)
        loss_jvp = torch.zeros((), device=device)
        loss_v_head = torch.zeros((), device=device)
        per_sample_v_head = None
        
        # Giám sát v-head phụ trợ (luôn luôn trên toàn bộ lô dữ liệu)
        # Theo Phụ lục A của iMF: v-head sử dụng mục tiêu FM thô (e - x), KHÔNG PHẢI mục tiêu v_target được tăng cường CFG
        # Tối ưu hiệu năng: tái sử dụng _cached_hidden từ forward pass đầu tiên,
        # tránh gọi model.get_hidden_state() — tiết kiệm 1 forward pass hoàn chỉnh.
        if v_head is not None:
            if _cached_hidden is not None:
                # Tái sử dụng hidden state đã tính từ forward pass #1 (cùng input z_t, t, ctx, r=t)
                v_head_pred = v_head(_cached_hidden)
                vh_diff2 = (v_head_pred - raw_v_target) ** 2 * position_weights.unsqueeze(-1)
                per_sample_v_head = vh_diff2.mean(dim=tuple(range(1, v_head_pred.ndim)))
                loss_v_head = per_sample_v_head.mean()
            elif hasattr(model, 'get_hidden_state'):
                # Fallback: gọi get_hidden_state() nếu model không hỗ trợ return_hidden
                h = model.get_hidden_state(
                    z_t, t, context,
                    r=t,
                    omega=omega_effective,
                    cfg_tmin=cfg_tmin,
                    cfg_tmax=cfg_tmax,
                )
                v_head_pred = v_head(h)
                vh_diff2 = (v_head_pred - raw_v_target) ** 2 * position_weights.unsqueeze(-1)
                per_sample_v_head = vh_diff2.mean(dim=tuple(range(1, v_head_pred.ndim)))
                loss_v_head = per_sample_v_head.mean()
            else:
                raise RuntimeError(
                    "v_head requires a backbone hidden-state interface. "
                    "Implement get_hidden_state() or forward(..., return_hidden=True) on the backbone."
                )

        # Contrastive auxiliary loss (2026-05-20): force backbone hidden state ENCODE context.
        # Diagnostic v8 ep14 phát hiện model degenerate "predict average magnitude" — loss giảm
        # nhưng identity differentiation = 0. Standard MSE loss không pressure context conditioning.
        # Solution: InfoNCE on (pooled_hidden → predicted_ctx) vs (true context). Force model
        # backbone phải encode discriminative context info trong hidden state.
        loss_contrastive = torch.zeros((), device=device)
        if ctx_classifier is not None and contrastive_loss_weight > 0.0 and _cached_hidden is not None and b >= 2:
            hidden_pooled = _cached_hidden.mean(dim=1)  # [B, hidden_dim]
            pred_ctx = ctx_classifier(hidden_pooled)
            ctx_tgt = slice_contrastive_context(context.detach(), mode=contrastive_mode)
            # Bỏ mẫu ArcFace = 0 (face detect fail)
            valid = ctx_tgt.norm(dim=-1) > 1e-3
            if int(valid.sum()) >= 2:
                pred_v = pred_ctx[valid]
                tgt_v = ctx_tgt[valid]
                pred_norm = F.normalize(pred_v, dim=-1)
                target_norm = F.normalize(tgt_v, dim=-1)
                sim_matrix = pred_norm @ target_norm.t()
                sim_matrix = sim_matrix / max(contrastive_temperature, 1e-4)
                labels = torch.arange(int(valid.sum()), device=device)
                loss_contrastive = F.cross_entropy(sim_matrix, labels)

        loss_context_sep = torch.zeros((), device=device)
        ctx_sep_cos_monitor = torch.zeros((), device=device)
        if float(context_velocity_sep_weight) > 0.0 and b >= 2:
            loss_context_sep, ctx_sep_cos_monitor = context_velocity_separation_loss(
                model,
                z_t,
                t,
                r,
                context_cond,
                contrastive_mode=contrastive_mode,
                margin=float(context_velocity_sep_margin),
                omega=omega_effective,
                cfg_tmin=cfg_tmin,
                cfg_tmax=cfg_tmax,
            )

        if mask_eq_all:
            # Tất cả đều là biên → hàm mất mát khớp luồng đơn giản
            per_sample_boundary = self._per_sample_weighted_mse(
                v_theta, v_target_loss, channel_weights, position_weights
            )
            loss, adaptive_w = self._reduce_adaptive_loss(
                per_sample_boundary,
                t,
                per_sample_v_head,
                float(v_loss_weight) if v_head is not None else 0.0,
            )
            if contrastive_loss_weight > 0.0:
                loss = loss + float(contrastive_loss_weight) * loss_contrastive
            if float(context_velocity_sep_weight) > 0.0:
                loss = loss + float(context_velocity_sep_weight) * loss_context_sep
            loss_boundary = per_sample_boundary.mean()
            if return_components:
                return {
                    "loss": loss,
                    "loss_boundary": loss_boundary.detach(),
                    "loss_jvp": loss_jvp.detach(),
                    "loss_shape": loss_shape.detach(),
                    "loss_material": loss_material.detach(),
                    "loss_v_head": loss_v_head.detach(),
                    "loss_contrastive": loss_contrastive.detach(),
                    "loss_context_sep": loss_context_sep.detach(),
                    "ctx_sep_cos": ctx_sep_cos_monitor.detach(),
                    "material_supervision_keep_ratio": material_keep_ratio.detach(),
                    "cfg_context_keep_ratio": cfg_context_keep_ratio.detach(),
                    "adaptive_weight_scale": adaptive_w.mean().detach(),
                }
            return loss
        
        # === Tính toán JVP cho r≠t ===
        # Cần bật requires_grad cho z_t và t để tính JVP
        z_t_jvp = z_t[~mask_eq].detach().requires_grad_(True)
        t_jvp = t[~mask_eq].detach().requires_grad_(True)
        r_jvp = r[~mask_eq]
        omega_jvp = omega_effective[~mask_eq]
        cfg_tmin_jvp = cfg_tmin[~mask_eq]
        cfg_tmax_jvp = cfg_tmax[~mask_eq]
        ctx_jvp = context_cond[~mask_eq] if context_cond.shape[0] == b else context_cond
        v_target_jvp = v_target_loss[~mask_eq]
        
        # Lan truyền xuôi cho dự đoán u (u prediction)
        u_pred = model(
            z_t_jvp,
            t_jvp,
            ctx_jvp,
            r=r_jvp,
            omega=omega_jvp,
            cfg_tmin=cfg_tmin_jvp,
            cfg_tmax=cfg_tmax_jvp,
        )

        # Tính du/dt thông qua JVP với tiếp tuyến (dz/dt=v, dr/dt=0, dt/dt=1).
        # v được xấp xỉ bởi v-head phụ trợ nếu có, ngược lại sử dụng dự đoán biên v_theta.
        if v_head is not None:
            v_tangent_jvp = v_head_pred[~mask_eq].detach()
        else:
            v_tangent_jvp = v_theta[~mask_eq].detach()
            
        try:
            dudt = self._compute_dudt_jvp(
                model,
                z_t_jvp,
                t_jvp,
                r_jvp,
                ctx_jvp,
                v_tangent_jvp,
                omega=omega_jvp,
                cfg_tmin=cfg_tmin_jvp,
                cfg_tmax=cfg_tmax_jvp,
            )
        except RuntimeError:
            # Dự phòng cho các môi trường/toán tử không hỗ trợ JVP đáng tin cậy.
            dt = 1e-3
            t_plus = (t_jvp + dt).clamp(max=1.0 - self.sigma_min)
            z_t_plus = self._interpolate(
                x_data[~mask_eq], e[~mask_eq], t_plus
            )
            with torch.no_grad():
                u_plus = model(
                    z_t_plus,
                    t_plus,
                    ctx_jvp,
                    r=r_jvp,
                    omega=omega_jvp,
                    cfg_tmin=cfg_tmin_jvp,
                    cfg_tmax=cfg_tmax_jvp,
                )
                dudt = (u_plus - u_pred.detach()) / dt
        
        # Hàm hợp (Compound function): V = u + (t-r) * stop_grad(du/dt)
        t_minus_r = (t_jvp - r_jvp).view(-1, *([1] * (u_pred.ndim - 1)))
        V = u_pred + t_minus_r * dudt.detach()  # dừng lan truyền ngược (stop_grad) trên dudt
        
        # Hàm mất mát JVP: ||V - v_target||²
        pos_jvp = position_weights[~mask_eq]
        loss_jvp = self._weighted_mse(V, v_target_jvp, channel_weights, pos_jvp)
        per_sample_jvp = self._per_sample_weighted_mse(V, v_target_jvp, channel_weights, pos_jvp)
        
        # Hàm mất mát biên (Boundary loss): ||v_theta - v_target||² (chỉ dành cho các mẫu có r=t)
        if mask_eq_any:
            pos_eq = position_weights[mask_eq]
            loss_boundary = self._weighted_mse(
                v_theta[mask_eq], v_target_loss[mask_eq], channel_weights, pos_eq
            )
            per_sample_boundary = self._per_sample_weighted_mse(
                v_theta[mask_eq], v_target_loss[mask_eq], channel_weights, pos_eq
            )
        else:
            loss_boundary = torch.tensor(0.0, device=device)
            per_sample_boundary = torch.zeros((0,), device=device)
        
        # --- Adaptive loss weighting (MeanFlow paper, iMF Appendix A) ---
        # CRITICAL FIX 2026-05-21: per_sample_raw MUST NOT be allocated inside torch.no_grad()
        # — that detaches main boundary/JVP loss from gradient graph. EMA update uses detached
        # copy explicitly to avoid contaminating model gradient.
        per_sample_raw = torch.zeros(b, device=device, dtype=per_sample_jvp.dtype)
        if mask_eq_any:
            per_sample_raw[mask_eq] = per_sample_boundary
        if (~mask_eq).any():
            per_sample_raw[~mask_eq] = per_sample_jvp

        loss, adaptive_scale = self._reduce_adaptive_loss(
            per_sample_raw,
            t,
            per_sample_v_head,
            float(v_loss_weight) if v_head is not None else 0.0,
        )
        if contrastive_loss_weight > 0.0:
            loss = loss + float(contrastive_loss_weight) * loss_contrastive
        if float(context_velocity_sep_weight) > 0.0:
            loss = loss + float(context_velocity_sep_weight) * loss_context_sep

        if return_components:
            return {
                "loss": loss,
                "loss_boundary": loss_boundary.detach(),
                "loss_jvp": loss_jvp.detach(),
                "loss_shape": loss_shape.detach(),
                "loss_material": loss_material.detach(),
                "loss_v_head": loss_v_head.detach(),
                "loss_contrastive": loss_contrastive.detach(),
                "loss_context_sep": loss_context_sep.detach(),
                "ctx_sep_cos": ctx_sep_cos_monitor.detach(),
                "material_supervision_keep_ratio": material_keep_ratio.detach(),
                "cfg_context_keep_ratio": cfg_context_keep_ratio.detach(),
                "adaptive_weight_scale": adaptive_scale.detach(),
            }
        return loss
    
    @torch.no_grad()
    def sample_1_step(
        self,
        model: nn.Module,
        context: torch.Tensor,
        shape: Tuple[int, ...],
        omega: Optional[torch.Tensor] = None,
        cfg_tmin: Optional[torch.Tensor] = None,
        cfg_tmax: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Lấy mẫu 1 bước iMF (1-Step Sampling): z_0 = z_1 - u_θ(z_1, r=0, t=1)
        
        Vì u_θ(z, t, t) ≡ v_θ(z, t) (điều kiện biên),
        và khi r=0, t=1: u là vận tốc trung bình trên toàn bộ quỹ đạo (trajectory),
        nên: z_0 = z_1 - u_θ(z_1, t=1)
        
        Bài báo: z₀ = z₁ - u_θ(z₁, r=0, t=1)
        Mô hình nhận t=1 (tương ứng pure noise → dự đoán hướng về phía dữ liệu)
        """
        device = context.device
        b = context.shape[0]

        def _ensure_batch_scalar(val, default: float) -> torch.Tensor:
            if val is None:
                return torch.full((b,), default, device=device, dtype=context.dtype)
            if not torch.is_tensor(val):
                return torch.full((b,), float(val), device=device, dtype=context.dtype)
            out = val.to(device=device, dtype=context.dtype)
            if out.ndim == 0:
                return out.expand(b)
            if out.shape[0] != b:
                return out.reshape(-1)[0].expand(b)
            return out

        # 2026-05-21 FIX: Defaults must match training when cfg_conditioning=False.
        # Training: omega=1.0, cfg_tmin=0.0, cfg_tmax=1.0
        # Old defaults: omega=4.0, cfg_tmin=0.2, cfg_tmax=0.8 ← CAUSED OOD!
        omega_b = _ensure_batch_scalar(omega, 1.0).clamp_min(1.0)
        cfg_tmin_b = _ensure_batch_scalar(cfg_tmin, 0.0)
        cfg_tmax_b = _ensure_batch_scalar(cfg_tmax, 1.0)
        
        # Bắt đầu từ nhiễu thuần túy (pure noise)
        z_1 = torch.randn(shape, device=device)
        r_0 = torch.zeros(b, device=device)
        t_1 = torch.ones(b, device=device)  # t=1 (kết thúc tại nhiễu)
        
        # Áp dụng CFG interval: khi t nằm ngoài [tmin, tmax], đặt omega=1 (tắt CFG)
        in_interval = (t_1 >= cfg_tmin_b) & (t_1 <= cfg_tmax_b)
        omega_effective = torch.where(in_interval, omega_b, torch.ones_like(omega_b))

        # Mô hình dự đoán vận tốc trung bình từ nhiễu → dữ liệu
        u_pred = model(
            z_1,
            t_1,
            context,
            r=r_0,
            omega=omega_effective,
            cfg_tmin=cfg_tmin_b,
            cfg_tmax=cfg_tmax_b,
        )
        
        # Bước nhảy 1 lần (One-step jump): z_0 = z_1 - u (từ nhiễu đến dữ liệu)
        z_0 = z_1 - u_pred
        
        return z_0
