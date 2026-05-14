import torch
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None


_LPIPS_MODEL = None
_LPIPS_DISABLED_DEVICES = set()


def _is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _get_lpips_model(device: torch.device):
    global _LPIPS_MODEL
    if not LPIPS_AVAILABLE:
        return None
    device_key = str(device)
    if device_key in _LPIPS_DISABLED_DEVICES:
        return None
    if _LPIPS_MODEL is None:
        try:
            _LPIPS_MODEL = lpips.LPIPS(net='alex')
            _LPIPS_MODEL.eval()
            for p in _LPIPS_MODEL.parameters():
                p.requires_grad_(False)
        except Exception as exc:
            print(f"[Train] Warning: LPIPS init failed, fallback to L1/SSIM only: {exc}")
            _LPIPS_MODEL = False
            return None
    if _LPIPS_MODEL is False:
        return None
    try:
        return _LPIPS_MODEL.to(device)
    except Exception as exc:
        if _is_oom_error(exc):
            _LPIPS_DISABLED_DEVICES.add(device_key)
            print(
                f"[Train] Warning: LPIPS disabled on {device_key} due to OOM. "
                "Falling back to L1/SSIM-only perceptual loss."
            )
            try:
                _LPIPS_MODEL.to("cpu")
            except Exception:
                pass
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return None
        raise


def _project_features_to_map(
    points_xyz: torch.Tensor,
    features: torch.Tensor,
    axes: tuple,
    image_size: int,
) -> torch.Tensor:
    """Chiếu các đặc trưng trên mỗi điểm (per-point features) lên bản đồ đặc trưng trực giao (orthographic feature map)."""
    eps = 1e-6
    h = int(image_size)
    w = int(image_size)
    u_idx, v_idx, d_idx = axes

    points = points_xyz.clamp(-1.0, 1.0)
    u = points[:, u_idx]
    v = points[:, v_idx]
    depth = points[:, d_idx].to(features.dtype)

    ix = ((u + 1.0) * 0.5 * (w - 1)).round().to(torch.int64)
    iy = ((v + 1.0) * 0.5 * (h - 1)).round().to(torch.int64)
    ix = torch.clamp(ix, 0, w - 1)
    iy = torch.clamp(iy, 0, h - 1)
    flat_idx = iy * w + ix

    counts = torch.zeros(h * w, device=features.device, dtype=features.dtype)
    counts.scatter_add_(0, flat_idx, torch.ones_like(depth, dtype=features.dtype))

    feat_acc = torch.zeros(features.shape[1], h * w, device=features.device, dtype=features.dtype)
    feat_acc.index_add_(1, flat_idx, features.transpose(0, 1))
    feat_map = (feat_acc / (counts.unsqueeze(0) + eps)).view(features.shape[1], h, w)

    depth_acc = torch.zeros(h * w, device=features.device, dtype=features.dtype)
    depth_acc.scatter_add_(0, flat_idx, depth)
    depth_map = (depth_acc / (counts + eps)).view(1, h, w)
    mask_map = (counts > 0).to(features.dtype).view(1, h, w)

    return torch.cat([mask_map, depth_map, feat_map], dim=0)


def _depth_to_normal(
    depth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Tính pháp tuyến bề mặt từ bản đồ độ sâu bằng sai phân hữu hạn (finite differences).

    Args:
        depth: [B, 1, H, W] depth map (trực giao - orthographic)
        mask:  [B, 1, H, W] mặt nạ chiếm đóng nhị phân (binary occupancy mask)
    Returns:
        normals: [B, 3, H, W] bản đồ pháp tuyến (chỉ hợp lệ tại vùng mask > 0)
    """
    # Sai phân trung tâm trên depth (central finite differences)
    pad_d = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    dz_dx = (pad_d[:, :, 1:-1, 2:] - pad_d[:, :, 1:-1, :-2]) * 0.5
    dz_dy = (pad_d[:, :, 2:, 1:-1] - pad_d[:, :, :-2, 1:-1]) * 0.5
    # Normal = normalize(-dz/dx, -dz/dy, 1)
    normals = torch.cat([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
    normals = F.normalize(normals, dim=1, eps=1e-6)
    # Chỉ giữ pháp tuyến tại vùng mask > 0
    normals = normals * mask
    return normals


def _ssim_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Hàm loss SSIM nhỏ cho các bản đồ đặc trưng được chiếu (tensor CHW ưu tiên lô - batch-first CHW tensors)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(tgt, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(tgt * tgt, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * tgt, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    ssim_n = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1.0 - ssim_map) * 0.5, min=0.0, max=1.0).mean()


def _lpips_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """LPIPS trên các bản đồ được chiếu sử dụng góc nhìn 3 kênh (3-channel view) của bản đồ đặc trưng."""
    if pred.ndim != 4 or tgt.ndim != 4:
        return pred.new_zeros(())

    def to_lpips_3ch(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] >= 5:
            view = x[:, 2:5]
        elif x.shape[1] >= 3:
            view = x[:, :3]
        else:
            pad = x.new_zeros((x.shape[0], 3 - x.shape[1], x.shape[2], x.shape[3]))
            view = torch.cat([x, pad], dim=1)
        return view.clamp(-1.0, 1.0)

    model = _get_lpips_model(pred.device)
    if model is None:
        return pred.new_zeros(())

    try:
        return model(to_lpips_3ch(pred), to_lpips_3ch(tgt)).mean()
    except Exception as exc:
        if _is_oom_error(exc):
            device_key = str(pred.device)
            _LPIPS_DISABLED_DEVICES.add(device_key)
            print(
                f"[Train] Warning: LPIPS forward disabled on {device_key} due to OOM. "
                "Falling back to L1/SSIM-only perceptual loss."
            )
            try:
                model.to("cpu")
            except Exception:
                pass
            if pred.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return pred.new_zeros(())
        raise


def _d_p_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Khoảng cách nhận thức (Perceptual distance) được sử dụng trong phụ lục bài báo: L1 + 0.2*SSIM + 0.2*LPIPS."""
    l1 = F.l1_loss(pred, tgt)
    ssim = _ssim_loss(pred, tgt)
    lp = _lpips_loss(pred, tgt)
    return l1 + 0.2 * ssim + 0.2 * lp


def _build_render_targets(
    recon_s: torch.Tensor,
    target_s: torch.Tensor,
    pts_ref: torch.Tensor,
    axes: tuple,
    feature_mode: str,
    image_size: int,
):
    """Xây dựng các tensor kết xuất (render tensors) lấy cảm hứng từ bài báo cho một góc nhìn trực giao duy nhất."""
    if feature_mode in {"geom6", "shape_native", "geom_mat12", "shape_mat"} and target_s.shape[1] >= 6:
        pred_channels = recon_s[:, 3:6]
        tgt_channels = target_s[:, 3:6]
    elif target_s.shape[1] >= 6:
        pred_channels = recon_s[:, :6]
        tgt_channels = target_s[:, :6]
    else:
        pred_channels = recon_s[:, :3]
        tgt_channels = target_s[:, :3]

    def _composite(points_xyz: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        mask_map = _project_features_to_map(
            points_xyz,
            torch.ones((points_xyz.shape[0], 1), device=channels.device, dtype=channels.dtype),
            axes,
            image_size,
        )
        depth_map = _project_features_to_map(
            points_xyz,
            points_xyz[:, axes[2]].unsqueeze(-1),
            axes,
            image_size,
        )
        feat_map = _project_features_to_map(points_xyz, channels, axes, image_size)
        return torch.cat([mask_map, depth_map, feat_map], dim=0)

    pred_render = _composite(pts_ref, pred_channels)
    tgt_render = _composite(pts_ref, tgt_channels)
    return pred_render, tgt_render


def compute_stage2_render_perceptual_loss(
    recon_x: torch.Tensor,
    target_x: torch.Tensor,
    batch_items,
    feature_mode: str,
    image_size: int,
    num_views: int,
):
    """Xấp xỉ hàm loss kết xuất giai đoạn 2 (stage-2 render loss approximation) lấy cảm hứng từ bài báo, phù hợp với Phương trình (8)."""
    if recon_x.numel() == 0:
        zero = recon_x.new_zeros(())
        return {
            "render_loss": zero,
            "perceptual_loss": zero,
        }

    view_axes = [(0, 1, 2), (2, 1, 0)]
    if int(num_views) >= 3:
        view_axes.append((0, 2, 1))

    # Cắt (pre-slice) trước các tensor trên mỗi mẫu (per-sample) một lần. Điều này giữ cho vòng lặp trên mỗi góc nhìn (per-view loop) chặt chẽ và
    # cho phép gom lô (batching) LPIPS/SSIM trên toàn bộ micro-batch để đạt được mức độ sử dụng GPU cao hơn nhiều
    # (tránh việc gọi LPIPS hàng trăm lần với batch=1).
    samples = []
    offset = 0
    for item in batch_items:
        n = int(item["features"].shape[0])
        if n <= 0:
            continue
        recon_s = recon_x[offset:offset + n]
        target_s = target_x[offset:offset + n]
        offset += n

        coords = item.get("coords", None)
        if coords is not None:
            coords_f = coords.to(device=recon_s.device, dtype=recon_s.dtype)
            coords_max = torch.clamp(coords_f.max(), min=1.0)
            pts_ref = (coords_f / coords_max) * 2.0 - 1.0
        elif target_s.shape[1] >= 3:
            pts_ref = target_s[:, :3].clamp(-1.0, 1.0)
        else:
            continue

        if (
            feature_mode in {"geom6", "shape_native", "geom_mat12"}
            and recon_s.shape[1] >= 3
            and target_s.shape[1] >= 3
        ):
            pts_pred = recon_s[:, :3].clamp(-1.0, 1.0)
            pts_tgt = target_s[:, :3].clamp(-1.0, 1.0)
        elif (
            feature_mode == "shape_mat"
            and recon_s.shape[1] >= 3
            and target_s.shape[1] >= 3
            and coords is not None
        ):
            # TRELLIS.2 spec: use the activated dv offset for accurate surface positioning
            # Matches FlexiDualGridVaeDecoder activation:
            #   dv = (1 + 2*voxel_margin) * sigmoid(h) - voxel_margin   with m=0.5
            # i.e. predicted dv may extend slightly outside [0,1] (∈[-0.5, 1.5]) — exactly
            # what the dual contouring extractor expects for sub-voxel boundary fitting.
            from src.models.sc_vae import apply_dv_activation, TRELLIS2_VOXEL_MARGIN
            pred_dv = apply_dv_activation(recon_s[:, 0:3], voxel_margin=TRELLIS2_VOXEL_MARGIN)
            # GT dv lives in [0,1] inside the cell; clamp covers any rounding noise.
            tgt_dv = target_s[:, 0:3].clamp(-TRELLIS2_VOXEL_MARGIN, 1.0 + TRELLIS2_VOXEL_MARGIN)
            # Project to [-1, 1] using the same scaling for both branches so the
            # render-loss compares apples-to-apples.
            scale_denom = torch.clamp(coords_f.max() + 1.0, min=1.0)
            pts_pred = ((coords_f + pred_dv.clamp(0.0, 1.0)) / scale_denom) * 2.0 - 1.0
            pts_tgt = ((coords_f + tgt_dv.clamp(0.0, 1.0)) / scale_denom) * 2.0 - 1.0
        else:
            pts_pred = pts_ref
            pts_tgt = pts_ref

        samples.append({
            "pts_ref": pts_ref,
            "pts_pred": pts_pred,
            "pts_tgt": pts_tgt,
            "recon_s": recon_s,
            "target_s": target_s,
        })

    if len(samples) == 0:
        zero = recon_x.new_zeros(())
        return {
            "render_loss": zero,
            "perceptual_loss": zero,
        }

    render_loss_acc = recon_x.new_zeros(())
    perceptual_loss_acc = recon_x.new_zeros(())
    views_used = 0

    for axes in view_axes:
        if feature_mode in {"geom6", "shape_native"}:
            pred_mask_list = []
            tgt_mask_list = []
            pred_depth_list = []
            tgt_depth_list = []
            pred_norm_list = []
            tgt_norm_list = []

            for s in samples:
                recon_s = s["recon_s"]
                target_s = s["target_s"]
                if recon_s.shape[1] < 6 or target_s.shape[1] < 6:
                    continue
                pred_map = _project_features_to_map(s["pts_pred"], recon_s[:, 3:6], axes, image_size)
                tgt_map = _project_features_to_map(s["pts_tgt"], target_s[:, 3:6], axes, image_size)

                pred_mask_list.append(pred_map[0:1])
                tgt_mask_list.append(tgt_map[0:1])
                pred_depth_list.append(pred_map[1:2])
                tgt_depth_list.append(tgt_map[1:2])
                pred_norm_list.append(pred_map[2:5])
                tgt_norm_list.append(tgt_map[2:5])

            if len(pred_norm_list) == 0:
                continue

            pred_mask = torch.stack(pred_mask_list, dim=0)
            tgt_mask = torch.stack(tgt_mask_list, dim=0)
            pred_depth = torch.stack(pred_depth_list, dim=0)
            tgt_depth = torch.stack(tgt_depth_list, dim=0)
            pred_norm = torch.stack(pred_norm_list, dim=0)
            tgt_norm = torch.stack(tgt_norm_list, dim=0)

            render_loss_acc = render_loss_acc + F.l1_loss(pred_mask, tgt_mask) + 10.0 * F.l1_loss(pred_depth, tgt_depth)
            perceptual_loss_acc = perceptual_loss_acc + _d_p_loss(pred_norm, tgt_norm)
            # Depth-to-normal loss (λ_normal=1, TRELLIS.2 Eq.8)
            pred_surf_n = _depth_to_normal(pred_depth, pred_mask)
            tgt_surf_n = _depth_to_normal(tgt_depth, tgt_mask)
            joint_m = ((pred_mask > 0) & (tgt_mask > 0)).float()
            render_loss_acc = render_loss_acc + F.l1_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            perceptual_loss_acc = perceptual_loss_acc + _d_p_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            views_used += 1
            continue

        if feature_mode == "rgb3":
            pred_c_list = []
            tgt_c_list = []

            for s in samples:
                recon_s = s["recon_s"]
                target_s = s["target_s"]
                if recon_s.shape[1] < 3 or target_s.shape[1] < 3:
                    continue
                pts_ref = s["pts_ref"]
                pred_cmap = _project_features_to_map(pts_ref, recon_s[:, 0:3], axes, image_size)
                tgt_cmap = _project_features_to_map(pts_ref, target_s[:, 0:3], axes, image_size)
                pred_c_list.append(pred_cmap[2:5])
                tgt_c_list.append(tgt_cmap[2:5])

            if len(pred_c_list) == 0:
                continue

            pred_c = torch.stack(pred_c_list, dim=0)
            tgt_c = torch.stack(tgt_c_list, dim=0)
            perceptual_loss_acc = perceptual_loss_acc + _d_p_loss(pred_c, tgt_c)
            views_used += 1
            continue

        if feature_mode == "mat6":
            pred_c_list = []
            tgt_c_list = []
            pred_mra_list = []
            tgt_mra_list = []

            for s in samples:
                recon_s = s["recon_s"]
                target_s = s["target_s"]
                if recon_s.shape[1] < 6 or target_s.shape[1] < 6:
                    continue
                pts_ref = s["pts_ref"]
                pred_cmra = _project_features_to_map(pts_ref, recon_s[:, 0:6], axes, image_size)
                tgt_cmra = _project_features_to_map(pts_ref, target_s[:, 0:6], axes, image_size)
                pred_c_list.append(pred_cmra[2:5])
                tgt_c_list.append(tgt_cmra[2:5])
                pred_mra_list.append(pred_cmra[5:8])
                tgt_mra_list.append(tgt_cmra[5:8])

            if len(pred_c_list) == 0:
                continue

            pred_c = torch.stack(pred_c_list, dim=0)
            tgt_c = torch.stack(tgt_c_list, dim=0)
            pred_mra = torch.stack(pred_mra_list, dim=0)
            tgt_mra = torch.stack(tgt_mra_list, dim=0)
            perceptual_loss_acc = perceptual_loss_acc + _d_p_loss(pred_c, tgt_c) + _d_p_loss(pred_mra, tgt_mra)
            views_used += 1
            continue

        if feature_mode == "geom_mat12":
            pred_mask_list = []
            tgt_mask_list = []
            pred_depth_list = []
            tgt_depth_list = []
            pred_norm_list = []
            tgt_norm_list = []
            pred_c_list = []
            tgt_c_list = []
            pred_mra_list = []
            tgt_mra_list = []

            for s in samples:
                recon_s = s["recon_s"]
                target_s = s["target_s"]
                if recon_s.shape[1] < 12 or target_s.shape[1] < 12:
                    continue
                pred_nmra = _project_features_to_map(s["pts_pred"], recon_s[:, 3:12], axes, image_size)
                tgt_nmra = _project_features_to_map(s["pts_tgt"], target_s[:, 3:12], axes, image_size)

                pred_mask_list.append(pred_nmra[0:1])
                tgt_mask_list.append(tgt_nmra[0:1])
                pred_depth_list.append(pred_nmra[1:2])
                tgt_depth_list.append(tgt_nmra[1:2])
                pred_norm_list.append(pred_nmra[2:5])
                tgt_norm_list.append(tgt_nmra[2:5])
                pred_c_list.append(pred_nmra[5:8])
                tgt_c_list.append(tgt_nmra[5:8])
                pred_mra_list.append(pred_nmra[8:11])
                tgt_mra_list.append(tgt_nmra[8:11])

            if len(pred_norm_list) == 0:
                continue

            pred_mask = torch.stack(pred_mask_list, dim=0)
            tgt_mask = torch.stack(tgt_mask_list, dim=0)
            pred_depth = torch.stack(pred_depth_list, dim=0)
            tgt_depth = torch.stack(tgt_depth_list, dim=0)
            pred_norm = torch.stack(pred_norm_list, dim=0)
            tgt_norm = torch.stack(tgt_norm_list, dim=0)
            pred_c = torch.stack(pred_c_list, dim=0)
            tgt_c = torch.stack(tgt_c_list, dim=0)
            pred_mra = torch.stack(pred_mra_list, dim=0)
            tgt_mra = torch.stack(tgt_mra_list, dim=0)

            render_loss_acc = render_loss_acc + F.l1_loss(pred_mask, tgt_mask) + 10.0 * F.l1_loss(pred_depth, tgt_depth)
            # Depth-to-normal loss (λ_normal=1, TRELLIS.2 Eq.8)
            pred_surf_n = _depth_to_normal(pred_depth, pred_mask)
            tgt_surf_n = _depth_to_normal(tgt_depth, tgt_mask)
            joint_m = ((pred_mask > 0) & (tgt_mask > 0)).float()
            render_loss_acc = render_loss_acc + F.l1_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            perceptual_loss_acc = (
                perceptual_loss_acc
                + _d_p_loss(pred_norm, tgt_norm)
                + _d_p_loss(pred_c, tgt_c)
                + _d_p_loss(pred_mra, tgt_mra)
                + _d_p_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            )
            views_used += 1
            continue

        if feature_mode == "shape_mat":
            pred_mask_list = []
            tgt_mask_list = []
            pred_depth_list = []
            tgt_depth_list = []
            pred_norm_list = []
            tgt_norm_list = []
            pred_c_list = []
            tgt_c_list = []

            for s in samples:
                recon_s = s["recon_s"]
                target_s = s["target_s"]
                # Support both 10-channel and 11-channel (quad_lerp) output
                if recon_s.shape[1] < 10 or target_s.shape[1] < 10:
                    continue

                # Flags (ch 3:6) are raw logits per TRELLIS.2 SC-VAE output spec.
                # For the (differentiable) render-loss path we use sigmoid so gradients
                # flow into the intersection head; the inference path uses (logit > 0)
                # exactly as in FlexiDualGridVaeDecoder eval mode.
                pred_flags = torch.sigmoid(recon_s[:, 3:6])
                tgt_flags = target_s[:, 3:6].clamp(0.0, 1.0)
                
                # Thay thế hình học (Geometry proxy) từ các kênh hình dạng (shape channels).
                pred_shape = _project_features_to_map(s["pts_pred"], pred_flags, axes, image_size)
                tgt_shape = _project_features_to_map(s["pts_tgt"], tgt_flags, axes, image_size)
                pred_mask_list.append(pred_shape[0:1])
                tgt_mask_list.append(tgt_shape[0:1])
                pred_depth_list.append(pred_shape[1:2])
                tgt_depth_list.append(tgt_shape[1:2])
                pred_norm_list.append(pred_shape[2:5])
                tgt_norm_list.append(tgt_shape[2:5])

                # RGB projection uses dv-corrected positions (consistent with shape)
                pred_rgb = _project_features_to_map(s["pts_pred"], recon_s[:, 7:10], axes, image_size)
                tgt_rgb = _project_features_to_map(s["pts_tgt"], target_s[:, 7:10], axes, image_size)
                pred_c_list.append(pred_rgb[2:5])
                tgt_c_list.append(tgt_rgb[2:5])

            if len(pred_norm_list) == 0:
                continue

            pred_mask = torch.stack(pred_mask_list, dim=0)
            tgt_mask = torch.stack(tgt_mask_list, dim=0)
            pred_depth = torch.stack(pred_depth_list, dim=0)
            tgt_depth = torch.stack(tgt_depth_list, dim=0)
            pred_norm = torch.stack(pred_norm_list, dim=0)
            tgt_norm = torch.stack(tgt_norm_list, dim=0)
            pred_c = torch.stack(pred_c_list, dim=0)
            tgt_c = torch.stack(tgt_c_list, dim=0)

            render_loss_acc = render_loss_acc + F.l1_loss(pred_mask, tgt_mask) + 10.0 * F.l1_loss(pred_depth, tgt_depth)
            # Depth-to-normal loss (λ_normal=1, TRELLIS.2 Eq.8)
            pred_surf_n = _depth_to_normal(pred_depth, pred_mask)
            tgt_surf_n = _depth_to_normal(tgt_depth, tgt_mask)
            joint_m = ((pred_mask > 0) & (tgt_mask > 0)).float()
            render_loss_acc = render_loss_acc + F.l1_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            perceptual_loss_acc = (
                perceptual_loss_acc
                + _d_p_loss(pred_norm, tgt_norm)
                + _d_p_loss(pred_c, tgt_c)
                + _d_p_loss(pred_surf_n * joint_m, tgt_surf_n * joint_m)
            )
            views_used += 1
            continue


        # Luồng dự phòng (các chế độ đặc trưng hiếm - rare feature modes): giữ nguyên tính toán gốc cho từng mẫu (per-sample computation),
        # nhưng với việc cắt lớp kênh (channel slicing) chính xác cho mặt nạ/độ sâu (mask/depth) (các tensor CHW).
        for s in samples:
            recon_s = s["recon_s"]
            target_s = s["target_s"]
            pred_render, tgt_render = _build_render_targets(
                recon_s,
                target_s,
                s["pts_ref"],
                axes,
                feature_mode,
                image_size,
            )
            render_loss_acc = render_loss_acc + F.l1_loss(pred_render, tgt_render)
            perceptual_loss_acc = perceptual_loss_acc + _d_p_loss(pred_render.unsqueeze(0), tgt_render.unsqueeze(0))
        views_used += 1

    if views_used == 0:
        zero = recon_x.new_zeros(())
        return {
            "render_loss": zero,
            "perceptual_loss": zero,
        }

    render_loss = render_loss_acc / views_used
    perceptual_loss = perceptual_loss_acc / views_used
    return {
        "render_loss": render_loss,
        "perceptual_loss": perceptual_loss,
    }
