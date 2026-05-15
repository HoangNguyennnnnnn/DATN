import torch
import numpy as np
from scipy.spatial import cKDTree

def compute_chamfer_distance(recon_xyz: torch.Tensor, target_xyz: torch.Tensor):
    """
    Tính toán Khoảng cách Chamfer (Chamfer Distance - CD) hai chiều giữa hai đám mây điểm.
    Việc tính toán được thực hiện trên CPU sử dụng cKDTree để đạt hiệu quả cao trong quá trình xác thực (validation).
    
    Tham số:
        recon_xyz: [N, 3] tọa độ điểm được tái tạo (Tensor)
        target_xyz: [M, 3] tọa độ điểm mục tiêu (Tensor)
        
    Trả về:
        float: Khoảng cách Chamfer hai chiều.
    """
    if recon_xyz.shape[0] == 0 or target_xyz.shape[0] == 0:
        return 0.0
        
    p1 = recon_xyz.detach().cpu().numpy()
    p2 = target_xyz.detach().cpu().numpy()
    
    # Chiều đi (Forward): Khoảng cách từ p1 đến p2
    tree2 = cKDTree(p2)
    dist1, _ = tree2.query(p1)
    ch_1_to_2 = np.mean(dist1**2)
    
    # Chiều về (Backward): Khoảng cách từ p2 đến p1
    tree1 = cKDTree(p1)
    dist2, _ = tree1.query(p2)
    ch_2_to_1 = np.mean(dist2**2)
    
    return float(ch_1_to_2 + ch_2_to_1)

def compute_voxel_iou(rho_logits: torch.Tensor, rho_targets: torch.Tensor, threshold: float = 0.5):
    """
    Tính toán tỷ lệ Giao trên Hợp của Voxel (Voxel Intersection over Union - IoU) cho đầu cắt tỉa (pruning) rho.
    
    Tham số:
        rho_logits: [N, 1] Logits dự đoán sự chiếm dụng (occupancy)
        rho_targets: [N, 1] Sự chiếm dụng mục tiêu (0 hoặc 1)
        threshold: Ngưỡng Sigmoid cho sự chiếm dụng
        
    Trả về:
        float: Điểm số Voxel IoU.
    """
    pred = (torch.sigmoid(rho_logits) > threshold).bool()
    target = rho_targets.bool()
    
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    
    if union == 0:
        return 1.0
    return float(intersection / union)

def compute_normal_consistency(recon_nrm: torch.Tensor, target_nrm: torch.Tensor):
    """
    Tính toán Tính nhất quán của Pháp tuyến (Normal Consistency - NC) - trung bình tích vô hướng (dot product) của các pháp tuyến.
    
    Tham số:
        recon_nrm: [N, 3] Pháp tuyến dự đoán
        target_nrm: [N, 3] Pháp tuyến thực tế (Ground truth)
    """
    if recon_nrm.shape[1] < 3 or target_nrm.shape[1] < 3:
        return 0.0
        
    # Chuẩn hóa (Normalize)
    p1 = torch.nn.functional.normalize(recon_nrm.detach(), dim=-1)
    p2 = torch.nn.functional.normalize(target_nrm.detach(), dim=-1)
    
    dot = (p1 * p2).sum(dim=-1)
    return float(dot.abs().mean().item()) # Lấy trị tuyệt đối (Abs) để bỏ qua hướng nếu cần, hoặc bỏ abs để tính toán nghiêm ngặt hơn


# ============================================================
# TRELLIS.2-aligned Metrics (Section D.1.1)
# ============================================================

def compute_f_score(
    recon_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    tau: float = 1e-6,
) -> dict:
    """Tính toán F-score (Precision, Recall, F1) theo công thức TRELLIS.2 Eq. (12-13).

    F-score đánh giá sự tương ứng hình dạng bằng cách kết hợp
    Precision (tỷ lệ điểm recon nằm gần GT) và Recall (tỷ lệ điểm GT
    được bao phủ bởi recon), với ngưỡng khoảng cách τ.

    TRELLIS.2 dùng τ=1e-6 cho Chamfer Distance F-score
    và τ=1e-8 cho Mesh Distance F-score.

    Args:
        recon_xyz: [N, 3] tọa độ điểm tái tạo
        target_xyz: [M, 3] tọa độ điểm mục tiêu
        tau: Ngưỡng khoảng cách bình phương (squared distance threshold)

    Returns:
        dict với 'precision', 'recall', 'f_score'
    """
    if recon_xyz.shape[0] == 0 or target_xyz.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    p_recon = recon_xyz.detach().cpu().numpy()
    p_target = target_xyz.detach().cpu().numpy()

    # Precision: tỷ lệ điểm recon nằm trong τ của GT
    tree_gt = cKDTree(p_target)
    dist_r2g, _ = tree_gt.query(p_recon)
    precision = float(np.mean(dist_r2g ** 2 < tau))

    # Recall: tỷ lệ điểm GT nằm trong τ của recon
    tree_recon = cKDTree(p_recon)
    dist_g2r, _ = tree_recon.query(p_target)
    recall = float(np.mean(dist_g2r ** 2 < tau))

    # F-score = harmonic mean
    if precision + recall > 0:
        f_score = 2.0 * precision * recall / (precision + recall)
    else:
        f_score = 0.0

    return {"precision": precision, "recall": recall, "f_score": f_score}


def compute_mesh_distance(
    recon_verts: np.ndarray,
    recon_faces: np.ndarray,
    target_verts: np.ndarray,
    target_faces: np.ndarray,
    n_samples: int = 100_000,
) -> float:
    """Tính toán Mesh Distance (MD) hai chiều theo TRELLIS.2 Eq. (10).

    MD đo khoảng cách point-to-MESH (không chỉ point-to-point như Chamfer),
    chính xác hơn cho bề mặt liên tục. Sample điểm từ surface mesh rồi
    tính khoảng cách tới mesh đối diện.

    Yêu cầu: trimesh

    Args:
        recon_verts, recon_faces: Mesh tái tạo
        target_verts, target_faces: Mesh mục tiêu
        n_samples: Số điểm sample từ mỗi mesh surface

    Returns:
        float: Bidirectional Mesh Distance (mean squared)
    """
    try:
        import trimesh
    except ImportError:
        return -1.0

    mesh_recon = trimesh.Trimesh(vertices=recon_verts, faces=recon_faces, process=False)
    mesh_target = trimesh.Trimesh(vertices=target_verts, faces=target_faces, process=False)

    # Sample điểm từ bề mặt mesh
    pts_recon = mesh_recon.sample(n_samples)
    pts_target = mesh_target.sample(n_samples)

    # Khoảng cách point-to-mesh (closest point on surface)
    _, dist_r2t, _ = trimesh.proximity.closest_point(mesh_target, pts_recon)
    _, dist_t2r, _ = trimesh.proximity.closest_point(mesh_recon, pts_target)

    md = 0.5 * np.mean(dist_r2t ** 2) + 0.5 * np.mean(dist_t2r ** 2)
    return float(md)


def render_normal_map(
    verts: np.ndarray,
    faces: np.ndarray,
    image_size: int = 512,
    yaw: float = 30.0,
    pitch: float = 30.0,
    fov: float = 6.0,
    radius: float = 10.0,
) -> np.ndarray:
    """Render normal map từ mesh tại góc nhìn cho trước.

    Trả về ảnh normal map [H, W, 3] với RGB = (nx+1)/2, (ny+1)/2, (nz+1)/2.
    Background pixels có giá trị 0.

    Yêu cầu: open3d

    Args:
        verts, faces: Mesh geometry
        image_size: Kích thước ảnh output
        yaw, pitch: Góc quay camera (độ)
        fov: Field of View (độ)
        radius: Khoảng cách camera tới gốc tọa độ

    Returns:
        np.ndarray [H, W, 3] float32 in [0, 1]
    """
    try:
        import open3d as o3d
    except ImportError:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    # Tạo off-screen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "normals"
    render.scene.add_geometry("mesh", mesh, mat)

    # Setup camera
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    eye = np.array([
        radius * np.cos(pitch_rad) * np.cos(yaw_rad),
        radius * np.sin(pitch_rad),
        radius * np.cos(pitch_rad) * np.sin(yaw_rad),
    ])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    render.setup_camera(fov, center, eye, up)
    img = np.asarray(render.render_to_image())
    # Normalize to [0, 1]
    normal_map = img.astype(np.float32) / 255.0
    return normal_map


def compute_normal_psnr_lpips(
    recon_verts: np.ndarray,
    recon_faces: np.ndarray,
    target_verts: np.ndarray,
    target_faces: np.ndarray,
    image_size: int = 512,
    views: list = None,
    device: str = "cuda",
) -> dict:
    """Tính Normal PSNR và Normal LPIPS theo TRELLIS.2 Section D.1.1.

    Render normal maps từ 4 views cố định (yaw=30°, 120°, 210°, 300°,
    pitch=30°, FoV=6°), tính PSNR và LPIPS giữa recon vs GT.

    Args:
        recon_verts, recon_faces: Mesh tái tạo
        target_verts, target_faces: Mesh mục tiêu
        image_size: Kích thước render
        views: List of (yaw, pitch) tuples. Default: TRELLIS.2 protocol
        device: CUDA device cho LPIPS

    Returns:
        dict: 'normal_psnr' (dB), 'normal_lpips', 'per_view_psnr', 'per_view_lpips'
    """
    if views is None:
        # TRELLIS.2 default: 4 views, pitch=30°, yaw=[30°, 120°, 210°, 300°]
        views = [(30.0, 30.0), (120.0, 30.0), (210.0, 30.0), (300.0, 30.0)]

    psnr_list = []
    lpips_list = []

    # Lazy-load LPIPS
    lpips_fn = None
    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net='alex').to(device).eval()
    except Exception:
        pass

    for yaw, pitch in views:
        nmap_recon = render_normal_map(recon_verts, recon_faces, image_size, yaw, pitch)
        nmap_target = render_normal_map(target_verts, target_faces, image_size, yaw, pitch)

        # Tạo mask: chỉ tính trên pixel có geometry (non-black)
        mask_recon = np.sum(nmap_recon, axis=-1) > 0.01
        mask_target = np.sum(nmap_target, axis=-1) > 0.01
        mask = mask_recon | mask_target

        if mask.sum() < 100:
            # Quá ít pixel → skip view này
            continue

        # PSNR trên masked region
        diff_sq = (nmap_recon[mask] - nmap_target[mask]) ** 2
        mse = np.mean(diff_sq)
        if mse < 1e-10:
            psnr = 50.0  # Cap tại 50 dB
        else:
            psnr = float(10.0 * np.log10(1.0 / mse))
        psnr_list.append(psnr)

        # LPIPS trên full image
        if lpips_fn is not None:
            try:
                # LPIPS yêu cầu [B, 3, H, W] in [-1, 1]
                t_recon = torch.from_numpy(nmap_recon).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
                t_target = torch.from_numpy(nmap_target).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
                with torch.no_grad():
                    lp = lpips_fn(t_recon, t_target).item()
                lpips_list.append(lp)
            except Exception:
                pass

    result = {
        "normal_psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "normal_lpips": float(np.mean(lpips_list)) if lpips_list else -1.0,
        "per_view_psnr": psnr_list,
        "per_view_lpips": lpips_list,
    }

    # Cleanup LPIPS model
    if lpips_fn is not None:
        del lpips_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result
