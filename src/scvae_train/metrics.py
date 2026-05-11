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
