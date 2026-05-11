import os
import torch
import numpy as np


def _resolve_reconstruction_meta(item, spatial_size: int, device: torch.device):
    """Phân giải hộp bao chuẩn (AABB) trên mỗi mẫu và độ phân giải lưới với các cơ chế dự phòng an toàn."""
    aabb = item.get("aabb", None) if isinstance(item, dict) else None
    if isinstance(aabb, torch.Tensor) and aabb.numel() == 6:
        aabb_t = aabb.to(device=device, dtype=torch.float32)
    elif aabb is not None:
        aabb_t = torch.as_tensor(aabb, device=device, dtype=torch.float32)
    else:
        aabb_t = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device, dtype=torch.float32)

    resolution = spatial_size
    if isinstance(item, dict):
        raw_resolution = item.get("resolution", spatial_size)
        if isinstance(raw_resolution, torch.Tensor) and raw_resolution.numel() > 0:
            resolution = int(raw_resolution.reshape(-1)[0].item())
        elif isinstance(raw_resolution, (int, float)):
            resolution = int(raw_resolution)
        elif hasattr(raw_resolution, "item"):
            try:
                resolution = int(raw_resolution.item())
            except Exception:
                resolution = int(spatial_size)

    return aabb_t, max(2, int(resolution))

def save_validation_samples(
    model, 
    val_items, 
    device, 
    epoch, 
    output_dir, 
    num_samples=3, 
    spatial_size=256,
    feature_mode="shape_native"
):
    """
    Trích xuất và lưu các bản tái tạo lưới có màu từ các mẫu xác thực (validation samples).
    
    Tham số:
        model: Mô hình SC-VAE trong chế độ đánh giá (eval mode)
        val_items: Danh sách các mục dữ liệu thô từ DataLoader
        device: Thiết bị Torch (Torch device)
        epoch: Số epoch hiện tại
        output_dir: Thư mục gốc để lưu các mẫu
        num_samples: Số lượng lưới cần tái tạo
        spatial_size: Độ phân giải của lưới O-Voxel
        feature_mode: Bố cục đặc trưng O-Voxel (O-Voxel feature layout)
    """
    model.eval()
    samples_path = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(samples_path, exist_ok=True)
    
    # Chúng ta lấy một vài mục đầu tiên từ lô (batch)
    subset = val_items[:num_samples]
    
    from src.scvae_train.data import build_sparse_batch
    
    try:
        # Xây dựng một lô thưa thớt nhỏ (small sparse batch) chỉ dành cho các mẫu này
        sparse_input, _ = build_sparse_batch(subset, device=device, spatial_size=spatial_size)
        
        with torch.no_grad():
            recon, _, _, _, _ = model(sparse_input)
        
        # Chúng ta cần một đối tượng generator giả (dummy generator instance) để tái sử dụng logic Dual Contouring của nó
        # Hoặc chúng ta có thể chỉ cần gọi logic _voxel_to_mesh nội bộ nếu làm cho nó có thể truy cập được
        # Để đơn giản, chúng ta sẽ triển khai một trình ghi lưới độc lập (standalone mesh writer) ở đây sử dụng trimesh hoặc định dạng OBJ thô
        
        # Lấy các chỉ số thưa thớt (sparse indices) cho lô
        indices = sparse_input.indices # [N, 4] -> (B, Z, Y, X)
        
        for b in range(len(subset)):
            mask = (indices[:, 0] == b)
            b_feats = recon[mask]
            b_indices = indices[mask]
            
            if b_feats.shape[0] == 0:
                continue
                
            # Tái tạo sử dụng Dual Contouring nếu có thể
            try:
                from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh

                if not torch.cuda.is_available():
                    print("  [Visual] Skipping Dual Contouring on CPU (current o_voxel build is CUDA-reliable only).")
                    continue

                dc_device = torch.device("cuda")
                b_feats = b_feats.detach().to(dtype=torch.float32)
                b_indices = b_indices.to(dtype=torch.int32)
                
                # Cắt các kênh dựa trên feature_mode
                # shape_native: [v(3), delta(3), gamma(1), rgb(3)]
                v = torch.clamp(b_feats[:, :3], 0.0, 1.0).to(dc_device)
                delta = (b_feats[:, 3:6] > 0.0).to(dc_device)

                split_weight = None
                if b_feats.shape[1] >= 7:
                    gamma_raw = torch.nn.functional.softplus(b_feats[:, 6:7]).to(dc_device)
                    if gamma_raw.numel() > 0:
                        gamma_pos = gamma_raw.clamp_min(1e-3)
                        gamma_span = float((gamma_pos.max() - gamma_pos.min()).item())
                        if gamma_span > 1e-6:
                            split_weight = gamma_pos
                
                # RGB
                colors = None
                if b_feats.shape[1] >= 10:
                    colors = torch.clamp(b_feats[:, 7:10], 0.0, 1.0).cpu().numpy()
                
                coords = b_indices[:, 1:4].to(device=dc_device, dtype=torch.int32)
                aabb, grid_size = _resolve_reconstruction_meta(subset[b], spatial_size, dc_device)
                
                # Gọi DC (Dual Contouring)
                verts, faces = flexible_dual_grid_to_mesh(
                    coords,
                    v,
                    delta,
                    split_weight,
                    aabb=aabb,
                    grid_size=int(grid_size),
                )

                verts_np = verts.cpu().numpy()
                faces_np = faces.cpu().numpy().astype(np.int64)

                if len(faces_np) > 0 and len(verts_np) > 0:
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

                if len(faces_np) == 0:
                    print(f"  [Visual] Sample {b}: Dual Contouring returned empty faces after cleanup.")
                    continue

                vertex_colors = None
                if colors is not None and len(verts_np) > 0:
                    try:
                        from scipy.spatial import cKDTree

                        voxel_size = (aabb[1] - aabb[0]) / float(max(int(grid_size), 1))
                        dual_vertices_world = (coords.to(torch.float32) + v) * voxel_size.unsqueeze(0) + aabb[0].unsqueeze(0)
                        support_points = dual_vertices_world.detach().cpu().numpy()
                        k = int(max(1, min(8, len(support_points))))
                        tree = cKDTree(support_points)
                        dist, nn_idx = tree.query(verts_np, k=k)
                        if k == 1:
                            nn_idx = np.asarray(nn_idx).reshape(-1, 1)
                            dist = np.asarray(dist).reshape(-1, 1)
                        weights = 1.0 / np.maximum(np.asarray(dist, dtype=np.float32), 1e-6)
                        weights /= weights.sum(axis=1, keepdims=True)
                        vertex_colors_linear = (colors[np.asarray(nn_idx, dtype=np.int64)] * weights[..., None]).sum(axis=1)
                        vertex_colors_linear = np.clip(vertex_colors_linear, 0.0, 1.0)
                        
                        # Apply Gamma Correction (1/2.2)
                        vertex_colors = np.power(vertex_colors_linear, 1.0 / 2.2)
                    except Exception:
                        vertex_colors = np.repeat(colors[:1], repeats=len(verts_np), axis=0)
                
                # Lưu định dạng OBJ
                out_name = os.path.join(samples_path, f"sample_{b}.obj")
                with open(out_name, "w") as f:
                    for i, vert in enumerate(verts_np):
                        if vertex_colors is not None:
                            c = vertex_colors[i]
                            f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                        else:
                            f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
                    for face in faces_np:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                        
            except Exception as e:
                print(f"  [Visual] Failed to reconstruct mesh for sample {b}: {e}")
                
    except Exception as e:
        print(f"  [Visual] Batch visualization failed: {e}")
