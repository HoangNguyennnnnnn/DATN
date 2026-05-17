"""
GPU-accelerated mesh post-processing helpers.

Thay thế các CPU bottlenecks (scipy.cKDTree, Open3D Taubin, np.unique dedup)
bằng PyTorch GPU equivalents. Tận dụng VRAM đã được cấp phát thay vì idle 0% GPU util.

Functions:
  - gpu_knn_idw_colors(verts, voxel_pts, voxel_colors, k=8): KNN + IDW color transfer
  - gpu_taubin_smooth(verts, faces, iters, lam, mu): Taubin mesh smoothing
  - gpu_dedup_faces(faces): Remove duplicate triangles by sorted-vertex hash
"""
from __future__ import annotations

import torch


@torch.no_grad()
def gpu_knn_idw_colors(
    verts: torch.Tensor,
    voxel_pts: torch.Tensor,
    voxel_colors: torch.Tensor,
    k: int = 8,
    chunk: int = 8192,
) -> torch.Tensor:
    """Inverse-distance-weighted KNN color transfer trên GPU.

    Args:
        verts: [V, 3] vertex positions
        voxel_pts: [P, 3] reference voxel-center positions
        voxel_colors: [P, 3] RGB colors per voxel (linear or sRGB, same as caller)
        k: số neighbor (default 8)
        chunk: số verts xử lý cùng lúc để tránh OOM trên cdist [chunk, P]

    Returns:
        [V, 3] interpolated colors

    Complexity: O(V × P / chunk) gpu kernel launches × O(P) reduction per launch.
    Memory: chunk × P × 4 bytes peak (chunk=8192, P=1M → ~32 GB FP32 — chunk nhỏ hơn nếu P lớn).

    Tương đương `scipy.spatial.cKDTree.query` + IDW weighting.
    """
    assert verts.ndim == 2 and verts.shape[-1] == 3
    assert voxel_pts.ndim == 2 and voxel_pts.shape[-1] == 3
    assert voxel_colors.ndim == 2 and voxel_colors.shape[-1] == 3
    device = verts.device
    dtype = verts.dtype
    voxel_pts = voxel_pts.to(device=device, dtype=dtype)
    voxel_colors = voxel_colors.to(device=device, dtype=dtype)

    V = verts.shape[0]
    P = voxel_pts.shape[0]
    k = min(k, P)

    out = torch.empty((V, 3), device=device, dtype=dtype)
    # Auto-adjust chunk to fit free VRAM: ~30% of available, ép minimum 256 nếu P quá lớn.
    # cdist tạo [chunk, P] float32 = chunk*P*4 bytes. Cộng overhead topk ~2x.
    if device.type == "cuda":
        free_bytes = torch.cuda.mem_get_info(device)[0]
        bytes_per_row = P * 4 * 3  # cdist + topk intermediates
        max_chunk = max(256, int(free_bytes * 0.3 / bytes_per_row))
        chunk = min(chunk, max_chunk, 32_768)
        chunk = max(chunk, 256)  # tối thiểu 256 vert/chunk để tránh quá chậm

    for i in range(0, V, chunk):
        end = min(i + chunk, V)
        d = torch.cdist(verts[i:end], voxel_pts)              # [chunk, P]
        top_d, top_idx = d.topk(k, dim=1, largest=False)       # [chunk, k]
        w = 1.0 / top_d.clamp_min(1e-6)
        w = w / w.sum(dim=1, keepdim=True)                     # [chunk, k]
        nbr_colors = voxel_colors[top_idx]                     # [chunk, k, 3]
        out[i:end] = (nbr_colors * w.unsqueeze(-1)).sum(dim=1)
    return out


@torch.no_grad()
def gpu_taubin_smooth(
    verts: torch.Tensor,
    faces: torch.Tensor,
    iters: int = 6,
    lam: float = 0.5,
    mu: float = -0.53,
) -> torch.Tensor:
    """Taubin λ|μ mesh smoothing thuần PyTorch GPU.

    Args:
        verts: [V, 3] float
        faces: [F, 3] long
        iters: số iterations (mỗi iter gồm 2 steps: λ shrink + μ inflate)
        lam: positive shrinking step (default 0.5)
        mu: negative inflating step (default -0.53, theo Open3D default)

    Returns:
        [V, 3] smoothed verts

    Tương đương Open3D `filter_smooth_taubin`. Builds sparse adjacency từ faces một lần,
    sau đó iterate trên GPU.
    """
    assert verts.ndim == 2 and verts.shape[-1] == 3
    assert faces.ndim == 2 and faces.shape[-1] == 3
    device = verts.device
    V = verts.shape[0]
    faces = faces.long().to(device)

    # Build symmetric edge list từ 3 cạnh của mỗi face
    src = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                     faces[:, 1], faces[:, 2], faces[:, 0]])
    dst = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                     faces[:, 0], faces[:, 1], faces[:, 2]])
    # Dedup edges via hashing (V × V có thể overflow nếu V > 3M, dùng long)
    key = src * V + dst
    key_unique = torch.unique(key)
    src_u = key_unique // V
    dst_u = key_unique % V

    indices = torch.stack([src_u, dst_u], dim=0)
    values = torch.ones(indices.shape[1], device=device, dtype=verts.dtype)
    adj = torch.sparse_coo_tensor(indices, values, (V, V)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0).unsqueeze(-1)

    out = verts.clone().contiguous()
    for _ in range(iters):
        # λ shrink: out += lam * (avg_neighbor - out)
        neigh = torch.sparse.mm(adj, out) / deg
        out = out + lam * (neigh - out)
        # μ inflate: out += mu * (avg_neighbor - out)
        neigh = torch.sparse.mm(adj, out) / deg
        out = out + mu * (neigh - out)
    return out


@torch.no_grad()
def gpu_dedup_faces(faces: torch.Tensor) -> torch.Tensor:
    """Loại bỏ triangles trùng lặp (so sánh sau khi sort 3 verts).

    Args:
        faces: [F, 3] long

    Returns:
        [F', 3] long với F' ≤ F, không có triangle trùng

    Tương đương `np.unique(np.sort(faces, axis=1), axis=0)`.
    """
    assert faces.ndim == 2 and faces.shape[-1] == 3
    device = faces.device
    f = faces.long()
    f_sorted, _ = f.sort(dim=1)

    # Hash 3 verts thành 1 int64 key (tránh overflow: dùng f_sorted.max() + 1 làm base)
    base = int(f_sorted.max().item()) + 1
    key = f_sorted[:, 0] * (base * base) + f_sorted[:, 1] * base + f_sorted[:, 2]

    # Find first occurrence of each unique key via inverse_idx + scatter_reduce(amin)
    unique_key, inverse_idx = torch.unique(key, return_inverse=True)
    n_unique = unique_key.numel()
    perm = torch.arange(key.numel(), device=device, dtype=torch.long)
    first_occ = torch.full((n_unique,), key.numel(), device=device, dtype=torch.long)
    first_occ.scatter_reduce_(0, inverse_idx, perm, reduce="amin", include_self=True)
    return f[first_occ]
