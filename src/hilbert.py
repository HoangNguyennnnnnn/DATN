"""
Hilbert Space-Filling Curve cho Voxel Grid 3D
==============================================
Tính toán thứ tự Hilbert cho lưới 3D kích thước 2^p × 2^p × 2^p.

Đặc tính cốt lõi: điểm gần nhau trong không gian 3D → gần nhau trong chuỗi 1D.
GRU/Mamba xử lý tuần tự → hidden state mang context từ token lân cận hiệu quả hơn.

Ví dụ: grid 16×16×16 (4096 tokens)
  - Raster order: token[256] = (0,1,0), token[255] = (0,0,15) → xa nhau trong 3D nhưng liền nhau
  - Hilbert order: đảm bảo token liền nhau luôn là neighbor trong 3D

Tham khảo: Skilling, J. (2004). "Programming the Hilbert curve."

VRAM: Chỉ lưu 2 tensor int64 kích thước 4096 = 64KB. Không đáng kể.
"""

import torch
import numpy as np
from functools import lru_cache
from typing import Tuple


def _hilbert_3d_distance(x: int, y: int, z: int, order: int) -> int:
    """
    Tính Hilbert distance cho điểm (x, y, z) trong grid 2^order.
    
    Thuật toán Skilling (2004): transpose-based, hoạt động với mọi số chiều.
    
    Args:
        x, y, z: Tọa độ nguyên trong [0, 2^order - 1]
        order: Bậc của đường cong (grid_size = 2^order)
    
    Returns:
        Hilbert distance (int) trong [0, 2^(3*order) - 1]
    """
    n = 3  # số chiều
    coords = [int(x), int(y), int(z)]
    
    # Bước 1: Inverse undo — biến đổi tọa độ Cartesian sang transposed form
    M = 1 << (order - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q >>= 1
    
    # Bước 2: Gray encode
    for i in range(1, n):
        coords[i] ^= coords[i - 1]
    
    t = 0
    Q = M
    while Q > 1:
        if coords[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coords[i] ^= t
    
    # Bước 3: Interleave bits — chuyển từ transposed form sang 1D distance
    h = 0
    for bit in range(order - 1, -1, -1):
        for dim in range(n):
            h <<= 1
            if coords[dim] & (1 << bit):
                h |= 1
    
    return h


@lru_cache(maxsize=8)
def compute_hilbert_permutations(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính bảng hoán vị Hilbert cho grid 3D. Kết quả cached (chỉ tính 1 lần).
    
    Args:
        grid_size: Kích thước grid mỗi chiều (phải là lũy thừa 2, vd: 16)
    
    Returns:
        hilbert_to_raster: array[N], hilbert_to_raster[h] = raster index
        raster_to_hilbert: array[N], raster_to_hilbert[r] = hilbert distance
    """
    assert grid_size > 0 and (grid_size & (grid_size - 1)) == 0, \
        f"grid_size phải là lũy thừa 2, nhận {grid_size}"
    
    order = int(np.log2(grid_size))
    total = grid_size ** 3
    
    raster_to_hilbert = np.zeros(total, dtype=np.int64)
    for r in range(total):
        z = r // (grid_size * grid_size)
        y = (r // grid_size) % grid_size
        x = r % grid_size
        h = _hilbert_3d_distance(x, y, z, order)
        raster_to_hilbert[r] = h
    
    # Inverse permutation
    hilbert_to_raster = np.zeros(total, dtype=np.int64)
    hilbert_to_raster[raster_to_hilbert] = np.arange(total)
    
    return hilbert_to_raster, raster_to_hilbert


def get_hilbert_permutation_tensors(
    grid_size: int = 16,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về tensor hoán vị Hilbert trên device chỉ định.
    
    Returns:
        hilbert_to_raster: LongTensor[N] — dùng để reorder raster → hilbert
        raster_to_hilbert: LongTensor[N] — dùng để inverse hilbert → raster
    
    Cách dùng:
        h2r, r2h = get_hilbert_permutation_tensors(16, device)
        x_hilbert = x[:, h2r, :]    # raster → hilbert order
        x_raster  = x[:, r2h, :]    # hilbert → raster order (nghịch đảo)
        
        # Đúng vì: x_raster[:, r, :] = x_hilbert[:, r2h[r], :]
        #           = x_orig[:, h2r[r2h[r]], :] = x_orig[:, r, :]  ✓
    """
    h2r, r2h = compute_hilbert_permutations(grid_size)
    return (
        torch.from_numpy(h2r).long().to(device),
        torch.from_numpy(r2h).long().to(device),
    )


def verify_hilbert_locality(grid_size: int = 16) -> dict:
    """
    Kiểm tra tính locality của Hilbert ordering so với raster ordering.
    Trả về thống kê khoảng cách 3D giữa token liền kề.
    """
    h2r, _ = compute_hilbert_permutations(grid_size)
    
    def _raster_to_xyz(r: int) -> Tuple[int, int, int]:
        z = r // (grid_size * grid_size)
        y = (r // grid_size) % grid_size
        x = r % grid_size
        return x, y, z
    
    # Tính khoảng cách Manhattan giữa token liền kề
    hilbert_dists = []
    raster_dists = []
    for i in range(len(h2r) - 1):
        # Hilbert ordering
        x1, y1, z1 = _raster_to_xyz(h2r[i])
        x2, y2, z2 = _raster_to_xyz(h2r[i + 1])
        hilbert_dists.append(abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1))
        
        # Raster ordering
        x1, y1, z1 = _raster_to_xyz(i)
        x2, y2, z2 = _raster_to_xyz(i + 1)
        raster_dists.append(abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1))
    
    return {
        "hilbert_mean_dist": float(np.mean(hilbert_dists)),
        "hilbert_max_dist": int(np.max(hilbert_dists)),
        "raster_mean_dist": float(np.mean(raster_dists)),
        "raster_max_dist": int(np.max(raster_dists)),
        "unique_hilbert": len(set(h2r.tolist())),
        "total_tokens": grid_size ** 3,
    }


if __name__ == "__main__":
    stats = verify_hilbert_locality(16)
    print("=== Hilbert vs Raster Locality (16×16×16 grid) ===")
    print(f"  Hilbert: mean neighbor dist = {stats['hilbert_mean_dist']:.2f}, max = {stats['hilbert_max_dist']}")
    print(f"  Raster:  mean neighbor dist = {stats['raster_mean_dist']:.2f}, max = {stats['raster_max_dist']}")
    print(f"  Unique indices: {stats['unique_hilbert']}/{stats['total_tokens']}")
    print(f"  Hilbert locality improvement: {stats['raster_mean_dist']/stats['hilbert_mean_dist']:.1f}x better")
