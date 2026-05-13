# 🚀 Hướng Dẫn Thiết Lập Máy GPU Cloud — FaceDiff

> **Cấu hình máy**: RTX 3090 Ti (24GB VRAM) · 64GB RAM · 700GB SSD
> **Mục tiêu**: Resume SC-VAE Epoch 398 + Khởi động VoxelMamba (Stage 2)
> **Cập nhật**: 2026-05-13

---

## 📊 Tình Trạng Máy Server Cũ (RTX 4090)

| Tiến trình | Trạng thái | Chi tiết |
|---|---|---|
| `build_context_lmdb` | ✅ **Hoàn tất** | 20,939 / 20,968 mẫu (99.9%). Đã nén + upload Drive. |
| `upload ovoxel_cache.tar` | ✅ **Hoàn tất** | 272 GB trên Drive. |
| `upload hybrid_context_lmdb.tar` | ✅ **Hoàn tất** | 87 MB trên Drive. |
| `upload mesh_manifest.json` | ✅ **Hoàn tất** | 812 KB trên Drive. |
| SC-VAE Training | 🟢 **Đang chạy** | Step 621,300 · Epoch 398 · Loss: 0.1771 |

> [!NOTE]
> SC-VAE trên máy cũ vẫn đang chạy (Epoch 398). Checkpoint `latest_step.pt` (Epoch 397) đã có trên Drive.
> Bạn có thể resume từ Epoch 397 trên máy mới bất cứ lúc nào.

---

## 📦 Tổng Quan Dung Lượng Ổ Cứng (700 GB SSD)

| File/Thư mục | Kích thước | Ghi chú |
|---|---|---|
| Hệ điều hành + CUDA | ~30 GB | Đã có sẵn trên máy thuê |
| Conda + PyTorch + Dependencies | ~15 GB | Cài mới |
| `ovoxel_cache.tar` (tải về tạm) | **272 GB** | **Xóa ngay sau khi giải nén** |
| `ovoxel_cache_lmdb/` (giải nén) | **272 GB** | Dữ liệu chính cho SC-VAE |
| `hybrid_context.lmdb/` | 84 MB | Context 946-D cho VoxelMamba |
| `mesh_manifest.json` | 812 KB | Bản đồ chỉ mục mesh |
| Checkpoint SC-VAE | ~400 MB | Resume training |
| Checkpoints sinh ra khi train | ~2 GB | Dự trù cho 5 checkpoint |
| **Tổng tối đa** | **~591 GB** | ✅ Vừa đủ cho 700 GB |

> [!WARNING]
> **QUAN TRỌNG**: Bạn PHẢI **xóa file `ovoxel_cache.tar` ngay sau khi giải nén** để giải phóng 272 GB.
> Nếu không xóa, tổng sẽ là 863 GB → **tràn ổ cứng**.
> Hoặc sử dụng phương pháp **Streaming Extraction** (Bước 3, Phương án B) để không bao giờ lưu file `.tar`.

---

## 🔧 Các Bước Thiết Lập Chi Tiết

### Bước 1: Clone Source Code

```bash
# Clone repo từ GitHub
git clone https://github.com/HoangNguyennnnnnn/DATN.git ~/facediff
cd ~/facediff
```

### Bước 2: Cài Đặt Môi Trường Python

```bash
# 2.1 — Tạo môi trường Conda (Python 3.11 khuyến nghị)
conda create -n facediff python=3.11 -y
conda activate facediff

# 2.2 — Cài PyTorch + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 2.3 — Cài các dependencies chính
pip install -r requirements.txt

# 2.4 — Cài build tools (cần cho bước compile CUDA extensions)
sudo apt-get update && sudo apt-get install -y git build-essential cmake ninja-build

# 2.5 — Cài o_voxel (Microsoft TRELLIS.2) — cần nvcc (CUDA compiler)
#   Nếu máy chỉ có CUDA runtime (không có nvcc), cài thêm:
#   sudo apt-get install -y cuda-nvcc-12-4
bash scripts/install_o_voxel.sh

# 2.6 — Cài Mamba SSM (cho VoxelMamba Stage 2)
pip install -r requirements-mamba.txt

# 2.7 — Xác nhận mọi thứ hoạt động
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import spconv; print('spconv OK')"
python -c "import lmdb; print('lmdb OK')"
python -c "import o_voxel; print('o_voxel OK')"
```

> [!TIP]
> Nếu `install_o_voxel.sh` bị lỗi vì không có `nvcc`, bạn vẫn có thể train SC-VAE và iMF ở chế độ
> `--lmdb-only` (đọc dữ liệu từ LMDB đã pack sẵn, không cần convert mesh → O-Voxel realtime).

### Bước 3: Cấu Hình Rclone + Tải Dữ Liệu Từ Google Drive

```bash
# 3.1 — Cài rclone
curl https://rclone.org/install.sh | sudo bash

# 3.2 — Cấu hình rclone kết nối Drive
#   Chạy lệnh sau và làm theo hướng dẫn trên màn hình:
rclone config
#   - Chọn "n" (New remote)
#   - Tên: facediffgdrive
#   - Loại: drive (Google Drive)
#   - Scope: drive
#   - Làm theo hướng dẫn OAuth để xác thực
```

> [!IMPORTANT]
> Nếu máy cloud không có trình duyệt, hãy dùng cờ `--config` để copy file cấu hình
> từ máy cũ sang, hoặc dùng `rclone authorize "drive"` trên máy có trình duyệt rồi paste token.

```bash
# 3.3 — Tạo thư mục đích
mkdir -p ~/facediff/data
mkdir -p ~/facediff/checkpoints/sc_vae_shape
```

#### Phương án A: Tải file `.tar` về rồi giải nén (Đơn giản, cần ~560 GB trống)

```bash
# Tải ovoxel_cache.tar (272 GB — mất khoảng 3-5 giờ tùy mạng)
rclone copy -P "facediffgdrive:FaceDiff Data/ovoxel_cache.tar" ~/facediff/data/

# Giải nén (mất khoảng 20-30 phút)
cd ~/facediff
tar -xvf data/ovoxel_cache.tar -C data/

# ⚠️ XÓA FILE TAR NGAY LẬP TỨC ĐỂ GIẢI PHÓNG 272 GB
rm -f data/ovoxel_cache.tar
echo "✅ Đã xóa ovoxel_cache.tar, giải phóng 272 GB!"
```

#### Phương án B: Streaming Extraction (Tiết kiệm ổ cứng, chỉ cần ~300 GB trống)

```bash
# Giải nén trực tiếp từ Drive qua pipe — KHÔNG lưu file .tar trên ổ cứng
cd ~/facediff
rclone cat "facediffgdrive:FaceDiff Data/ovoxel_cache.tar" | tar -xvf - -C data/
```

> [!TIP]
> Phương án B an toàn hơn cho ổ 700 GB vì không bao giờ tồn tại 2 bản copy cùng lúc.

```bash
# 3.4 — Tải các file nhỏ còn lại (rất nhanh)
rclone copy -P "facediffgdrive:FaceDiff Data/hybrid_context_lmdb.tar" ~/facediff/data/
rclone copy -P "facediffgdrive:FaceDiff Data/mesh_manifest.json" ~/facediff/data/
rclone copy -P "facediffgdrive:FaceDiff Data/checkpoints/sc_vae_shape/latest_step.pt" \
    ~/facediff/checkpoints/sc_vae_shape/

# 3.5 — Giải nén hybrid_context
cd ~/facediff
tar -xvf data/hybrid_context_lmdb.tar -C data/
rm -f data/hybrid_context_lmdb.tar
```

### Bước 4: Xác Nhận Dữ Liệu Đã Đầy Đủ

```bash
cd ~/facediff

# Kiểm tra cấu trúc thư mục
echo "=== Kiểm tra dữ liệu ==="
ls -lh data/ovoxel_cache_lmdb/        # Phải tồn tại, ~272 GB
ls -lh data/hybrid_context.lmdb/      # Phải tồn tại, ~84 MB
ls -lh data/mesh_manifest.json        # Phải tồn tại, ~812 KB
ls -lh checkpoints/sc_vae_shape/latest_step.pt  # Phải tồn tại, ~403 MB

# Kiểm tra số lượng bản ghi LMDB
python -c "
import lmdb
env = lmdb.open('data/hybrid_context.lmdb', readonly=True)
print(f'Hybrid Context entries: {env.stat()[\"entries\"]}')  # Phải ≥ 20,939
env.close()
"

# Kiểm tra manifest
python -c "
import json
m = json.load(open('data/mesh_manifest.json'))
print(f'FaceVerse: {len(m[\"faceverse\"])} meshes')   # Phải = 2,310
print(f'FaceScape: {len(m[\"facescape\"])} meshes')   # Phải = 18,658
"

# Kiểm tra dung lượng ổ cứng còn lại
df -h .
```

> [!IMPORTANT]
> Nếu bất kỳ file nào thiếu, hãy quay lại Bước 3 để tải lại. Đặc biệt kiểm tra
> `ovoxel_cache_lmdb/` — nếu giải nén bằng Streaming bị đứt giữa chừng, phải chạy lại.

---

## 🏋️ Chạy Huấn Luyện

### Lựa chọn 1: Resume SC-VAE (Stage 1) — Tiếp tục từ Epoch 397

```bash
cd ~/facediff
conda activate facediff

# SC-VAE resume — RTX 3090 Ti config
python -u src/train_sc_vae.py \
    --dataset both \
    --feature-mode shape_mat \
    --in-channels 10 \
    --lmdb-only \
    --checkpoint-dir checkpoints/sc_vae_shape \
    --resume checkpoints/sc_vae_shape/latest_step.pt \
    --gradient-accumulation-steps 132 \
    --batch-size 1 \
    --resume-scheduler-mode cosine_restart \
    --resume-extend-epochs 100 \
    --resume-target-min-lr 1e-6 \
    --enable-stage2-render-loss \
    --no-torch-compile \
    --lr 1e-5
```

> [!NOTE]
> Cấu hình này giống y hệt đang chạy trên máy cũ (RTX 4090).
> RTX 3090 Ti có VRAM bằng (24 GB) nhưng tốc độ chậm hơn ~25%.
> Dự kiến mỗi epoch mất khoảng 2-3 giờ thay vì 1.5-2 giờ.

### Lựa chọn 2: Train VoxelMamba / iMF (Stage 2) — Chế độ Offline

```bash
cd ~/facediff
conda activate facediff

# VoxelMamba — chế độ offline, KHÔNG cần file .obj, KHÔNG nạp extractor lên GPU
python src/train_imf.py \
    --offline-data \
    --context-lmdb data/hybrid_context.lmdb \
    --manifest data/mesh_manifest.json \
    --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \
    --batch-size 32 \
    --num-workers 8 \
    --dataset both
```

> [!TIP]
> **Batch size gợi ý cho RTX 3090 Ti (24 GB VRAM)**:
> - Chế độ `--offline-data`: **32 → 64** (tùy kích thước slat_length)
> - Nếu bị OOM: Giảm xuống **16** hoặc bật gradient accumulation
> - Với `--offline-data`, VRAM tiết kiệm ~4-8 GB do không nạp ArcFace/FLAME/DINOv2

---

## 🔍 Khắc Phục Sự Cố Thường Gặp

### Lỗi "No module named 'o_voxel'"
```bash
# Kiểm tra nvcc có sẵn không
nvcc --version
# Nếu không có, cài CUDA toolkit:
sudo apt-get install -y cuda-nvcc-12-4
# Rồi chạy lại:
bash scripts/install_o_voxel.sh
```

### Lỗi "CUDA out of memory"
```bash
# Giảm batch-size
--batch-size 16   # hoặc 8
# Hoặc bật gradient accumulation
--gradient-accumulation-steps 4
```

### Lỗi "Disk Full" khi giải nén
```bash
# Kiểm tra dung lượng
df -h .
# Xóa file .tar nếu chưa xóa
rm -f data/ovoxel_cache.tar
# Hoặc dùng Streaming Extraction (Bước 3, Phương án B)
```

### Lỗi "rclone: directory not found"
```bash
# Liệt kê nội dung Drive để kiểm tra tên thư mục chính xác
rclone lsd facediffgdrive:
rclone ls "facediffgdrive:FaceDiff Data/"
```

### Lỗi "spconv" hoặc "mamba-ssm" cài không được
```bash
# Đảm bảo có build tools
sudo apt-get install -y build-essential cmake ninja-build
# Đảm bảo CUDA_HOME đúng
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
pip install --no-cache-dir spconv-cu121==2.3.8
pip install --no-cache-dir causal-conv1d==1.6.1
pip install --no-cache-dir mamba-ssm==2.3.1
```

---

## 📋 Checklist Nhanh

- [ ] Clone repo từ GitHub
- [ ] Tạo conda env `facediff` (Python 3.11)
- [ ] Cài PyTorch 2.5.1 + CUDA 12.1
- [ ] Cài `requirements.txt`
- [ ] Cài `o_voxel` (bash `scripts/install_o_voxel.sh`)
- [ ] Cài `requirements-mamba.txt` (causal-conv1d + mamba-ssm)
- [ ] Cấu hình rclone kết nối Google Drive
- [ ] Tải + giải nén `ovoxel_cache.tar` (272 GB)
- [ ] **Xóa** `ovoxel_cache.tar` sau giải nén
- [ ] Tải + giải nén `hybrid_context_lmdb.tar`
- [ ] Tải `mesh_manifest.json`
- [ ] Tải checkpoint `latest_step.pt`
- [ ] Chạy script xác nhận dữ liệu (Bước 4)
- [ ] Bắt đầu training!
