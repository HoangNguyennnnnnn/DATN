#!/bin/bash
# ============================================================
# FaceDiff — Script thiết lập TẤT CẢ trên Cloud GPU
# ============================================================
# Chỉ cần chạy DUY NHẤT 1 lệnh:
#   bash scripts/cloud_full_setup.sh
#
# Script sẽ tự động:
#   1. Cài môi trường Conda + PyTorch + Dependencies
#   2. Cấu hình rclone từ token
#   3. Tải + giải nén dữ liệu từ Google Drive (Streaming — không cần ổ cứng lớn)
#   4. Xác nhận dữ liệu đầy đủ
#   5. In lệnh train sẵn sàng copy-paste
#
# Yêu cầu máy: RTX 3090 Ti (24GB VRAM), 64GB RAM, ≥500GB SSD
# ============================================================

set -e  # Dừng ngay nếu có lỗi

# ---- Màu sắc cho terminal ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✅ OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[⚠️  WARN]${NC} $1"; }
err()  { echo -e "${RED}[❌ ERROR]${NC} $1"; }
info() { echo -e "${CYAN}[📌 INFO]${NC} $1"; }

echo ""
echo "============================================================"
echo "  🚀 FaceDiff — Cloud GPU Full Setup"
echo "  Cấu hình: RTX 3090 Ti · 24GB VRAM · 64GB RAM"
echo "============================================================"
echo ""

# ---- Xác định thư mục gốc ----
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
info "Thư mục repo: $REPO_DIR"

# ============================================================
# BƯỚC 1: Cài đặt môi trường Python
# ============================================================
echo ""
echo "============================================================"
echo "  [1/5] Cài đặt môi trường Python"
echo "============================================================"

# Kiểm tra conda
if ! command -v conda &>/dev/null; then
    warn "Conda chưa được cài. Đang cài Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    log "Miniconda đã cài xong"
fi

# Tạo/kích hoạt env
if conda env list | grep -q "facediff"; then
    info "Conda env 'facediff' đã tồn tại, kích hoạt..."
    eval "$(conda shell.bash hook)"
    conda activate facediff
else
    info "Tạo conda env 'facediff' (Python 3.11)..."
    conda create -n facediff python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate facediff
    log "Conda env 'facediff' đã tạo xong"
fi

# Cài PyTorch
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "PyTorch + CUDA đã sẵn sàng: $(python -c 'import torch; print(torch.__version__)')"
else
    info "Cài PyTorch 2.5.1 + CUDA 12.1..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    log "PyTorch đã cài xong"
fi

# Cài dependencies
info "Cài dependencies từ requirements.txt..."
pip install -r requirements.txt 2>&1 | tail -n 5
log "Dependencies cơ bản đã cài xong"

# Cài build tools
info "Cài build tools (cmake, ninja)..."
sudo apt-get update -qq && sudo apt-get install -y -qq git build-essential cmake ninja-build 2>/dev/null || \
    warn "Không thể cài build tools qua apt (có thể cần sudo). Bỏ qua."

# Cài o_voxel (tùy chọn — có thể fail nếu không có nvcc)
info "Cài o_voxel (Microsoft TRELLIS.2)..."
if bash scripts/install_o_voxel.sh 2>&1 | tail -n 3; then
    log "o_voxel đã cài xong"
else
    warn "o_voxel cài thất bại (có thể thiếu nvcc). SC-VAE --lmdb-only vẫn chạy được."
fi

# Cài Mamba SSM (cho VoxelMamba Stage 2)
info "Cài Mamba SSM (causal-conv1d + mamba-ssm)..."
if pip install -r requirements-mamba.txt 2>&1 | tail -n 3; then
    log "Mamba SSM đã cài xong"
else
    warn "Mamba SSM cài thất bại. VoxelMamba sẽ dùng GRU fallback."
fi

# ============================================================
# BƯỚC 2: Cấu hình Rclone
# ============================================================
echo ""
echo "============================================================"
echo "  [2/5] Cấu hình Rclone kết nối Google Drive"
echo "============================================================"

# Cài rclone nếu chưa có
if ! command -v rclone &>/dev/null; then
    info "Cài rclone..."
    curl -s https://rclone.org/install.sh | sudo bash
    log "Rclone đã cài xong"
else
    log "Rclone đã có: $(rclone version | head -1)"
fi

# Kiểm tra cấu hình rclone
if rclone listremotes 2>/dev/null | grep -q "facediffgdrive"; then
    log "Rclone remote 'facediffgdrive' đã được cấu hình"
else
    warn "Rclone chưa được cấu hình cho 'facediffgdrive'."
    echo ""
    echo "  Bạn cần chạy: rclone config"
    echo "  - Chọn 'n' (New remote)"
    echo "  - Tên: facediffgdrive"
    echo "  - Loại: drive (Google Drive)"
    echo "  - Scope: drive"
    echo "  - Làm theo hướng dẫn OAuth"
    echo ""
    echo "  HOẶC copy file .config/rclone/rclone.conf từ máy cũ sang:"
    echo "    mkdir -p ~/.config/rclone"
    echo "    # Paste nội dung rclone.conf vào ~/.config/rclone/rclone.conf"
    echo ""
    read -p "  Nhấn Enter sau khi đã cấu hình xong rclone..." _
fi

# ============================================================
# BƯỚC 3: Tải + Giải nén dữ liệu từ Google Drive
# ============================================================
echo ""
echo "============================================================"
echo "  [3/5] Tải dữ liệu từ Google Drive"
echo "============================================================"

mkdir -p data
mkdir -p checkpoints/sc_vae_shape

# --- 3a: O-Voxel Cache (272 GB) — Streaming Extraction ---
if [ -d "data/ovoxel_cache_lmdb" ] && [ "$(ls data/ovoxel_cache_lmdb/ 2>/dev/null | wc -l)" -gt 0 ]; then
    log "data/ovoxel_cache_lmdb/ đã tồn tại, bỏ qua tải."
else
    info "Tải + giải nén ovoxel_cache.tar (272 GB) trực tiếp từ Drive (Streaming)..."
    info "Phương pháp này KHÔNG lưu file .tar trên ổ cứng — tiết kiệm 272 GB!"
    info "Thời gian ước tính: 3-8 giờ tùy tốc độ mạng."
    echo ""
    rclone cat "facediffgdrive:FaceDiff Data/ovoxel_cache.tar" | tar -xf - -C data/
    log "ovoxel_cache_lmdb đã giải nén xong!"
fi

# --- 3b: Hybrid Context (87 MB) ---
if [ -d "data/hybrid_context.lmdb" ]; then
    log "data/hybrid_context.lmdb/ đã tồn tại, bỏ qua."
else
    info "Tải hybrid_context_lmdb.tar (87 MB)..."
    rclone copy -P "facediffgdrive:FaceDiff Data/hybrid_context_lmdb.tar" data/
    tar -xf data/hybrid_context_lmdb.tar -C data/
    rm -f data/hybrid_context_lmdb.tar
    log "hybrid_context.lmdb đã sẵn sàng!"
fi

# --- 3c: Mesh Manifest ---
if [ -f "data/mesh_manifest.json" ]; then
    log "data/mesh_manifest.json đã tồn tại."
else
    info "Tải mesh_manifest.json..."
    rclone copy -P "facediffgdrive:FaceDiff Data/mesh_manifest.json" data/
    log "mesh_manifest.json đã tải xong!"
fi

# --- 3d: SC-VAE Checkpoint ---
if [ -f "checkpoints/sc_vae_shape/latest_step.pt" ]; then
    log "Checkpoint SC-VAE đã tồn tại."
else
    info "Tải checkpoint SC-VAE (latest_step.pt ~403 MB)..."
    rclone copy -P "facediffgdrive:FaceDiff Data/checkpoints/sc_vae_shape/latest_step.pt" \
        checkpoints/sc_vae_shape/
    log "Checkpoint SC-VAE đã tải xong!"
fi

# ============================================================
# BƯỚC 4: Xác nhận dữ liệu đầy đủ
# ============================================================
echo ""
echo "============================================================"
echo "  [4/5] Xác nhận dữ liệu"
echo "============================================================"

ERRORS=0

# Kiểm tra ovoxel_cache_lmdb
if [ -d "data/ovoxel_cache_lmdb" ] && [ "$(ls data/ovoxel_cache_lmdb/ 2>/dev/null | wc -l)" -gt 0 ]; then
    SIZE=$(du -sh data/ovoxel_cache_lmdb/ | cut -f1)
    log "ovoxel_cache_lmdb: $SIZE"
else
    err "data/ovoxel_cache_lmdb/ KHÔNG TỒN TẠI hoặc trống!"
    ERRORS=$((ERRORS+1))
fi

# Kiểm tra hybrid_context.lmdb
if [ -d "data/hybrid_context.lmdb" ]; then
    log "hybrid_context.lmdb: $(du -sh data/hybrid_context.lmdb/ | cut -f1)"
else
    err "data/hybrid_context.lmdb/ KHÔNG TỒN TẠI!"
    ERRORS=$((ERRORS+1))
fi

# Kiểm tra manifest
if [ -f "data/mesh_manifest.json" ]; then
    log "mesh_manifest.json: $(du -sh data/mesh_manifest.json | cut -f1)"
else
    err "data/mesh_manifest.json KHÔNG TỒN TẠI!"
    ERRORS=$((ERRORS+1))
fi

# Kiểm tra checkpoint
if [ -f "checkpoints/sc_vae_shape/latest_step.pt" ]; then
    log "SC-VAE checkpoint: $(du -sh checkpoints/sc_vae_shape/latest_step.pt | cut -f1)"
else
    err "checkpoints/sc_vae_shape/latest_step.pt KHÔNG TỒN TẠI!"
    ERRORS=$((ERRORS+1))
fi

# Kiểm tra dung lượng ổ cứng
AVAIL=$(df -h . | tail -1 | awk '{print $4}')
info "Dung lượng ổ cứng còn trống: $AVAIL"

if [ $ERRORS -gt 0 ]; then
    err "Có $ERRORS lỗi! Vui lòng kiểm tra lại."
    exit 1
fi

# ============================================================
# BƯỚC 5: In lệnh train sẵn sàng
# ============================================================
echo ""
echo "============================================================"
echo "  [5/5] ✅ SETUP HOÀN TẤT! Các lệnh train sẵn sàng"
echo "============================================================"
echo ""
echo "  📌 GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  📌 VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  📌 Disk Free: $AVAIL"
echo ""
echo "  ─────────────────────────────────────────────────────────"
echo "  OPTION 1: Resume SC-VAE (Stage 1) — Tiếp tục từ Epoch 397"
echo "  ─────────────────────────────────────────────────────────"
echo ""
echo "  conda activate facediff"
echo "  cd $REPO_DIR"
echo "  python -u src/train_sc_vae.py \\"
echo "      --dataset both \\"
echo "      --feature-mode shape_mat \\"
echo "      --in-channels 10 \\"
echo "      --lmdb-only \\"
echo "      --checkpoint-dir checkpoints/sc_vae_shape \\"
echo "      --resume checkpoints/sc_vae_shape/latest_step.pt \\"
echo "      --gradient-accumulation-steps 132 \\"
echo "      --batch-size 1 \\"
echo "      --resume-scheduler-mode cosine_restart \\"
echo "      --resume-extend-epochs 100 \\"
echo "      --resume-target-min-lr 1e-6 \\"
echo "      --enable-stage2-render-loss \\"
echo "      --no-torch-compile \\"
echo "      --lr 1e-5"
echo ""
echo "  ─────────────────────────────────────────────────────────"
echo "  OPTION 2: Train VoxelMamba (Stage 2) — Chế độ Offline"
echo "  ─────────────────────────────────────────────────────────"
echo ""
echo "  conda activate facediff"
echo "  cd $REPO_DIR"
echo "  python src/train_imf.py \\"
echo "      --offline-data \\"
echo "      --context-lmdb data/hybrid_context.lmdb \\"
echo "      --manifest data/mesh_manifest.json \\"
echo "      --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \\"
echo "      --batch-size 32 \\"
echo "      --num-workers 8 \\"
echo "      --dataset both"
echo ""
echo "============================================================"
echo "  🎉 Chúc bạn train thành công trên Cloud GPU!"
echo "============================================================"
