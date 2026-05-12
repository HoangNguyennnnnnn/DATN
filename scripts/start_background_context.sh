#!/bin/bash
# Script để chạy ngầm tiến trình trích xuất Hybrid Context (ArcFace, FLAME, DINOv2)
# sang LMDB và tự động đưa lên Google Drive khi hoàn thành.

set -e

WORKSPACE_DIR="/mnt/18TData/facediff"
PYTHON_BIN="$WORKSPACE_DIR/miniconda3/envs/facediff/bin/python"
OUTPUT_LMDB="$WORKSPACE_DIR/data/hybrid_context.lmdb"

REMOTE_NAME="facediffgdrive"
REMOTE_FOLDER="FaceDiff Data"

cd "$WORKSPACE_DIR"

echo "========================================================="
echo "Bắt đầu trích xuất Hybrid Context sang LMDB..."
echo "Thời gian: $(date)"
echo "Output: $OUTPUT_LMDB"
echo "========================================================="

# Thiết lập PYTHONPATH để import module src/...
export PYTHONPATH=.

# Bỏ qua torchao cho transformers
export FACEDIFF_DISABLE_TORCHAO=1

# Chạy script (sẽ tự động dùng CPU)
$PYTHON_BIN scripts/build_context_lmdb.py --out-lmdb "$OUTPUT_LMDB"

echo "========================================================="
echo "Trích xuất hoàn thành! Bắt đầu nén và upload lên Google Drive..."
echo "Thời gian: $(date)"
echo "========================================================="

# Lấy folder chứa LMDB (LMDB thực chất là một folder chứa data.mdb và lock.mdb)
# Nén lại thành tar để upload cho lẹ và tránh lỗi
TAR_FILE="$WORKSPACE_DIR/data/hybrid_context_lmdb.tar"
tar -cf "$TAR_FILE" -C "$WORKSPACE_DIR/data" "hybrid_context.lmdb"

echo "Nén xong: $(du -sh "$TAR_FILE")"

# Đồng bộ lên Drive (đường dẫn remote có khoảng trắng → một chuỗi remote:folder/)
DEST_REMOTE="${REMOTE_NAME}:${REMOTE_FOLDER}/"
if rclone copy -P "$TAR_FILE" "$DEST_REMOTE"; then
    echo "===================================================="
    echo "THÀNH CÔNG: Đã upload hybrid_context_lmdb.tar lên Google Drive!"
    echo "Lên Cloud, bạn có thể giải nén và dùng ngay cho VoxelMamba."
    echo "===================================================="
else
    echo "LỖI: Upload thất bại. Kiểm tra: rclone config ($REMOTE_NAME), mạng, quota Drive." >&2
    exit 1
fi
