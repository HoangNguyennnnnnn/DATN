#!/usr/bin/env bash
# Nén hybrid_context.lmdb → .tar và đẩy lên Google Drive.
# CHẠY SAU KHI build_context_lmdb.py ĐÃ XONG (không nên tar trong lúc LMDB đang ghi).
#
# Usage: bash scripts/tar_and_upload_hybrid_context.sh

set -euo pipefail
ROOT="${ROOT:-/mnt/18TData/facediff}"
cd "$ROOT"

REMOTE_NAME="${REMOTE_NAME:-facediffgdrive}"
REMOTE_FOLDER="${REMOTE_FOLDER:-FaceDiff Data}"
DEST_REMOTE="${REMOTE_NAME}:${REMOTE_FOLDER}/"
LMDB_DIR="$ROOT/data/hybrid_context.lmdb"
TAR_FILE="$ROOT/data/hybrid_context_lmdb.tar"

if pgrep -f 'build_context_lmdb\.py' >/dev/null 2>&1; then
  echo "CẢNH BÁO: build_context_lmdb.py vẫn đang chạy — tar trong lúc ghi LMDB có thể không nhất quán."
  if [[ "${SKIP_CONFIRM:-0}" == "1" ]]; then
    echo "SKIP_CONFIRM=1 → tiếp tục (bạn chịu rủi ro)."
  elif [[ -t 0 ]]; then
    read -r -p "Tiếp tục? [y/N] " a || true
    [[ "${a:-}" =~ ^[yY]$ ]] || { echo "Đã hủy."; exit 1; }
  else
    echo "Không có terminal tương tác: thoát. Chờ build xong hoặc dùng SKIP_CONFIRM=1 (rủi ro)."
    exit 1
  fi
fi

if [[ ! -d "$LMDB_DIR" ]]; then
  echo "Không thấy $LMDB_DIR"
  exit 1
fi

echo "[1/2] Đang nén $LMDB_DIR → $TAR_FILE"
tar -cf "$TAR_FILE" -C "$ROOT/data" "hybrid_context.lmdb"
echo "Xong: $(du -sh "$TAR_FILE")"

echo "[2/2] rclone → $DEST_REMOTE"
rclone copy -P \
  --retries 10 \
  --low-level-retries 20 \
  "$TAR_FILE" "$DEST_REMOTE"

echo "Hoàn tất upload hybrid_context_lmdb.tar"
