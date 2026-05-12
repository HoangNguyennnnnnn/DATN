#!/usr/bin/env bash
#
# auto_upload_to_drive.sh — Đợi file tar hoàn tất rồi upload lên Google Drive (rclone).
#
# Vấn đề cũ: PID cứng (1464977) → sai máy hoặc PID không tồn tại thì vòng while bị bỏ qua,
# script upload ngay khi tar chưa xong → file hỏng trên Drive.
#
# Cách dùng:
#   # 1) Chờ kích thước file không đổi (mặc định, khuyến nghị cho tmux):
#   bash scripts/auto_upload_to_drive.sh
#
#   # 2) Chờ tiến trình tar kết thúc (truyền PID của lệnh tar):
#   tar -cf data/ovoxel_cache.tar ... & echo $! > /tmp/tar.pid
#   bash scripts/auto_upload_to_drive.sh --pid "$(cat /tmp/tar.pid)"
#
#   # 3) File đích khác:
#   TARGET_FILE=/path/to/file.tar bash scripts/auto_upload_to_drive.sh
#
# Biến môi trường:
#   WORKSPACE_DIR, TARGET_FILE, REMOTE_NAME, REMOTE_FOLDER,
#   STABLE_POLL_SEC (mặc định 30), STABLE_ROUNDS (mặc định 4 = ~2 phút không đổi)

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/18TData/facediff}"
TARGET_FILE="${TARGET_FILE:-$WORKSPACE_DIR/data/ovoxel_cache.tar}"
REMOTE_NAME="${REMOTE_NAME:-facediffgdrive}"
REMOTE_FOLDER="${REMOTE_FOLDER:-FaceDiff Data}"
STABLE_POLL_SEC="${STABLE_POLL_SEC:-30}"
STABLE_ROUNDS="${STABLE_ROUNDS:-4}"

WAIT_PID=""

usage() {
    echo "Usage: $0 [--pid PID] [--help]"
    echo "  --pid PID   Wait until this process exits (e.g. tar), then upload."
    echo "  (no args)   Wait until TARGET_FILE exists and size is stable (see STABLE_* env vars)."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)
            WAIT_PID="${2:?}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

DEST_REMOTE="${REMOTE_NAME}:${REMOTE_FOLDER}/"

wait_for_pid() {
    local pid="$1"
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        log "ERROR: --pid must be a numeric PID, got: $pid"
        exit 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        log "WARN: PID $pid is not running — không chờ được (có thể tar đã xong hoặc PID sai)."
        log "      Chuyển sang chờ file ổn định thay vì bỏ qua và upload sớm..."
        return 1
    fi
    log "Đang chờ tiến trình PID=$pid (tar/pack) kết thúc..."
    while kill -0 "$pid" 2>/dev/null; do
        if [[ -f "$TARGET_FILE" ]]; then
            log "  Still packing... size: $(du -sh "$TARGET_FILE" 2>/dev/null | cut -f1 || echo '?')"
        else
            log "  Waiting for $TARGET_FILE to appear..."
        fi
        sleep "$STABLE_POLL_SEC"
    done
    log "PID=$pid đã thoát."
    return 0
}

wait_until_file_stable() {
    log "Chế độ: chờ file tồn tại và kích thước không đổi trong $((STABLE_POLL_SEC * STABLE_ROUNDS)) giây (~$((STABLE_POLL_SEC * STABLE_ROUNDS))s)."
    local last_size=-1
    local stable_count=0
    while [[ $stable_count -lt $STABLE_ROUNDS ]]; do
        if [[ ! -f "$TARGET_FILE" ]]; then
            log "  Chưa có file: $TARGET_FILE — đợi ${STABLE_POLL_SEC}s..."
            sleep "$STABLE_POLL_SEC"
            stable_count=0
            last_size=-1
            continue
        fi
        local sz
        sz=$(stat -c '%s' "$TARGET_FILE" 2>/dev/null || echo 0)
        if [[ "$sz" == "$last_size" ]] && [[ "$sz" != "-1" ]] && [[ "$sz" != "0" ]]; then
            stable_count=$((stable_count + 1))
            log "  Size stable (${stable_count}/${STABLE_ROUNDS}): $(du -sh "$TARGET_FILE" | cut -f1) ($sz bytes)"
        else
            stable_count=0
            log "  Growing or changed: $(du -sh "$TARGET_FILE" 2>/dev/null | cut -f1 || echo '?') ($sz bytes)"
        fi
        last_size=$sz
        sleep "$STABLE_POLL_SEC"
    done
    log "File coi như hoàn tất (ổn định)."
}

echo "=========================================="
log "Auto upload → rclone"
log "TARGET_FILE=$TARGET_FILE"
log "REMOTE=$DEST_REMOTE"
echo "=========================================="

if [[ -n "$WAIT_PID" ]]; then
    if ! wait_for_pid "$WAIT_PID"; then
        wait_until_file_stable
    fi
else
    wait_until_file_stable
fi

if [[ ! -f "$TARGET_FILE" ]]; then
    log "ERROR: Không thấy file upload: $TARGET_FILE"
    exit 1
fi

log "Kích thước cuối: $(du -sh "$TARGET_FILE")"
log "Bắt đầu rclone copy..."

if rclone copy -P "$TARGET_FILE" "$DEST_REMOTE"; then
    echo "===================================================="
    log "SUCCESS: Đã upload lên Google Drive."
    log "Time: $(date)"
    echo "===================================================="
else
    echo "====================================================" >&2
    log "ERROR: rclone thất bại — kiểm tra mạng, quota Drive, và cấu hình remote '$REMOTE_NAME'."
    echo "====================================================" >&2
    exit 1
fi
