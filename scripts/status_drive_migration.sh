#!/usr/bin/env bash
# Hiển thị nhanh: upload ovoxel → Drive, tiến độ hybrid_context LMDB, tmux.
set -euo pipefail
ROOT="${ROOT:-/mnt/18TData/facediff}"
export ROOT
PY="${ROOT}/miniconda3/envs/facediff/bin/python"

echo "=== tmux sessions ==="
tmux list-sessions 2>/dev/null || echo "(không có tmux)"

echo ""
echo "=== tiến trình liên quan (rclone ovoxel / build_context) ==="
pgrep -af 'rclone copy.*ovoxel_cache\.tar' 2>/dev/null || echo "(không thấy rclone ovoxel)"
pgrep -af 'build_context_lmdb\.py' 2>/dev/null || echo "(không thấy build_context)"

echo ""
echo "=== file local ==="
ls -lah "${ROOT}/data/ovoxel_cache.tar" 2>/dev/null || true
du -sh "${ROOT}/data/hybrid_context.lmdb" 2>/dev/null || true

echo ""
echo "=== hybrid_context.lmdb (entries vs manifest) ==="
"$PY" <<PY
import json, os, lmdb
root = os.environ.get("ROOT", "/mnt/18TData/facediff")
mdb = os.path.join(root, "data/hybrid_context.lmdb")
man = os.path.join(root, "data/mesh_manifest.json")
if os.path.isdir(mdb):
    env = lmdb.open(mdb, readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
    env.close()
else:
    n = 0
total = 0
if os.path.isfile(man):
    with open(man) as f:
        m = json.load(f)
    total = len(m.get("faceverse", [])) + len(m.get("facescape", []))
print(f"  LMDB entries: {n}")
print(f"  mesh_manifest total: {total}")
if total:
    print(f"  tiến độ ~{100.0 * n / total:.1f}%")
PY

echo ""
echo "Gợi ý:"
echo "  - ovoxel_cache.tar: đừng chạy thêm rclone trùng — chỉ một tiến trình upload 272GB."
echo "  - hybrid_context: chờ build_context_lmdb.py xong rồi chạy:"
echo "      bash ${ROOT}/scripts/tar_and_upload_hybrid_context.sh"
