#!/usr/bin/env bash
# Eval gen-self định kỳ trên checkpoint mới nhất → ghi trend vào logs/eval_trend_both20k.log
# Gate: dừng khi diag-off đạt mục tiêu (mặc định 0.35 ~ mức v4). Chạy nền song song train.
set -o pipefail
cd "$(dirname "$0")/.."
source miniconda3/etc/profile.d/conda.sh
conda activate facediff
PY=miniconda3/envs/facediff/bin/python
LOG=logs/eval_trend_both20k.log
CKPT=checkpoints/imf_both20k/latest_step.pt
LMDB=data/slat_context_both20k.lmdb
STATS=data/slat_stats_both20k.pt
INTERVAL="${INTERVAL:-1800}"   # 30 phút
TARGET="${TARGET:-0.35}"        # diag-off mục tiêu

echo "[$(date +%H:%M:%S)] eval-trend bắt đầu (mỗi ${INTERVAL}s, target diag-off=$TARGET) → $LOG" | tee -a "$LOG"
while true; do
  if [ -f "$CKPT" ]; then
    # đo CẢ 2 dataset: FaceVerse (skip 0) + FaceScape (skip 2300)
    geteval(){ $PY scripts/test/gen_scale_indep_noise.py --ckpt "$CKPT" --lmdb "$LMDB" --stats "$STATS" \
          --n 8 --skip "$1" --stride 25 --steps 8 --omega 2 --prediction-type velocity 2>/dev/null \
          | grep -aE "epoch=|gen-self diag|diag - off"; }
    FV=$(geteval 0); FS=$(geteval 2300)
    EP=$(echo "$FV" | grep -aoE "epoch=[0-9]+" | head -1)
    FV_D=$(echo "$FV" | grep -aoE "diag - off    = [0-9.]+" | grep -aoE "[0-9.]+$")
    FS_D=$(echo "$FS" | grep -aoE "diag - off    = [0-9.]+" | grep -aoE "[0-9.]+$")
    echo "[$(date +%m-%d_%H:%M)] $EP FaceVerse_diagoff=$FV_D FaceScape_diagoff=$FS_D" | tee -a "$LOG"
    # gate: báo khi EITHER dataset đạt target
    if { [ -n "$FV_D" ] && awk "BEGIN{exit !($FV_D>=$TARGET)}"; } || { [ -n "$FS_D" ] && awk "BEGIN{exit !($FS_D>=$TARGET)}"; }; then
      echo "[$(date +%m-%d_%H:%M)] 🎯 ĐẠT GATE (FV=$FV_D FS=$FS_D >= $TARGET) — cân nhắc dừng + test ảnh thật" | tee -a "$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
