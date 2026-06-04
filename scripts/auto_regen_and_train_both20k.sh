#!/usr/bin/env bash
# ============================================================================
# AUTO PIPELINE (03/06/2026) — 1 tiến trình tự động làm TẤT CẢ:
#   regen slat 20K (both) → pack LMDB → stats → variance → train iMF tới ep8000
#
# SC-VAE: sc_vae_both/latest_step.pt (ep602, đã thấy cả 2 dataset).
# Context: LẤY SẴN từ data/hybrid_context.lmdb (không render lại — nhanh).
# Whiten: TÁI DÙNG context_whiten_v4.pt (giữ ctx_tokenizer 632-dim của model →
#         warm-resume conditioning còn nguyên giá trị; recompute sẽ đổi frame → hỏng).
# iMF: RESUME imf_v4/latest_step.pt (ep5084) → ep8000, sang dir mới imf_both20k.
#      Phase B config: JVP ratio 0.5 + adaptive=paper (chống nổ) + CFG.
# Chạy: nohup bash scripts/auto_regen_and_train_both20k.sh > logs/auto_both20k.log 2>&1 &
# ============================================================================
set -eo pipefail
cd "$(dirname "$0")/.."
source miniconda3/etc/profile.d/conda.sh
conda activate facediff
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=miniconda3/envs/facediff/bin/python
TS="$(date +%Y%m%d_%H%M%S)"

# ── Artifacts (đặt tên _both20k để tách hẳn run cũ) ──────────────────────────
SC_VAE=checkpoints/sc_vae_both/latest_step.pt
FV_CACHE=data/fv_slat_cache_v5
FS_CACHE=data/fs_slat_cache_v5
LMDB=data/slat_context_both20k.lmdb
STATS=data/slat_stats_both20k.pt
VAR=data/voxel_variance_both20k.pt
WHITEN=data/context_whiten_v4.pt          # TÁI DÙNG (632-dim, khớp model)
RESUME_FROM=checkpoints/imf_v4/latest_step.pt
CKPT_DIR=checkpoints/imf_both20k
TARGET_EPOCH=8000
NUM_SHARDS="${NUM_SHARDS:-6}"
EXPECTED=20968                            # tổng mesh (fv 2310 + fs 18658)

mkdir -p "$FV_CACHE" "$FS_CACHE" "$CKPT_DIR" logs logs/precompute_both20k

log(){ echo "[$(date +%H:%M:%S)] $*"; }
[ -f "$SC_VAE" ]      || { log "FATAL: thiếu $SC_VAE"; exit 1; }
[ -f "$WHITEN" ]      || { log "FATAL: thiếu $WHITEN"; exit 1; }
[ -f "$RESUME_FROM" ] || { log "FATAL: thiếu $RESUME_FROM"; exit 1; }

log "===== AUTO PIPELINE both-20K bắt đầu ====="
log "SC-VAE=$SC_VAE  shards=$NUM_SHARDS  target_ep=$TARGET_EPOCH"

# ── BƯỚC 1: precompute slat song song (N shards) ────────────────────────────
log "BƯỚC 1/6: precompute slat 20K — $NUM_SHARDS shards song song"
PIDS=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY scripts/data/precompute_slat_cache.py \
      --sc-vae-ckpt "$SC_VAE" \
      --dataset both \
      --fv-cache-dir "$FV_CACHE" \
      --fs-cache-dir "$FS_CACHE" \
      --context-lmdb data/hybrid_context.lmdb \
      --ovoxel-lmdb data/ovoxel_cache_lmdb \
      --manifest data/mesh_manifest.json \
      --shard-id $i --num-shards $NUM_SHARDS \
      --num-workers 0 --disable-id-filters --skip-existing \
      --device cuda:0 > "logs/precompute_both20k/shard_${i}.log" 2>&1 &
  PIDS+=($!)
  sleep 3
done
log "  Launched shards: ${PIDS[*]}"
for p in "${PIDS[@]}"; do wait "$p" || log "  WARN shard pid=$p exit non-zero (sẽ mop-up)"; done
log "  Shards xong."

# ── BƯỚC 2: mop-up 1 tiến trình (bắt mesh OOM/lỗi trong pass song song) ──────
log "BƯỚC 2/6: mop-up single-process (skip-existing) — đảm bảo đủ"
$PY scripts/data/precompute_slat_cache.py \
    --sc-vae-ckpt "$SC_VAE" --dataset both \
    --fv-cache-dir "$FV_CACHE" --fs-cache-dir "$FS_CACHE" \
    --context-lmdb data/hybrid_context.lmdb --ovoxel-lmdb data/ovoxel_cache_lmdb \
    --manifest data/mesh_manifest.json \
    --num-workers 0 --disable-id-filters --skip-existing \
    --device cuda:0 > "logs/precompute_both20k/mopup.log" 2>&1 || true

NPT=$(( $(find "$FV_CACHE" -name '*.pt' | wc -l) + $(find "$FS_CACHE" -name '*.pt' | wc -l) ))
log "  cache .pt total = $NPT / $EXPECTED"
if [ "$NPT" -lt $((EXPECTED * 95 / 100)) ]; then
  log "FATAL: cache < 95% ($NPT/$EXPECTED) — dừng để kiểm tra log shard."
  exit 1
fi

# ── BƯỚC 3: pack → LMDB ─────────────────────────────────────────────────────
log "BƯỚC 3/6: pack slat cache → $LMDB"
$PY scripts/data/pack_slat_lmdb.py \
    --fv-cache-dir "$FV_CACHE" --fs-cache-dir "$FS_CACHE" \
    --output "$LMDB" --map-size-gb 120
[ -f "$LMDB/data.mdb" ] || { log "FATAL: pack thất bại, thiếu $LMDB"; exit 1; }

# ── BƯỚC 4: slat stats (BẮT BUỘC mới — đã thêm facescape) ───────────────────
log "BƯỚC 4/6: compute slat stats → $STATS"
$PY scripts/data/compute_slat_stats.py --lmdb "$LMDB" --out "$STATS" --sc-vae-ckpt "$SC_VAE"
[ -f "$STATS" ] || { log "FATAL: thiếu $STATS"; exit 1; }

# ── BƯỚC 5: voxel variance (loss-weight, refresh theo data mới) ─────────────
log "BƯỚC 5/6: compute voxel variance → $VAR"
$PY scripts/data/compute_voxel_variance.py "$LMDB" "$STATS" "$VAR"
[ -f "$VAR" ] || { log "FATAL: thiếu $VAR"; exit 1; }

# ── BƯỚC 6: train iMF — RESUME ep5084 → ep8000 (Phase B: JVP + adaptive + CFG)
log "BƯỚC 6/6: train iMF resume $RESUME_FROM → ep$TARGET_EPOCH (dir=$CKPT_DIR)"
LOG_TRAIN="logs/imf_both20k_${TS}.log"
log "  train log: $LOG_TRAIN"
export VOXEL_VARIANCE_PATH="$VAR" VOXEL_VARIANCE_MULT=4.0
# "Cái để IoU": bật occupancy head BCE (occ_conv đã có trong ep5084) → model học mask
# voxel occupied/empty → gate output lúc inference → bớt floater/lỗ → IoU cao hơn.
# ep5084 đã train occ head; KHÔNG bật loss = occ head đứng yên, không cải thiện thêm.
export OCCUPANCY_LOSS_WEIGHT="${OCCUPANCY_LOSS_WEIGHT:-1.0}"
export PREDICTION_TYPE="${PREDICTION_TYPE:-velocity}"
log "  occupancy_loss_weight=$OCCUPANCY_LOSS_WEIGHT prediction_type=$PREDICTION_TYPE"
IMEFLOW_ADAPTIVE=paper IMEFLOW_ADAPTIVE_ON=1 \
$PY -u src/train_imf.py \
    --offline-data --dataset both \
    --slat-lmdb "$LMDB" --slat-stats "$STATS" \
    --context-lmdb data/hybrid_context.lmdb \
    --sc-vae-ckpt "$SC_VAE" \
    --manifest data/mesh_manifest.json \
    --resume "$RESUME_FROM" \
    --checkpoint-dir "$CKPT_DIR" \
    --backbone unet3d --unet-base 128 \
    --facescape-all-expressions \
    --context-use-all --context-whiten "$WHITEN" \
    --cfg-context-dropout 0.1 \
    --contrastive-loss-weight 0.0 --context-velocity-sep-weight 0.0 \
    --t-sampler logit_normal --ratio-r-neq-t 0.5 \
    --enable-cfg-conditioning --cfg-omega-max 8 --v-loss-weight 1.0 \
    --batch-size "${BATCH:-32}" --gradient-accumulation-steps "${ACCUM:-8}" \
    --num-workers 4 --prefetch-factor 4 --lr "${LR:-1e-4}" \
    --epochs "$TARGET_EPOCH" \
    > "$LOG_TRAIN" 2>&1
log "===== AUTO PIPELINE HOÀN TẤT (train tới ep$TARGET_EPOCH) ====="
