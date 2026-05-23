#!/usr/bin/env bash
# Phase A (lite) → chờ xong → Phase B (CFG) tự động.
#   bash scripts/train_imf_v8_lite_pipeline.sh
# Chỉ Phase A: CHAIN_PHASE_B=0 bash scripts/train_imf_v8_lite_pipeline.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/imf_v8_lite"
CHAIN_PHASE_B="${CHAIN_PHASE_B:-1}"
FRESH_START="${FRESH_START:-1}"

_wait_train_pid() {
  local pid="$1"
  local label="$2"
  echo "[pipeline] polling ${label} PID=${pid} (nohup — không dùng wait)"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
  done
  echo "[pipeline] ${label} process ended"
}

_run_pipeline() {
  echo "[pipeline] start $(date -Is) CHAIN_PHASE_B=${CHAIN_PHASE_B} FRESH_START=${FRESH_START}"

  pkill -f "train_imf.py.*imf_v8_lite" 2>/dev/null || true
  sleep 3

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  if [ "${FRESH_START}" = "1" ]; then
    if compgen -G "${CKPT_DIR}/*.pt" >/dev/null 2>&1; then
      local backup="checkpoints/imf_v8_lite_no_dropout_backup_${ts}"
      echo "[pipeline] backup ${CKPT_DIR} -> ${backup}"
      mv "${CKPT_DIR}" "${backup}"
      mkdir -p "${CKPT_DIR}"
    fi
  fi

  echo "[pipeline] === Phase A ==="
  export FRESH_START="${FRESH_START}"
  bash scripts/train_imf_v8_lite.sh
  sleep 5
  if [ ! -f "${CKPT_DIR}/train.pid" ]; then
    echo "[pipeline] ERROR: missing ${CKPT_DIR}/train.pid after Phase A launch"
    exit 1
  fi
  local phase_a_pid
  phase_a_pid="$(cat "${CKPT_DIR}/train.pid")"
  _wait_train_pid "${phase_a_pid}" "Phase A"
  if [ ! -f "${CKPT_DIR}/latest_step.pt" ]; then
    echo "[pipeline] ERROR: Phase A finished but no ${CKPT_DIR}/latest_step.pt"
    exit 1
  fi
  echo "[pipeline] Phase A done: ${CKPT_DIR}/latest_step.pt"

  if [ "${CHAIN_PHASE_B}" != "1" ]; then
    echo "[pipeline] CHAIN_PHASE_B=0 — skip Phase B"
    exit 0
  fi

  echo "[pipeline] === Phase B (CFG) ==="
  export CKPT_DIR
  export RESUME_CKPT="${CKPT_DIR}/latest_step.pt"
  bash scripts/train_imf_v8_phaseB_cfg.sh
  sleep 5
  if [ ! -f "${CKPT_DIR}/train.pid" ]; then
    echo "[pipeline] ERROR: missing train.pid after Phase B launch"
    exit 1
  fi
  local phase_b_pid
  phase_b_pid="$(cat "${CKPT_DIR}/train.pid")"
  _wait_train_pid "${phase_b_pid}" "Phase B"
  echo "[pipeline] all done $(date -Is)"
}

if [ "${1:-}" = "--worker" ]; then
  _run_pipeline
  exit 0
fi

# Chỉ chờ Phase A đang chạy → Phase B (không kill / không restart A)
if [ "${1:-}" = "--wait-phase-a" ]; then
  if [ ! -f "${CKPT_DIR}/train.pid" ]; then
    echo "ERROR: ${CKPT_DIR}/train.pid missing"
    exit 1
  fi
  phase_a_pid="$(cat "${CKPT_DIR}/train.pid")"
  _wait_train_pid "${phase_a_pid}" "Phase A"
  [ -f "${CKPT_DIR}/latest_step.pt" ] || { echo "ERROR: no latest_step.pt"; exit 1; }
  if [ "${CHAIN_PHASE_B}" = "1" ]; then
    export CKPT_DIR RESUME_CKPT="${CKPT_DIR}/latest_step.pt"
    bash scripts/train_imf_v8_phaseB_cfg.sh
    sleep 5
    phase_b_pid="$(cat "${CKPT_DIR}/train.pid")"
    _wait_train_pid "${phase_b_pid}" "Phase B"
  fi
  echo "[pipeline] all done $(date -Is)"
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="logs/train_imf_v8_lite_pipeline_${TS}.log"
mkdir -p "${CKPT_DIR}" logs

echo "=============================================="
echo "  iMF v8 LITE pipeline (A → B)"
echo "=============================================="
echo "  CHAIN_PHASE_B=${CHAIN_PHASE_B}  FRESH_START=${FRESH_START}"
echo "  Log: ${PIPELINE_LOG}"
echo "=============================================="

export CHAIN_PHASE_B FRESH_START CKPT_DIR
nohup bash "$0" --worker >> "${PIPELINE_LOG}" 2>&1 &
PIPE_PID=$!
echo "${PIPE_PID}" > "${CKPT_DIR}/pipeline.pid"
echo "Pipeline PID=${PIPE_PID}  log=${PIPELINE_LOG}"
echo "  tail -f ${PIPELINE_LOG}"
