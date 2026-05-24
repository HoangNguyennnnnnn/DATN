#!/usr/bin/env bash
# Chờ Phase A hiện tại → eval → Phase B (CFG) → eval — không kill train đang chạy.
#   bash scripts/train_imf_v8_orchestrate.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/imf_v8_lite"
CHAIN_PHASE_B="${CHAIN_PHASE_B:-1}"
RUN_POST_A_EVAL="${RUN_POST_A_EVAL:-1}"
RUN_POST_B_EVAL="${RUN_POST_B_EVAL:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="logs/train_imf_v8_orchestrate_${TS}.log"
mkdir -p logs "${CKPT_DIR}"

if [ ! -f "${CKPT_DIR}/train.pid" ]; then
  echo "ERROR: Phase A not running and no ${CKPT_DIR}/train.pid"
  echo "  Start Phase A: bash scripts/train_imf_v8_lite.sh"
  exit 1
fi

PHASE_A_PID="$(cat "${CKPT_DIR}/train.pid")"
if ! kill -0 "${PHASE_A_PID}" 2>/dev/null; then
  echo "WARN: train.pid ${PHASE_A_PID} not alive — will only chain Phase B if latest_step exists"
fi

echo "=============================================="
echo "  iMF v8 orchestrator (wait A → eval → B → eval)"
echo "=============================================="
echo "  Phase A PID: ${PHASE_A_PID} (unchanged)"
echo "  CHAIN_PHASE_B=${CHAIN_PHASE_B}"
echo "  Log: ${ORCH_LOG}"
echo "=============================================="

export CHAIN_PHASE_B RUN_POST_A_EVAL RUN_POST_B_EVAL CKPT_DIR
nohup bash scripts/train_imf_v8_lite_pipeline.sh --wait-phase-a >> "${ORCH_LOG}" 2>&1 &
ORCH_PID=$!
echo "${ORCH_PID}" > "${CKPT_DIR}/orchestrate.pid"
echo "Orchestrator PID=${ORCH_PID}"
echo "  tail -f ${ORCH_LOG}"
