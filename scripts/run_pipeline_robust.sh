#!/bin/bash
# ============================================================
# Robust pipeline launcher: nohup + auto-restart on crash
# Usage: bash scripts/run_pipeline_robust.sh
# Logs:  logs/pipeline_wrapper.log
# PID:   logs/pipeline.pid
# ============================================================
cd /mnt/18TData/facediff

PIPELINE_SCRIPT="scripts/resume_from_step4.sh"
WRAPPER_LOG="logs/pipeline_wrapper.log"
PID_FILE="logs/pipeline.pid"
MAX_RESTARTS=10
RESTART_DELAY=30

mkdir -p logs

# Kill old pipeline if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date)] Killing old pipeline PID=$OLD_PID" | tee -a "$WRAPPER_LOG"
        kill "$OLD_PID" 2>/dev/null
        sleep 5
    fi
    rm -f "$PID_FILE"
fi

# Disable systemd-oomd for this process tree
echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

echo "[$(date)] Pipeline wrapper started (PID=$$)" | tee -a "$WRAPPER_LOG"
echo $$ > "$PID_FILE"

restart_count=0
while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "[$(date)] Starting pipeline (attempt $((restart_count+1))/$MAX_RESTARTS)" | tee -a "$WRAPPER_LOG"

    # Run pipeline script — inherits OOM protection
    bash "$PIPELINE_SCRIPT" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Pipeline completed successfully!" | tee -a "$WRAPPER_LOG"
        rm -f "$PID_FILE"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    echo "[$(date)] Pipeline crashed (exit=$EXIT_CODE). Restart $restart_count/$MAX_RESTARTS in ${RESTART_DELAY}s..." | tee -a "$WRAPPER_LOG"
    sleep "$RESTART_DELAY"
done

echo "[$(date)] Max restarts ($MAX_RESTARTS) reached. Giving up." | tee -a "$WRAPPER_LOG"
rm -f "$PID_FILE"
exit 1
