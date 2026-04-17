#!/bin/bash
# Crash-safe wrapper for overnight_chain.py
#
# Features:
#   - Auto-relaunches on crash (up to MAX_RETRIES times)
#   - 30s cooldown between retries (let GPU/RAM settle)
#   - Logs all output to overnight_chain.log
#   - Works with nohup for detached execution
#
# Usage (foreground):
#   bash scripts/restart_wrapper.sh
#
# Usage (detached — survives logout/lid-close):
#   nohup bash scripts/restart_wrapper.sh &
#   tail -f overnight_chain.log

set -uo pipefail

cd "/home/lucian/Codex/Swim -Fractal"
source .venv/bin/activate

MAX_RETRIES=10
RETRY_DELAY=30
LOG_FILE="overnight_chain.log"

echo "[$(date)] ═══ RESTART WRAPPER STARTED ═══" | tee -a "$LOG_FILE"

retry=0
while [ $retry -lt $MAX_RETRIES ]; do
    echo "[$(date)] Starting overnight_chain.py (attempt $((retry + 1))/$MAX_RETRIES)" | tee -a "$LOG_FILE"
    
    python -u scripts/overnight_chain.py 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] ═══ CHAIN COMPLETED SUCCESSFULLY ═══" | tee -a "$LOG_FILE"
        exit 0
    fi
    
    retry=$((retry + 1))
    echo "[$(date)] ⚠️  Crashed with exit code $EXIT_CODE (attempt $retry/$MAX_RETRIES)" | tee -a "$LOG_FILE"
    
    if [ $retry -lt $MAX_RETRIES ]; then
        echo "[$(date)] Waiting ${RETRY_DELAY}s before retry..." | tee -a "$LOG_FILE"
        sleep $RETRY_DELAY
    fi
done

echo "[$(date)] ═══ MAX RETRIES ($MAX_RETRIES) EXHAUSTED ═══" | tee -a "$LOG_FILE"
exit 1
