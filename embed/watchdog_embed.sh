#!/usr/bin/env bash
# Watchdog for batch_embed.py — monitors a running PID, then retries on failure.
# Checkpoints live in Supabase, so restarts are safe: already-embedded chunks are skipped.
#
# Usage: ./embed/watchdog_embed.sh <PID> [max_retries]
#   PID          — PID of the currently running batch_embed.py (required)
#   max_retries  — how many restart attempts before giving up (default: 10)

set -uo pipefail

WATCH_PID="${1:?Usage: watchdog_embed.sh <PID> [max_retries]}"
MAX_RETRIES="${2:-10}"
LOG="data_files/watchdog_embed.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

cd /Users/raphaelchen/Desktop/legal_rag

mkdir -p data_files
log "=== Watchdog started (watching PID $WATCH_PID, max retries $MAX_RETRIES) ==="

# Wait for the VS Code terminal process to exit (we're not its parent, so use poll)
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 15
done
log "PID $WATCH_PID has exited. Entering retry loop."

n=0
while [ "$n" -lt "$MAX_RETRIES" ]; do
    n=$((n + 1))
    log "--- Attempt $n / $MAX_RETRIES ---"

    if python3 embed/batch_embed.py >> "$LOG" 2>&1; then
        log "SUCCESS: batch_embed.py completed on attempt $n."
        exit 0
    fi

    if [ "$n" -ge "$MAX_RETRIES" ]; then
        log "FAILED: exhausted $MAX_RETRIES retries. Giving up."
        exit 1
    fi

    # Exponential backoff capped at 5 minutes
    backoff=$(( 30 * (2 ** (n - 1)) ))
    [ "$backoff" -gt 300 ] && backoff=300
    log "Attempt $n failed. Retrying in ${backoff}s..."
    sleep "$backoff"
done
