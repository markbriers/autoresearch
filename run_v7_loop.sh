#!/usr/bin/env bash
#
# run_v7_loop.sh — External bash wrapper for unattended v7 operation.
#
# Starts a fresh Claude Code session per cycle. Each session runs one
# complete Research -> Pre-Eval -> Implement -> Post-Eval -> Git snapshot
# cycle via the /run-v7 slash command.
#
# State lives entirely in files (hypotheses.md, evaluations.md, findings.md,
# results.tsv, research_log.md). Each cycle starts with zero context from
# prior cycles — the files ARE the memory.
#
# Usage:
#   chmod +x run_v7_loop.sh
#   nohup ./run_v7_loop.sh > orchestrator.log 2>&1 &
#   # or:
#   tmux new-session -d -s autoresearch './run_v7_loop.sh'
#
# To stop gracefully: touch STOP
# The loop checks for this file between cycles.

set -uo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$WORKDIR/orchestrator.log"
MAX_BACKOFF=300
BACKOFF=5
CYCLE=0

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG"
}

cd "$WORKDIR"

log "=== v7 loop started ==="
log "Working directory: $WORKDIR"
log "Stop file: $WORKDIR/STOP (touch to halt)"

while true; do
    if [ -f "$WORKDIR/STOP" ]; then
        log "STOP file detected. Halting loop."
        rm -f "$WORKDIR/STOP"
        exit 0
    fi

    CYCLE=$((CYCLE + 1))
    log "--- Cycle $CYCLE starting ---"

    EXIT_CODE=0
    claude --dangerously-skip-permissions \
        -p "/run-v7" \
        --output-format text \
        >> "$LOG" 2>&1 || EXIT_CODE=$?

    case $EXIT_CODE in
        0)
            log "Cycle $CYCLE completed successfully."
            BACKOFF=5
            ;;
        *)
            log "Cycle $CYCLE exited with code $EXIT_CODE."
            ;;
    esac

    # Check for rate limiting in recent output
    if tail -20 "$LOG" | grep -qi "529\|overloaded\|rate.limit\|too many requests"; then
        log "Rate limit detected. Backing off for ${BACKOFF}s."
        sleep "$BACKOFF"
        BACKOFF=$((BACKOFF * 2))
        if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then
            BACKOFF=$MAX_BACKOFF
        fi
    else
        sleep 10
    fi
done
