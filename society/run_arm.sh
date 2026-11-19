#!/usr/bin/env bash
# Run one arm until it has used its training-run budget.
#
#   cd ../arm-society && RUN_BUDGET=40 ../autoresearch/society/run_arm.sh
#
# The budget counts every training run: experiments in results.tsv (excluding
# the starting baseline) plus every seed run logged by the audit. Both arms get
# the same budget, so the comparison is at equal compute.
set -uo pipefail
RUN_BUDGET="${RUN_BUDGET:-40}"
LOG=orchestrator.log
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG"; }

runs_used() {
  # Every row after the header in results.tsv is a training run (experiments and
  # champion reference runs); every "run" line in audits.jsonl is a seed run.
  local e=0 a=0
  [ -f results.tsv ] && e=$(tail -n +2 results.tsv | grep -c . || true)
  [ -f audits.jsonl ] && a=$(grep -c '"kind": "run"' audits.jsonl || true)
  echo $((e + a))
}

source arm.env
log "=== arm $ARM started, budget $RUN_BUDGET runs ==="
CYCLE=0
while true; do
  [ -f STOP ] && { log "STOP file found"; break; }
  USED=$(runs_used)
  if [ "$USED" -ge "$RUN_BUDGET" ]; then log "budget reached ($USED/$RUN_BUDGET)"; break; fi
  CYCLE=$((CYCLE + 1))
  log "cycle $CYCLE starting ($USED/$RUN_BUDGET runs used)"
  claude -p "/run-society" --dangerously-skip-permissions >> "$LOG" 2>&1 || { log "cycle $CYCLE exited non-zero; backing off 30s"; sleep 30; }
done
log "=== arm $ARM finished ==="
