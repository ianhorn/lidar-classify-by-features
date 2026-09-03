#!/usr/bin/env bash
# Runs the Coiled pipeline repeatedly until stac_item_list.parquet is empty.
# Each iteration: refresh+upload the todo list, launch main.py, wait for it
# to exit (either the 12h cluster_timeout, a clean finish, or an early kill
# if the known scheduler-connection-lost bug recurs), then loop.
#
# Meant to be launched detached (nohup ... & disown) so it outlives the
# Claude Code session that started it -- it's a plain OS process, not tied
# to Claude in any way.
#
# Hard-stops (instead of looping) on:
#   - a known fatal error signature (Coiled billing/compute quota exceeded,
#     auth failure) in the driver's log
#   - two iterations in a row where the driver dies within 90s of launch --
#     cluster creation alone takes 1-2 min, so anything faster than that is
#     a crash before real work started, not a real 12h run ending early.
#     A single fast failure is tolerated (could be a one-off blip); two in a
#     row means something is fundamentally broken and retrying a third time
#     would just burn API calls for nothing.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LOGDIR="$REPO_ROOT/run-logs/continuous_run"
mkdir -p "$LOGDIR"
ITER=0
CONSECUTIVE_FAST_FAILS=0
FAST_FAIL_THRESHOLD_S=90

while true; do
  ITER=$((ITER+1))
  echo "=== [$(date -Is)] iteration $ITER: refreshing + uploading todo list ==="
  REFRESH_LOG="$LOGDIR/refresh_${ITER}.log"
  python3 scripts/refresh_todo_list.py > "$REFRESH_LOG" 2>&1
  cat "$REFRESH_LOG"
  REMAINING=$(grep -oP 'REMAINING_COUNT=\K\d+' "$REFRESH_LOG")

  if [ -z "$REMAINING" ]; then
    echo "=== [$(date -Is)] iteration $ITER: could not parse remaining count, refresh_todo_list.py likely failed -- stopping orchestrator ==="
    exit 1
  fi

  if [ "$REMAINING" -eq 0 ]; then
    echo "=== [$(date -Is)] ALL TILES PROCESSED (0 remaining) -- continuous run complete after $ITER iteration(s) ==="
    exit 0
  fi

  echo "=== [$(date -Is)] iteration $ITER: launching main.py against $REMAINING remaining tiles ==="
  MAIN_LOG="$LOGDIR/main_${ITER}.log"
  python3 src/main.py > "$MAIN_LOG" 2>&1 &
  PID=$!
  echo "driver pid $PID, log $MAIN_LOG"

  START=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - START ))
    # Only treat this as the early reconnect-bug failure signature inside the
    # first 30 minutes -- past that (near the 12h cluster_timeout) the same
    # message is the expected/benign end-of-run cancellation flood.
    if [ "$ELAPSED" -lt 1800 ] && grep -qE "scheduler-connection-lost|CommClosedError|dictionary changed size during iteration" "$MAIN_LOG"; then
      echo "=== [$(date -Is)] iteration $ITER: early scheduler-connection-lost signature at ${ELAPSED}s -- killing driver, moving to next iteration ==="
      kill -9 "$PID" 2>/dev/null
      break
    fi
    sleep 5
  done
  wait "$PID" 2>/dev/null
  EXIT_CODE=$?
  DURATION=$(( $(date +%s) - START ))
  echo "=== [$(date -Is)] iteration $ITER: driver finished (pid $PID, exit $EXIT_CODE, ${DURATION}s) ==="

  # Known fatal errors -- no point retrying, the next iteration will just
  # hit the exact same wall immediately.
  if grep -qiE "total compute quota|ServerError|AuthenticationError|Unauthorized|InvalidAccessKeyId" "$MAIN_LOG"; then
    echo "=== [$(date -Is)] FATAL: known unrecoverable error detected in $MAIN_LOG -- stopping orchestrator, not retrying ==="
    grep -iE "total compute quota|ServerError|AuthenticationError|Unauthorized|InvalidAccessKeyId" "$MAIN_LOG" | head -5
    exit 2
  fi

  if [ "$DURATION" -lt "$FAST_FAIL_THRESHOLD_S" ]; then
    CONSECUTIVE_FAST_FAILS=$((CONSECUTIVE_FAST_FAILS+1))
    echo "=== [$(date -Is)] iteration $ITER: driver exited after only ${DURATION}s (< ${FAST_FAIL_THRESHOLD_S}s, cluster creation alone takes longer) -- consecutive fast fails: $CONSECUTIVE_FAST_FAILS ==="
    if [ "$CONSECUTIVE_FAST_FAILS" -ge 2 ]; then
      echo "=== [$(date -Is)] FATAL: $CONSECUTIVE_FAST_FAILS consecutive fast failures -- stopping orchestrator, not retrying. Check $MAIN_LOG ==="
      exit 3
    fi
  else
    CONSECUTIVE_FAST_FAILS=0
  fi
done
