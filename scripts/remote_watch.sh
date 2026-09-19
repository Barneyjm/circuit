#!/usr/bin/env bash
# Babysit a GPU box running a pipeline: collect weights + results every
# INTERVAL seconds, and when the pipeline reports done (or the box goes
# away), collect once more and terminate the pod so nothing idles.
#
#   scripts/remote_watch.sh <ssh-host> <ssh-port> <pod-id> [log-path] [interval-seconds]
set -uo pipefail
HOST="${1:?ssh host}"; PORT="${2:?ssh port}"; POD="${3:?pod id}"
LOG="${4:-/workspace/s1-proto/results/pipeline21.log}"; INTERVAL="${5:-1200}"
KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(grep '^RUNPOD_API_KEY=' "$SRC/.env" | cut -d= -f2-)}"

terminate () {
  echo "$(date '+%H:%M:%S') terminating pod $POD"
  (cd "$SRC" && uv run python -c "import os, runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; print(runpod.terminate_pod('$POD'))")
}
collect () {
  "$SRC/scripts/remote_collect.sh" "$HOST" "$PORT" 2>/dev/null | tail -1
}

while true; do
  # grep exits 1 on "no match"; only the ssh connection itself decides reachability, so end the remote command with `true`.
  status=$(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "root@$HOST" "grep -aoE 'PIPELINE[0-9]* DONE|SELF-DESTRUCT' $LOG 2>/dev/null | head -1; grep -acE 'Traceback' $LOG 2>/dev/null; true" </dev/null 2>/dev/null)
  rc=$?
  echo "$(date '+%H:%M:%S') status: $(echo "$status" | tr '\n' ' ') (ssh rc=$rc)"
  if [ $rc -ne 0 ]; then
    fails=$((${fails:-0}+1)); [ "$fails" -ge 6 ] && { echo "box unreachable for 6 checks; terminating"; terminate; exit 1; }
  else
    fails=0
    collect
    if echo "$status" | grep -qE "PIPELINE[0-9]* DONE|SELF-DESTRUCT"; then
      echo "pipeline finished; final collect"; collect; terminate; echo "=== WATCH DONE"; exit 0
    fi
  fi
  sleep "$INTERVAL"
done
