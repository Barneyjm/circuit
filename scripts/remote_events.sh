#!/usr/bin/env bash
# Event stream for a GPU box: prints one line per NEW event and nothing when
# quiet, so it can drive a monitor that wakes on changes only.
#
#   scripts/remote_events.sh <ssh-host> <ssh-port> <log-path> [interval-seconds]
#
# Events: new pipeline stage lines, validation lines, tracebacks, "!!!" aborts,
# out-of-memory, GPU driver failure (nvidia-smi), no training/eval process while
# the pipeline is unfinished, and the box becoming unreachable or reachable again.
set -uo pipefail
HOST="${1:?}"; PORT="${2:?}"; LOG="${3:?}"; INTERVAL="${4:-120}"
KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
seen=0; down=0
while true; do
  out=$(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "root@$HOST" "
    grep -aE '^=== |val:|Traceback|!!!|OutOfMemory|done in|SELF-DESTRUCT' $LOG 2>/dev/null | tail -n +$((seen + 1)) | cut -c1-200
    echo '@@lines' \$(grep -acE '^=== |val:|Traceback|!!!|OutOfMemory|done in|SELF-DESTRUCT' $LOG 2>/dev/null)
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >/dev/null 2>&1 && echo '@@gpu ok' || echo '@@gpu FAIL'
    pgrep -f 'train_lora|eval_set' >/dev/null && echo '@@proc running' || echo '@@proc none'
    grep -aqE 'DONE|SELF-DESTRUCT|!!!' $LOG 2>/dev/null && echo '@@state finished' || echo '@@state running'
    true" </dev/null 2>/dev/null)
  if [ $? -ne 0 ]; then
    [ "$down" -eq 0 ] && echo "$(date '+%H:%M') UNREACHABLE: $HOST:$PORT"; down=1
  else
    [ "$down" -eq 1 ] && echo "$(date '+%H:%M') reachable again"; down=0
    echo "$out" | grep -vE '^@@' | grep -vE '^\s*$' | sed "s/^/$(date '+%H:%M') /"
    seen=$(echo "$out" | grep -oE '^@@lines [0-9]+' | awk '{print $2}'); seen=${seen:-0}
    echo "$out" | grep -q '@@gpu FAIL' && echo "$(date '+%H:%M') GPU FAIL: nvidia-smi cannot talk to the driver"
    if echo "$out" | grep -q '@@proc none' && echo "$out" | grep -q '@@state running'; then echo "$(date '+%H:%M') NO PROCESS: nothing training or evaluating but the pipeline is not finished"; fi
    echo "$out" | grep -q '@@state finished' && { echo "$(date '+%H:%M') PIPELINE FINISHED"; exit 0; }
  fi
  sleep "$INTERVAL"
done
