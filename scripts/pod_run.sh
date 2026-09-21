#!/usr/bin/env bash
# Rent the cheapest card that fits, run one job, bring the weights home, terminate. From the laptop:
#   scripts/pod_run.sh <run-name> <bases> <job...>
#   scripts/pod_run.sh circuit-8b-v2b Qwen/Qwen3-8B-Base ./results/pipeline28.sh 8b
# Commit and push first: the pod clones main. The pod is terminated on success, on failure, if
# setup makes no progress for STALL_MIN minutes, and after MAX_HOURS regardless.
set -uo pipefail
RUN="$1"; BASES="$2"; shift 2
SRC="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SRC"
set -a; . ./.env; set +a
KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
GPUS="${GPUS:-NVIDIA A40,NVIDIA RTX A6000}"          # 48 GB, about $0.50/hr; enough for LoRA on 8B and 14B
STALL_MIN="${STALL_MIN:-7}"; MAX_HOURS="${MAX_HOURS:-4}"
api () { curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }

[ -z "$(git log origin/main..HEAD --oneline)" ] || { echo "!!! push first; the pod clones main"; exit 1; }
DIRTY=$(git status --porcelain --untracked-files=no); [ -z "$DIRTY" ] || echo "note: the pod will not see these uncommitted changes:"$'\n'"$DIRTY"

REQ=$(GPUS="$GPUS" RUN="$RUN" KEYFILE="$KEY.pub" python3 -c '
import json, os
print(json.dumps({"name": os.environ["RUN"], "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "gpuTypeIds": os.environ["GPUS"].split(","), "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 30,
  "volumeInGb": 80, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": open(os.environ["KEYFILE"]).read().strip()}}))')
POD=$(api -X POST https://rest.runpod.io/v1/pods -d "$REQ" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or ""); print(d, file=sys.stderr) if not d.get("id") else None')
[ -n "$POD" ] || { echo "!!! could not create a pod"; exit 1; }
kill_pod () { api -X DELETE -o /dev/null -w "terminated $POD (HTTP %{http_code})\n" "https://rest.runpod.io/v1/pods/$POD"; echo "pods remaining: $(api https://rest.runpod.io/v1/pods)"; }
trap kill_pod EXIT
START=$(date +%s)

for _ in $(seq 60); do
  ADDR=$(api "https://rest.runpod.io/v1/pods/$POD" | python3 -c 'import json,sys; d=json.load(sys.stdin); pm=d.get("portMappings") or {}; print(d.get("publicIp") or "", pm.get("22") or "", d.get("costPerHr"))')
  read -r H P COST <<<"$ADDR"; [ -n "$H" ] && [ -n "$P" ] && break; sleep 5
done
[ -n "${P:-}" ] || { echo "!!! pod never got an address"; exit 1; }
echo "pod $POD at \$$COST/hr"
S="ssh -i $KEY -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$H"
for _ in $(seq 40); do $S true 2>/dev/null && break; sleep 5; done

$S 'mkdir -p /workspace/stage'
scp -q -i "$KEY" -P "$P" .env scripts/pod_bootstrap.sh "root@$H:/workspace/stage/"
for f in ${STAGE:-}; do scp -q -i "$KEY" -P "$P" "$f" "root@$H:/workspace/stage/"; done
JOB=$(printf '%q ' "$@")
$S "nohup bash /workspace/stage/pod_bootstrap.sh '$BASES' $JOB > /workspace/boot.log 2>&1 &" </dev/null
echo "launched after $(( $(date +%s) - START )) s"

seen=0; last_change=$(date +%s); training=0
while true; do
  sleep 60
  now=$(date +%s)
  LOG=$($S 'cat /workspace/boot.log' 2>/dev/null) || continue
  n=$(printf '%s\n' "$LOG" | wc -l)
  if [ "$n" -gt "$seen" ]; then
    printf '%s\n' "$LOG" | tail -n +$((seen + 1)) | grep -E "== stage|^torch|cached|=== |val:|kept step|done in|View run|Traceback|Error|!!!" || true
    seen=$n; last_change=$now
  fi
  printf '%s\n' "$LOG" | grep -q "=== train" && training=1
  if printf '%s\n' "$LOG" | grep -qE "PIPELINE[0-9]+ .*DONE"; then echo "job finished after $(( (now - START) / 60 )) min"; break; fi
  if printf '%s\n' "$LOG" | grep -qE "!!!|Traceback"; then echo "!!! job failed"; printf '%s\n' "$LOG" | tail -n 15; break; fi
  if [ "$training" = 0 ] && [ $(( now - last_change )) -gt $(( STALL_MIN * 60 )) ]; then echo "!!! setup stalled for $STALL_MIN min; terminating"; printf '%s\n' "$LOG" | tail -n 8; exit 2; fi
  if [ $(( now - START )) -gt $(( MAX_HOURS * 3600 )) ]; then echo "!!! over $MAX_HOURS h; collecting what exists and terminating"; break; fi
done

# the kept checkpoint and the results; the per-step checkpoints stay behind
R="rsync -rltz --no-owner --no-group -e"
$R "ssh -i $KEY -p $P" --include='/*/' --include='/*/adapter/***' --include='/*/head.pt' --include='/*/config.json' --exclude='*' "root@$H:/workspace/s1-proto/runs/" runs/
$R "ssh -i $KEY -p $P" --include='*.json' --exclude='*' "root@$H:/workspace/s1-proto/results/" results/
ls -la "runs/$RUN/head.pt" "runs/$RUN/adapter/adapter_model.safetensors" 2>&1 | awk '{print $5, $9}'
echo "cost: about \$$(python3 -c "print(round(($(date +%s) - $START) / 3600 * $COST, 2))")"
