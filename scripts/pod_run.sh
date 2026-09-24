#!/usr/bin/env bash
# Rent the cheapest card that fits, run one job, bring the weights home, terminate. From the laptop:
#   scripts/pod_run.sh <run-name> <bases> <job...>
#   scripts/pod_run.sh circuit-8b-v2b Qwen/Qwen3-8B-Base ./results/pipeline28.sh 8b
# Commit and push first: the pod clones main. The pod is terminated on success, on failure, if
# setup makes no progress for STALL_MIN minutes, and after MAX_HOURS regardless.
# Only the keys a job needs go to the pod (POD_KEYS, default W&B and Hugging Face); the RunPod key
# and everything else in .env stay here. CLOUD=community rents from community hosts, cheaper and
# run by whoever owns the machine, so use it for public data only.
set -uo pipefail
RUN="$1"; BASES="$2"; shift 2
SRC="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SRC"
set -a; . ./.env; set +a
KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
GPUS="${GPUS:-NVIDIA A40,NVIDIA RTX A6000}"          # 48 GB, about $0.50/hr; enough for LoRA on 8B and 14B
STALL_MIN="${STALL_MIN:-7}"; MAX_HOURS="${MAX_HOURS:-4}"
CLOUD="${CLOUD:-secure}"; POD_KEYS="${POD_KEYS:-WANDB_API_KEY HF_TOKEN}"
# Minutes to wait for the pod's address. A community host may pull the image itself first.
ADDR_MIN="${ADDR_MIN:-$([ "$CLOUD" = community ] && echo 12 || echo 5)}"
case "$CLOUD" in secure) CLOUD_TYPE=SECURE ;; community) CLOUD_TYPE=COMMUNITY ;; *) echo "!!! CLOUD is secure or community"; exit 1 ;; esac
case " $POD_KEYS " in *" RUNPOD_API_KEY "*) echo "!!! RUNPOD_API_KEY never goes to a pod"; exit 1 ;; esac
POD_ENV=$(mktemp); trap 'rm -f "$POD_ENV"' EXIT
for k in $POD_KEYS; do [ -n "${!k:-}" ] && printf '%s=%q\n' "$k" "${!k}" >> "$POD_ENV"; done
api () { curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }

[ -z "$(git log origin/main..HEAD --oneline)" ] || { echo "!!! push first; the pod clones main"; exit 1; }
DIRTY=$(git status --porcelain --untracked-files=no); [ -z "$DIRTY" ] || echo "note: the pod will not see these uncommitted changes:"$'\n'"$DIRTY"

# supportPublicIp: a community host gives no address for ssh unless asked; secure ones always do.
REQ=$(GPUS="$GPUS" RUN="$RUN" KEYFILE="$KEY.pub" CLOUD_TYPE="$CLOUD_TYPE" python3 -c '
import json, os
print(json.dumps({"name": os.environ["RUN"], "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "gpuTypeIds": os.environ["GPUS"].split(","), "gpuCount": 1, "cloudType": os.environ["CLOUD_TYPE"],
  "supportPublicIp": True, "containerDiskInGb": 30,
  "volumeInGb": 80, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": open(os.environ["KEYFILE"]).read().strip()}}))')
POD=$(api -X POST https://rest.runpod.io/v1/pods -d "$REQ" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or ""); print(d, file=sys.stderr) if not d.get("id") else None')
[ -n "$POD" ] || { echo "!!! could not create a pod"; exit 1; }
kill_pod () { api -X DELETE -o /dev/null -w "terminated $POD (HTTP %{http_code})\n" "https://rest.runpod.io/v1/pods/$POD"; echo "pods remaining: $(api https://rest.runpod.io/v1/pods)"; }
trap 'kill_pod; rm -f "$POD_ENV"' EXIT
START=$(date +%s)

for _ in $(seq $(( ADDR_MIN * 12 ))); do
  ADDR=$(api "https://rest.runpod.io/v1/pods/$POD" | python3 -c 'import json,sys; d=json.load(sys.stdin); pm=d.get("portMappings") or {}; print(d.get("publicIp") or "", pm.get("22") or "", d.get("costPerHr"))')
  read -r H P COST <<<"$ADDR"; [ -n "$H" ] && [ -n "$P" ] && break; sleep 5
done
[ -n "${P:-}" ] || { echo "!!! pod never got an address"; exit 1; }
echo "pod $POD at \$$COST/hr ($CLOUD cloud; keys staged: $(cut -d= -f1 "$POD_ENV" | tr '\n' ' '))"
S="ssh -i $KEY -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$H"
for _ in $(seq 40); do $S true 2>/dev/null && break; sleep 5; done

$S 'mkdir -p /workspace/stage'
scp -q -i "$KEY" -P "$P" "$POD_ENV" "root@$H:/workspace/stage/.env"
scp -q -i "$KEY" -P "$P" scripts/pod_bootstrap.sh "root@$H:/workspace/stage/"
for f in ${STAGE:-}; do scp -q -i "$KEY" -P "$P" "$f" "root@$H:/workspace/stage/"; done
JOB=$(printf '%q ' "$@")
$S "MEDIA_DATASET='${MEDIA_DATASET:-}' HF_TOKEN='${HF_TOKEN:-}' nohup bash /workspace/stage/pod_bootstrap.sh '$BASES' $JOB > /workspace/boot.log 2>&1 &" </dev/null
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

# The kept checkpoint, every step, and the results come home before anything is terminated. tar over ssh, because
# a fresh pod has no rsync; the per-step checkpoints stay behind. A pod whose weights could not be
# fetched is left running, loudly: an hour of an A40 costs less than the run that produced them.
collect () {
  # steps/ holds every evaluated checkpoint; for a 1.7B that is about 1.6 GB, for an 8B about 4 GB
  # RUN may name several runs separated by commas (a job that trains two models at once)
  local paths="" ok=1
  for r in ${RUN//,/ }; do paths="$paths runs/$r/adapter runs/$r/head.pt runs/$r/config.json \$( [ -d runs/$r/steps ] && echo runs/$r/steps )"; done
  $S "cd /workspace/s1-proto && tar -cf - $paths results/*.json" | tar -xf - -C "$SRC"
  for r in ${RUN//,/ }; do [ -s "runs/$r/head.pt" ] && [ -s "runs/$r/adapter/adapter_model.safetensors" ] && [ -s "runs/$r/config.json" ] || ok=0; done
  [ "$ok" = 1 ]
}
if [ "${collected:-0}" = 0 ]; then
  for attempt in 1 2 3; do collect && { collected=1; break; }; echo "collect attempt $attempt failed"; sleep 20; done
fi
if [ "${collected:-0}" != 1 ] && $S "for r in ${RUN//,/ }; do test -s /workspace/s1-proto/runs/\$r/head.pt && exit 0; done; exit 1" 2>/dev/null; then
  trap - EXIT
  echo "!!! WEIGHTS EXIST ON THE POD BUT COULD NOT BE FETCHED. POD $POD LEFT RUNNING at \$$COST/hr."
  echo "!!! fetch by hand: ssh -i $KEY -p $P root@$H   then terminate it."
  exit 4
fi
for r in ${RUN//,/ }; do ls -la "runs/$r/head.pt" "runs/$r/adapter/adapter_model.safetensors" 2>&1 | awk '{print $5, $9}'; done
echo "cost: about \$$(python3 -c "print(round(($(date +%s) - $START) / 3600 * $COST, 2))")"
