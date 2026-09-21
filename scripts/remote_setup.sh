#!/usr/bin/env bash
# Provision a fresh GPU box (RunPod / any Ubuntu + CUDA image) for s1proto.
#
#   scripts/remote_setup.sh <ssh-host> [ssh-port]        # from the laptop
#
# Rsyncs this repo (minus runs, wandb, .venv), installs uv and CUDA torch,
# copies the ignored .env (keys) so W&B and TypeSafe work, and pre-downloads
# the Qwen bases. Idempotent. Then:
#
#   ssh -p PORT root@HOST 'cd /workspace/s1-proto && ./results/pipeline21.sh'
set -euo pipefail
HOST="${1:?ssh host}"
PORT="${2:-22}"
KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

echo "== rsync repo"
$SSH 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)' </dev/null
rsync -rltDz --no-owner --no-group --delete -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude .venv --exclude runs --exclude wandb --exclude '.git/objects' --exclude '__pycache__' --exclude '.pytest_cache' \
  "$SRC/" "root@$HOST:/workspace/s1-proto/"
rsync -rltDz --no-owner --no-group --delete -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude .venv --exclude dist --exclude '.git/objects' --exclude '__pycache__' --exclude '.pytest_cache' \
  "$SRC/../decision-circuits/" "root@$HOST:/workspace/decision-circuits/"
scp -i "$KEY" -P "$PORT" "$SRC/.env" "root@$HOST:/workspace/s1-proto/.env"

$SSH BASES="${BASES:-}" bash -s <<'EOF'
set -euo pipefail
cd /workspace/s1-proto
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache
mkdir -p "$HF_HOME"
uv sync -q
# CUDA torch: the lockfile pins the Mac wheel; swap in the CUDA build
uv pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision 2>/dev/null || uv pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision
# --no-sync from here on: a sync puts the lockfile torch back, built for a newer CUDA than most rented drivers
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
import os
for m in os.environ.get("BASES", "Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-8B-Base").split(","):
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]); print("cached", m)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== ready"
EOF
