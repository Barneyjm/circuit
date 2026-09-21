#!/usr/bin/env bash
# Runs ON a rented pod. Clones the public repo (datacenter bandwidth, seconds), installs,
# checks CUDA, caches the base models, then execs the job. Small untracked files (.env, a
# gitignored eval set) are expected in /workspace/stage, put there by scripts/pod_run.sh.
#   bash pod_bootstrap.sh "Qwen/Qwen3-8B-Base" ./results/pipeline28.sh 8b
set -euo pipefail
BASES="$1"; shift
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache
mkdir -p "$HF_HOME" /workspace/stage
echo "== stage: clone"
cd /workspace
[ -d s1-proto/.git ] || git clone -q --depth 1 https://github.com/Barneyjm/circuit s1-proto
cd s1-proto
[ -f /workspace/stage/.env ] && cp /workspace/stage/.env .env
find /workspace/stage -maxdepth 1 -name '*.jsonl' -exec cp {} data/ \;
echo "== stage: deps"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
uv sync -q
echo "== stage: cuda"
cuda_ok () { uv run --no-sync python -c "import torch, sys; print('torch', torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); sys.exit(0 if torch.cuda.is_available() else 1)"; }
if ! cuda_ok; then
  # the lockfile's torch targets a newer CUDA than this host's driver
  uv pip install -q --reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  cuda_ok || { echo "!!! no usable CUDA torch; aborting"; exit 3; }
fi
echo "== stage: models"
BASES="$BASES" uv run --no-sync python - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in os.environ["BASES"].split(","):
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]); print("cached", m)
PY
echo "== stage: launch $*"
chmod +x results/*.sh
exec "$@"
