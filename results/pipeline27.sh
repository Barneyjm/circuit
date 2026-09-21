#!/usr/bin/env bash
# publish_train_v2 = the publish mix + groundedness and tool-call families (build_grounded_tools.py).
# Same recipe as pipeline21/25 otherwise. usage: pipeline27.sh 1.7b | 8b   (both fit one 80 GB card)
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
case "${1:?1.7b|8b}" in
  1.7b) N=circuit-1.7b-v2; M=Qwen/Qwen3-1.7B-Base; X="--epochs 2"; B=16 ;;
  8b)   N=circuit-8b-v2;   M=Qwen/Qwen3-8B-Base;   X="--epochs 1"; B=8 ;;
esac
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
echo "=== train $N"
uv run python -u scripts/train_lora.py $M data/publish_train_v2.jsonl --out runs/$N --head pointer $X --batch 4 --max-length 1024 --eval-every 400 --grad-checkpoint --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+000/|val:|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }
for pair in "unseen_eval unseen" "grounded_tools_eval gt" "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
  set -- $pair
  echo "=== $2 $N"; $E lora:runs/$N data/$1.jsonl --batch $B --out results/${2}_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE27 $N DONE"
