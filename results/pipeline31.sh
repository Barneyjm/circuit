#!/usr/bin/env bash
# The vision and audio circuits with choice options encoded side by side, otherwise their
# v1.1 recipes. usage: pipeline31.sh vision | audio
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${1:?vision|audio}" in
  vision) N=circuit-vl-4b-par; M=Qwen/Qwen3-VL-4B-Instruct; G=""; X="--epochs 2 --batch 4 --max-length 1536 --eval-every 100"; EV="scripts/eval_vision.py"; D=data/vision/grid ;;
  audio)  N=circuit-audio-7b-par; M=Qwen/Qwen2-Audio-7B-Instruct; G="--group audio"; X="--epochs 2 --batch 2 --max-length 2048 --eval-every 100"; EV="scripts/eval_audio.py"; D=data/audio/grid ;;
esac
uv run $G python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
echo "=== train $N"
uv run $G python -u scripts/train_lora.py $M $D/train.jsonl --out runs/$N --modality $1 --head pointer $X --grad-checkpoint --parallel-options --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+00/|val:|kept step|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }
echo "=== permutations $N"; uv run $G python scripts/eval_permutations.py lora:runs/$N --media $1 --out results/perm_$N.json 2>&1 | grep -E "^  all|Traceback|Error"
echo "=== grid $N"; uv run $G python $EV $D/eval.jsonl --lora runs/$N --out results/grid_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
[ "$1" = vision ] && { echo "=== pope $N"; uv run python $EV data/vision/unseen/eval.jsonl --lora runs/$N --out results/pope_$N.json 2>&1 | grep -E "^timing|Traceback|Error"; }
echo "=== PIPELINE31 $N DONE"
