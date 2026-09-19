#!/usr/bin/env bash
# circuit-vl-4b: Qwen3-VL-4B + LoRA (language model) + pointer head, trained on the vision grid.
set -u
cd /workspace/s1-proto
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! GPU LOST at start"; exit 3; }
uv pip install -q torchvision 2>/dev/null
echo "=== train circuit-vl-4b (vision grid, pointer head)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-VL-4B-Instruct data/vision/grid/train.jsonl --out runs/circuit-vl-4b --modality vision --head pointer --epochs 2 --batch 4 --max-length 1536 --eval-every 100 --grad-checkpoint --wandb s1proto --run-name circuit-vl-4b 2>&1 | grep --line-buffered -E "train=|step [0-9]+0/|val:|saved|done in|View run|Traceback|Error"
[ -f runs/circuit-vl-4b/head.pt ] || { echo "!!! circuit-vl-4b did not train"; exit 2; }
echo "=== vgrid circuit-vl-4b"; uv run python scripts/eval_vision.py data/vision/grid/eval.jsonl --lora runs/circuit-vl-4b --out results/vgrid_circuit-vl-4b.json 2>&1 | grep -E "^timing|Traceback|Error|acc="
echo "=== vgrid raw"; uv run python scripts/eval_vision.py data/vision/grid/eval.jsonl --out results/vgrid_qwen3vl4b_raw_gpu.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== PIPELINE24 DONE"
