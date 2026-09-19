#!/usr/bin/env bash
# circuit-8b on a 48GB card: bf16 LoRA, pointer head, clean data. CUDA is verified before every stage;
# a lost GPU aborts the pipeline instead of falling back to CPU.
set -u
cd /workspace/s1-proto
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
need_gpu () { uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! GPU LOST at $1; aborting"; exit 3; }; }
need_gpu "start"
echo "=== train circuit-8b (pointer head, clean data, bf16)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-8B-Base data/publish_train.jsonl --out runs/circuit-8b --head pointer --epochs 1 --batch 4 --max-length 1024 --eval-every 400 --grad-checkpoint --wandb s1proto --run-name circuit-8b 2>&1 | grep --line-buffered -E "train=|step [0-9]+00/|val:|saved|done in|View run|Traceback|Error"
[ -f runs/circuit-8b/head.pt ] || { echo "!!! circuit-8b did not train"; exit 2; }
for pair in "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy" "hf_eval_timing timing"; do
  set -- $pair
  need_gpu "$2"
  echo "=== $2 circuit-8b"; $E lora:runs/circuit-8b data/$1.jsonl --batch 8 --out results/${2}_circuit-8b.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE23 DONE"
