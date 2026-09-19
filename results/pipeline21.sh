#!/usr/bin/env bash
# The publish run, on a GPU box (see scripts/remote_setup.sh).
#   circuit-1.7b: pointer head, clean data (grid + commercial wide mix + CC real data), 2 epochs
#   then every benchmark: grid, cold eval, water calls, Cole's bench, timing slice
#   then the same recipe at 8B
set -u
cd /workspace/s1-proto
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
run_evals () {  # $1 = run dir, $2 = tag
  echo "=== grid $2";   $E lora:$1 data/grid_eval.jsonl        --batch 16 --out results/grid_$2.json   2>&1 | grep -E "^timing|Traceback|Error"
  echo "=== hf $2";     $E lora:$1 data/hf_eval.jsonl          --batch 16 --out results/hf_$2.json     2>&1 | grep -E "^timing|Traceback|Error"
  echo "=== water $2";  $E lora:$1 data/water_calls.jsonl      --batch 16 --out results/water_$2.json  2>&1 | grep -E "^timing|Traceback|Error"
  echo "=== cmdiy $2";  $E lora:$1 data/cmdiy_eval.jsonl       --batch 16 --out results/cmdiy_$2.json  2>&1 | grep -E "^timing|Traceback|Error"
  echo "=== timing $2"; $E lora:$1 data/hf_eval_timing.jsonl   --batch 1  --out results/timing_$2.json 2>&1 | grep -E "^timing|Traceback|Error"
}
echo "=== train circuit-1.7b (pointer head, clean data)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/publish_train.jsonl --out runs/circuit-1.7b --head pointer --epochs 2 --batch 4 --max-length 1024 --eval-every 400 --grad-checkpoint --wandb s1proto --run-name circuit-1.7b 2>&1 | grep --line-buffered -E "train=|step [0-9]+00/|val:|saved|done in|View run|Traceback|Error"
[ -f runs/circuit-1.7b/head.pt ] && run_evals runs/circuit-1.7b circuit-1.7b || echo "!!! circuit-1.7b did not train; skipping evals"
echo "=== train circuit-8b (pointer head, clean data)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-8B-Base data/publish_train.jsonl --out runs/circuit-8b --head pointer --epochs 1 --batch 2 --max-length 1024 --eval-every 800 --grad-checkpoint --load-4bit --wandb s1proto --run-name circuit-8b 2>&1 | grep --line-buffered -E "train=|step [0-9]+00/|val:|saved|done in|View run|Traceback|Error"
[ -f runs/circuit-8b/head.pt ] && run_evals runs/circuit-8b circuit-8b || echo "!!! circuit-8b did not train; skipping evals"
echo "=== PIPELINE21 DONE"
