#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
E="uv run python scripts/eval_set.py"
echo "=== P3 train 8B (lean)"; uv run python -u scripts/train_lora.py Qwen/Qwen3-8B-Base data/train.jsonl --out runs/8b-r16 --epochs 1 --batch 4 --limit 2400 --max-length 512 --eval-every 100 --grad-checkpoint 2>&1 | grep --line-buffered -v -iE "warning|Loading weights|it/s\]"
echo "=== P3 eval 8B lora";  $E lora:runs/8b-r16 data/eval.jsonl --batch 8 --out results/set_lora_8B.json 2>&1 | grep --line-buffered -v -iE "warning|Loading weights"
echo "=== PIPELINE5 DONE"
