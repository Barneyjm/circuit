#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
E="uv run python scripts/eval_set.py"
echo "=== train 1.7B on real+synthetic (8k items, 1 epoch, wandb)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/train_real_plus_syn.jsonl --out runs/1.7b-real --epochs 1 --batch 8 --limit 8000 --max-length 640 --eval-every 100 --wandb s1proto --run-name 1.7b-real-8k 2>&1 | grep --line-buffered -E "train=|step [0-9]+0/|val:|saved|done in|View run|Traceback|Error"
echo "=== scoreboard 1.7B-real"; $E lora:runs/1.7b-real data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B_real.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== our-eval 1.7B-real"; $E lora:runs/1.7b-real data/eval.jsonl --batch 8 --out results/set_lora_1.7B_real.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== cmdiy 1.7B-real"; $E lora:runs/1.7b-real data/cmdiy_eval.jsonl --batch 8 --out results/cmdiy_lora_1.7B_real.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== PIPELINE13 DONE"
