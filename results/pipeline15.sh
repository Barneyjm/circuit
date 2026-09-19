#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
E="uv run python scripts/eval_set.py"
echo "=== train 1.7B on real+synthetic (8k items, 1 epoch, left-truncation fix, max_length 1024)"
uv run python -u scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/train_real_plus_syn.jsonl --out runs/1.7b-real-fix --epochs 1 --batch 8 --limit 8000 --max-length 1024 --eval-every 100 --wandb s1proto --run-name 1.7b-real-fix-8k-v2 2>&1 | grep --line-buffered -E "train=|step [0-9]+0/|val:|saved|done in|View run|Traceback|Error"
echo "=== scoreboard 1.7B-real-fix"; $E lora:runs/1.7b-real-fix data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B_real_fix.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== our-eval 1.7B-real-fix"; $E lora:runs/1.7b-real-fix data/eval.jsonl --batch 8 --out results/set_lora_1.7B_real_fix.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== cmdiy 1.7B-real-fix"; $E lora:runs/1.7b-real-fix data/cmdiy_eval.jsonl --batch 8 --out results/cmdiy_lora_1.7B_real_fix.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== PIPELINE15 DONE"
echo "=== timing 1.7B-real-fix"; $E lora:runs/1.7b-real-fix data/hf_eval_timing.jsonl --batch 8 --out results/hf_timing_lora_1.7B_real_fix.json 2>&1 | grep --line-buffered -E "^timing|Traceback|Error"
echo "=== PIPELINE15 ALL DONE"
