#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
E="uv run python scripts/eval_set.py"
echo "=== train 1.7B on the grid, holding out temporal/* and */thread"
uv run python -u scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/grid_train_holdout.jsonl --out runs/1.7b-grid-holdout --epochs 1 --batch 8 --max-length 768 --eval-every 100 --wandb s1proto --run-name 1.7b-grid-holdout 2>&1 | grep --line-buffered -E "train=|step [0-9]+0/|val:|saved|done in|View run|Traceback|Error"
echo "=== grid eval (all cells incl. held out)"; $E lora:runs/1.7b-grid-holdout data/grid_eval.jsonl --batch 8 --out results/grid_lora_1.7B_grid_holdout.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== water calls"; $E lora:runs/1.7b-grid-holdout data/water_calls.jsonl --batch 8 --out results/water_lora_1.7B_grid_holdout.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== hf cold eval (transfer to real data)"; $E lora:runs/1.7b-grid-holdout data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B_grid_holdout.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== cmdiy (Cole's bench)"; $E lora:runs/1.7b-grid-holdout data/cmdiy_eval.jsonl --batch 8 --out results/cmdiy_lora_1.7B_grid_holdout.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== PIPELINE18 DONE"
