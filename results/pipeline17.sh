#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
until grep -q "PIPELINE16 DONE" results/pipeline16.log; do sleep 15; done
E="uv run python scripts/eval_set.py"
export TYPESAFE_API_KEY="$(grep '^TYPESAFE_API_KEY=' ~/Documents/code/camino-core/.env | cut -d= -f2- | tr -d '"')"
echo "=== rebuild grid (ambiguous rows)"; uv run python scripts/build_grid.py --per-cell 150 --eval-per-cell 40 2>&1 | tail -2
echo "=== grid: jev"; uv run python scripts/eval_jev.py data/grid_eval.jsonl --out results/grid_jev.json --concurrency 8 2>&1 | grep -E "^timing|Traceback" | head -3
echo "=== water: jev"; uv run python scripts/eval_jev.py data/water_calls.jsonl --out results/water_jev.json --concurrency 8 2>&1 | grep -E "^timing|Traceback" | head -3
echo "=== grid: 1.7B raw"; $E Qwen/Qwen3-1.7B-Base data/grid_eval.jsonl --batch 8 --out results/grid_1.7B_raw.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== water: 1.7B raw"; $E Qwen/Qwen3-1.7B-Base data/water_calls.jsonl --batch 8 --out results/water_1.7B_raw.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== grid: 1.7B real fine-tune"; $E lora:runs/1.7b-real data/grid_eval.jsonl --batch 8 --out results/grid_lora_1.7B_real.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== water: 1.7B real fine-tune"; $E lora:runs/1.7b-real data/water_calls.jsonl --batch 8 --out results/water_lora_1.7B_real.json 2>&1 | grep -E "^timing|Traceback|Error"
echo "=== PIPELINE17 DONE"
