#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
J="uv run python scripts/eval_jev.py"
U="--url http://localhost:8009/v1/systemone --model kev-latest --concurrency 2"
echo "=== kev: water"; $J data/water_calls.jsonl --out results/water_kev.json $U 2>&1 | grep -E "^timing|error" | head -3
echo "=== kev: hf cold eval"; $J data/hf_eval.jsonl --out results/hf_kev.json $U 2>&1 | grep -E "^timing|error" | head -3
echo "=== kev: grid"; $J data/grid_eval.jsonl --out results/grid_kev.json $U 2>&1 | grep -E "^timing|error" | head -3
echo "=== kev: cmdiy"; $J data/cmdiy_eval.jsonl --out results/cmdiy_kev.json $U 2>&1 | grep -E "^timing|error" | head -3
echo "=== PIPELINE19 DONE"
