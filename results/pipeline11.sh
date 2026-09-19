#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "PIPELINE10 DONE" results/pipeline10.log; do sleep 20; done
F='grep --line-buffered -v -iE "warning|Loading weights"'
echo "=== hf 1.7B_raw (control for the 1.7B fine-tune)"; uv run python scripts/eval_set.py Qwen/Qwen3-1.7B-Base data/hf_eval_small.jsonl --batch 8 --out results/hf_1.7B_raw.json 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
echo "=== PIPELINE11 DONE"
