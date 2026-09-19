#!/usr/bin/env bash
# circuit-1.7b evals on the laptop (insurance while the pod handles the 8B)
set -u
cd ~/Documents/code/s1-proto
E="uv run python scripts/eval_set.py"
for pair in "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy" "hf_eval_timing timing"; do
  set -- $pair
  echo "=== $2 circuit-1.7b"; $E lora:runs/circuit-1.7b data/$1.jsonl --batch 8 --out results/${2}_circuit-1.7b.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE22 DONE"
