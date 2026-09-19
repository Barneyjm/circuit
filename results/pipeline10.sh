#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
true
F='grep --line-buffered -v -iE "warning|Loading weights"'
E="uv run python scripts/eval_set.py"
for m in "Qwen/Qwen3-0.6B-Base:0.6B_raw" "Qwen/Qwen3-8B-Base:8B_raw" "Qwen/Qwen3-14B-Base:14B_raw" "lora:runs/1.7b-r16:lora_1.7B" "lora:runs/8b-r16:lora_8B"; do
  tag="${m##*:}"; spec="${m%:*}"
  echo "=== cmdiy $tag"; $E "$spec" data/cmdiy_eval.jsonl --batch 8 --out "results/cmdiy_${tag}.json" 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE10 DONE"
