#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "PIPELINE8 DONE" results/pipeline8.log; do sleep 20; done
F='grep --line-buffered -v -iE "warning|Loading weights"'
E="uv run python scripts/eval_set.py"
for m in "Qwen/Qwen3-0.6B-Base:0.6B_raw" "Qwen/Qwen3-1.7B-Base:1.7B_raw" "Qwen/Qwen3-8B-Base:8B_raw" "Qwen/Qwen3-14B-Base:14B_raw" "lora:runs/1.7b-r16:lora_1.7B" "lora:runs/8b-r16:lora_8B"; do
  tag="${m##*:}"; spec="${m%:*}"
  echo "=== timing $tag"; $E "$spec" data/hf_eval_timing.jsonl --batch 8 --out "results/hf_timing_${tag}.json" 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE9 DONE"
