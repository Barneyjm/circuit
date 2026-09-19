#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
F='grep --line-buffered -v -iE "warning|Loading weights"'
for m in "Qwen/Qwen3-8B-Base:8B_raw" "Qwen/Qwen3-14B-Base:14B_raw" "lora:runs/1.7b-r16:lora_1.7B" "lora:runs/8b-r16:lora_8B"; do
  spec="${m%%:*}"; [[ "$m" == lora:* ]] && spec="lora:${m#lora:}" && spec="${spec%%:*}"; tag="${m##*:}"
  [[ "$m" == lora:* ]] && spec="lora:$(echo "$m" | cut -d: -f2)"
  echo "=== $tag ($spec)"; uv run python scripts/eval_set.py "$spec" data/human_subset.jsonl --batch 8 --dump-logits "results/human_logits_${tag}.jsonl" --out "results/human_set_${tag}.json" 2>&1 | eval $F | head -12
done
echo "=== PIPELINE6 DONE"
