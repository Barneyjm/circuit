#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
F='grep --line-buffered -v -iE "warning|Loading weights"'
E="uv run python scripts/eval_set.py"
echo "=== hf lora_1.7B";  $E lora:runs/1.7b-r16 data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B.json 2>&1 | eval $F | grep -E "^\s+family:|\"all\"" 
echo "=== hf lora_8B";    $E lora:runs/8b-r16 data/hf_eval.jsonl --batch 8 --out results/hf_lora_8B.json 2>&1 | eval $F | grep -E "^\s+family:"
echo "=== hf 8B_raw";     $E Qwen/Qwen3-8B-Base data/hf_eval_small.jsonl --batch 8 --out results/hf_8B_raw.json 2>&1 | eval $F | grep -E "^\s+family:"
echo "=== hf 14B_raw";    $E Qwen/Qwen3-14B-Base data/hf_eval_small.jsonl --batch 8 --out results/hf_14B_raw.json 2>&1 | eval $F | grep -E "^\s+family:"
echo "=== PIPELINE7 DONE"
