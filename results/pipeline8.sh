#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "PIPELINE7 DONE" results/pipeline7.log; do sleep 20; done
F='grep --line-buffered -v -iE "warning|Loading weights"'
E="uv run python scripts/eval_set.py"
echo "=== hf lora_1.7B";  $E lora:runs/1.7b-r16 data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B.json 2>&1 | eval $F | grep -E "^\s+family:|Traceback|Error"
echo "=== hf lora_8B";    $E lora:runs/8b-r16 data/hf_eval.jsonl --batch 8 --out results/hf_lora_8B.json 2>&1 | eval $F | grep -E "^\s+family:|Traceback|Error"
echo "=== PIPELINE8 DONE"
