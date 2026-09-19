#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "PIPELINE10 DONE" results/pipeline10.log; do sleep 30; done
F='grep --line-buffered -v -iE "warning|Loading weights|it/s\]"'
E="uv run python scripts/eval_set.py"
echo "=== train 1.7B on real+synthetic"; uv run python -u scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/train_real_plus_syn.jsonl --out runs/1.7b-real --epochs 1 --batch 8 --max-length 640 --eval-every 200 2>&1 | eval $F | grep -E "train=|val:|saved|done in|Traceback|Error"
echo "=== scoreboard 1.7B-real"; $E lora:runs/1.7b-real data/hf_eval.jsonl --batch 8 --out results/hf_lora_1.7B_real.json 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
echo "=== our-eval 1.7B-real"; $E lora:runs/1.7b-real data/eval.jsonl --batch 8 --out results/set_lora_1.7B_real.json 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
echo "=== cmdiy 1.7B-real"; $E lora:runs/1.7b-real data/cmdiy_eval.jsonl --batch 8 --out results/cmdiy_lora_1.7B_real.json 2>&1 | eval $F | grep -E "^timing|Traceback|Error"
echo "=== PIPELINE12 DONE"
