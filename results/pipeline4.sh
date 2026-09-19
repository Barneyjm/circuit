#!/usr/bin/env bash
# Take over from pipeline3 when it reaches the 8B training step: the
# full 2-epoch 8B run would take ~15 h at MPS speeds. Run a bounded
# 8B job instead (1 epoch over 2,400 items), then evaluate.
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "=== P3 train 8B" results/pipeline3.log; do sleep 15; done
pkill -f "results/pipeline3.sh"; sleep 1; pkill -f "train_lora.py Qwen/Qwen3-8B-Base"; sleep 5
F='grep -v -iE "warning|Loading weights|it/s\]|examples/s\]"'
E="uv run python scripts/eval_set.py"
echo "=== P3 train 8B (bounded)"; uv run python scripts/train_lora.py Qwen/Qwen3-8B-Base data/train.jsonl --out runs/8b-r16 --epochs 1 --batch 8 --limit 2400 --eval-every 60 2>&1 | eval $F
echo "=== P3 eval 8B lora";       $E lora:runs/8b-r16 data/eval.jsonl --out results/set_lora_8B.json 2>&1 | eval $F
echo "=== P3 eval 8B lora fit-temps"; $E lora:runs/8b-r16 data/eval.jsonl --fit-temps --out results/set_lora_8B_temps.json 2>&1 | eval $F
echo "=== PIPELINE4 DONE"
