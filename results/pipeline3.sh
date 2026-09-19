#!/usr/bin/env bash
# Phase 2 (post-hoc calibration on the reference eval set) then Phase 3
# (LoRA + soft-target head), sequential on the GPU. Waits for the 14B
# pipeline and for both data generators to finish.
set -u
cd ~/Documents/code/s1-proto
while ! grep -q "PIPELINE2 DONE" results/pipeline2.log; do sleep 20; done
while ! grep -q "^done:" data/gen_eval.log; do sleep 20; done
F='grep -v -iE "warning|Loading weights|it/s\]|examples/s\]"'
E="uv run python scripts/eval_set.py"
echo "=== P2 8B raw";          $E Qwen/Qwen3-8B-Base data/eval.jsonl --out results/set_8B_raw.json --dump-logits results/logits_8B_raw.jsonl 2>&1 | eval $F
echo "=== P2 8B fit-temps";    $E Qwen/Qwen3-8B-Base data/eval.jsonl --fit-temps --out results/set_8B_temps.json 2>&1 | eval $F
echo "=== P2 8B perm3+fit";    $E Qwen/Qwen3-8B-Base data/eval.jsonl --permutations 3 --fit-temps --out results/set_8B_perm3_temps.json 2>&1 | eval $F
echo "=== P2 8B debias+fit";   $E Qwen/Qwen3-8B-Base data/eval.jsonl --debias --fit-temps --out results/set_8B_debias_temps.json 2>&1 | eval $F
echo "=== P2 14B raw";         $E Qwen/Qwen3-14B-Base data/eval.jsonl --out results/set_14B_raw.json 2>&1 | eval $F
echo "=== P2 14B perm3+fit";   $E Qwen/Qwen3-14B-Base data/eval.jsonl --permutations 3 --fit-temps --out results/set_14B_perm3_temps.json 2>&1 | eval $F
while ! grep -q "^done:" data/gen_train.log; do sleep 20; done
echo "=== P3 train 1.7B";      uv run python scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/train.jsonl --out runs/1.7b-r16 --epochs 2 --batch 8 --eval-every 100 2>&1 | eval $F
echo "=== P3 eval 1.7B lora";  $E lora:runs/1.7b-r16 data/eval.jsonl --out results/set_lora_1.7B.json 2>&1 | eval $F
echo "=== P3 train 8B";        uv run python scripts/train_lora.py Qwen/Qwen3-8B-Base data/train.jsonl --out runs/8b-r16 --epochs 2 --batch 8 --eval-every 100 2>&1 | eval $F
echo "=== P3 eval 8B lora";    $E lora:runs/8b-r16 data/eval.jsonl --out results/set_lora_8B.json 2>&1 | eval $F
echo "=== P3 eval 8B lora fit-temps"; $E lora:runs/8b-r16 data/eval.jsonl --fit-temps --out results/set_lora_8B_temps.json 2>&1 | eval $F
echo "=== PIPELINE3 DONE"
