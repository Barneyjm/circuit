#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
while kill -0 86832 2>/dev/null; do sleep 15; done
F='grep -v -iE "warning|Loading weights|it/s\]|examples/s\]"'
echo "=== eval 8B debias";  uv run python scripts/eval_calibration.py Qwen/Qwen3-8B-Base --n 200 --debias --out results/calib_Qwen3-8B-Base_T1_debias.json 2>&1 | eval $F
echo "=== bench 14B ON";    uv run python scripts/bench.py Qwen/Qwen3-14B-Base --n 15 2>&1 | eval $F
echo "=== eval 14B";        uv run python scripts/eval_calibration.py Qwen/Qwen3-14B-Base --n 200 --out results/calib_Qwen3-14B-Base_T1.json 2>&1 | eval $F
echo "=== eval 14B debias"; uv run python scripts/eval_calibration.py Qwen/Qwen3-14B-Base --n 200 --debias --out results/calib_Qwen3-14B-Base_T1_debias.json 2>&1 | eval $F
echo "=== PIPELINE2 DONE"
