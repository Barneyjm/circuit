#!/usr/bin/env bash
# Sequential GPU work: wait for the small-model eval, then benches, then 8B.
set -u
cd ~/Documents/code/s1-proto
while kill -0 81598 2>/dev/null; do sleep 10; done
F='grep -v -iE "warning|Loading weights|it/s\]|examples/s\]"'
echo "=== bench 0.6B prefix-cache ON";  uv run python scripts/bench.py Qwen/Qwen3-0.6B-Base --n 30 2>&1 | eval $F
echo "=== bench 0.6B prefix-cache OFF"; uv run python scripts/bench.py Qwen/Qwen3-0.6B-Base --n 30 --no-prefix-cache 2>&1 | eval $F
echo "=== bench 0.6B fp16 ON";          uv run python scripts/bench.py Qwen/Qwen3-0.6B-Base --n 30 --dtype float16 2>&1 | eval $F
echo "=== bench 1.7B ON";               uv run python scripts/bench.py Qwen/Qwen3-1.7B-Base --n 30 2>&1 | eval $F
echo "=== bench 8B ON";                 uv run python scripts/bench.py Qwen/Qwen3-8B-Base --n 20 2>&1 | eval $F
echo "=== bench 8B OFF";                uv run python scripts/bench.py Qwen/Qwen3-8B-Base --n 20 --no-prefix-cache 2>&1 | eval $F
echo "=== eval 8B";                     uv run python scripts/eval_calibration.py Qwen/Qwen3-8B-Base --n 200 --out results/calib_Qwen3-8B-Base_T1.json 2>&1 | eval $F
echo "=== PIPELINE DONE"
