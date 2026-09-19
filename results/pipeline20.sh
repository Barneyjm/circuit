#!/usr/bin/env bash
set -u
cd ~/Documents/code/s1-proto
N="uv run python scripts/eval_nimble.py"
D="--nimble /private/tmp/claude-501/-Users-jbarney-Documents-code-camino-core/c4d971aa-18dd-4093-be69-159fe0f4680f/scratchpad/nimble-model"
echo "=== nimble: water"; $N data/water_calls.jsonl $D --out results/water_nimble.json 2>&1 | grep -E "^timing|loaded|Traceback|Error" | head -4
echo "=== nimble: hf cold eval"; $N data/hf_eval.jsonl $D --out results/hf_nimble.json 2>&1 | grep -E "^timing|Traceback|Error" | head -3
echo "=== nimble: grid"; $N data/grid_eval.jsonl $D --out results/grid_nimble.json 2>&1 | grep -E "^timing|Traceback|Error" | head -3
echo "=== nimble: cmdiy"; $N data/cmdiy_eval.jsonl $D --out results/cmdiy_nimble.json 2>&1 | grep -E "^timing|Traceback|Error" | head -3
echo "=== PIPELINE20 DONE"
