#!/usr/bin/env bash
# circuit-8b with choice options encoded side by side, otherwise the v1.1 recipe. Every step
# kept and scored on the unseen sets and a 501-item permutation test; the kept step gets the
# full suite.
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
N=circuit-8b-par
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
echo "=== train $N"
uv run python -u scripts/train_lora.py Qwen/Qwen3-8B-Base data/publish_train_v2.jsonl --out runs/$N --head pointer --epochs 1 --batch 4 --max-length 1024 --eval-every 400 --grad-checkpoint --parallel-options --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+000/|val:|kept step|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }
echo "=== steps $N"
for d in runs/$N/steps/*/; do
  s=$(basename $d); cp runs/$N/config.json $d/config.json
  $E lora:$d data/unseen_eval.jsonl --batch 8 --temps "noul=1,choice=1,score=1" --out results/steps_${N}_unseen_$s.json 2>&1 | grep -E "Traceback|Error"
  uv run python scripts/eval_permutations.py lora:$d --batch 4 --per-family 40 --out results/steps_${N}_perm_$s.json 2>&1 | grep -E "^  all|Traceback|Error" | sed "s/^/  step $s /"
done
echo "=== permutations $N"; uv run python scripts/eval_permutations.py lora:runs/$N --batch 4 --out results/perm_$N.json 2>&1 | grep -E "^  all|Traceback|Error"
for pair in "unseen_eval unseen" "grounded_tools_eval gt" "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
  set -- $pair
  echo "=== $2 $N"; $E lora:runs/$N data/$1.jsonl --batch 8 --out results/${2}_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE33 $N DONE"
