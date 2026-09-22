#!/usr/bin/env bash
# circuit-8b v1.2 recipe plus the hard tier (long policy, contracts, multi-hop) at 4,096 tokens.
# Hard-tier JSONL arrives through STAGE (not in the repo). Every step kept and scored on the
# hard-tier eval and unseen sets; the kept step gets the full suite plus the permutation test.
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
N=circuit-8b-hard
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
[ -f data/hard_tier_train.jsonl ] && [ -f data/hard_tier_eval.jsonl ] || { echo "!!! hard tier data not staged"; exit 4; }
cat data/publish_train_v2.jsonl data/hard_tier_train.jsonl > data/hard_mix_train.jsonl
echo "mix rows: $(wc -l < data/hard_mix_train.jsonl)"
echo "=== train $N"
uv run python -u scripts/train_lora.py Qwen/Qwen3-8B-Base data/hard_mix_train.jsonl --out runs/$N --head pointer --epochs 1 --batch 4 --micro 2 --max-length 4096 --eval-every 400 --grad-checkpoint --parallel-options --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+000/|val:|kept step|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }
echo "=== steps $N"
for d in runs/$N/steps/*/; do
  s=$(basename $d); cp runs/$N/config.json $d/config.json
  $E lora:$d data/hard_tier_eval.jsonl --batch 4 --temps "noul=1,choice=1,score=1" --out results/steps_${N}_hard_$s.json 2>&1 | grep -E "Traceback|Error"
  $E lora:$d data/unseen_eval.jsonl --batch 8 --temps "noul=1,choice=1,score=1" --out results/steps_${N}_unseen_$s.json 2>&1 | grep -E "Traceback|Error"
done
echo "=== permutations $N"; uv run python scripts/eval_permutations.py lora:runs/$N --batch 4 --out results/perm_$N.json 2>&1 | grep -E "^  all|Traceback|Error"
for pair in "hard_tier_eval hard" "unseen_eval unseen" "grounded_tools_eval gt" "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
  set -- $pair
  echo "=== $2 $N"; $E lora:runs/$N data/$1.jsonl --batch 4 --out results/${2}_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE34 $N DONE"
