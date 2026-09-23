#!/usr/bin/env bash
# circuit v2.1: the v2 mix + rank and match questions + counterfactual groups, at 4,096
# tokens and 1 epoch, as pipeline35 (v2), so the additions are the only difference.
# BASE / N pick the model (default 1.7B). Staged, not in git:
#   STAGE="data/hard_tier_train.jsonl data/hard_tier_eval.jsonl data/v2_types_train.jsonl data/v2_types_eval.jsonl data/v21_types_train.jsonl data/v21_types_eval.jsonl data/cf_train.jsonl data/cf_eval.jsonl data/inj_eval.jsonl data/unseen_eval.jsonl data/grid_eval.jsonl data/hf_eval.jsonl data/cmdiy_eval.jsonl"
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
BASE=${BASE:-Qwen/Qwen3-1.7B-Base}
N=${N:-circuit-1.7b-v2.1}
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
for f in hard_tier_train hard_tier_eval v2_types_train v2_types_eval v21_types_train v21_types_eval cf_train cf_eval inj_eval unseen_eval grid_eval hf_eval cmdiy_eval grounded_tools_eval water_calls; do
  [ -f data/$f.jsonl ] || { echo "!!! data/$f.jsonl missing; add it to STAGE"; exit 4; }
done
cat data/publish_train_v2.jsonl data/hard_tier_train.jsonl data/v2_types_train.jsonl data/v21_types_train.jsonl data/cf_train.jsonl > data/v21_mix_train.jsonl
echo "mix rows: $(wc -l < data/v21_mix_train.jsonl)"
echo "=== train $N"
uv run python -u scripts/train_lora.py $BASE data/v21_mix_train.jsonl --out runs/$N --head pointer --epochs 1 --batch 4 --micro 2 --max-length 4096 --eval-every 400 --grad-checkpoint --parallel-options --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+000/|val:|kept step|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }
echo "=== permutations $N"; uv run python scripts/eval_permutations.py lora:runs/$N --batch 4 --out results/perm_$N.json 2>&1 | grep -E "^  all|Traceback|Error"
for pair in "v21_types_eval v21" "cf_eval cf" "inj_eval inj" "v2_types_eval v2" "hard_tier_eval hard" "unseen_eval unseen" "grounded_tools_eval gt" "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
  set -- $pair
  echo "=== $2 $N"; $E lora:runs/$N data/$1.jsonl --batch 4 --out results/${2}_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE36 $N DONE"
