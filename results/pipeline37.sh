#!/usr/bin/env bash
# circuit v2.2 (1.7B): v2.0 continued on open-taxonomy choice rows (user-defined option lists with
# descriptions and an "other" that is right only when the answer is missing), plus a replay
# slice of the v2 mix and v2.1's counterfactual and injection rows so nothing regresses.
# Starts from v2.0's adapter and head (--init), fresh optimizer, 1 epoch, half v2's LR.
# Staged, not in git:
#   STAGE="data/open_tax_train.jsonl data/open_tax_eval.jsonl data/open_tax_tagger_eval.jsonl data/hard_tier_train.jsonl data/hard_tier_eval.jsonl data/v2_types_train.jsonl data/v2_types_eval.jsonl data/cf_train.jsonl data/cf_eval.jsonl data/inj_train.jsonl data/inj_eval.jsonl data/unseen_eval.jsonl data/grid_eval.jsonl data/hf_eval.jsonl data/cmdiy_eval.jsonl"
set -u
cd /workspace/s1-proto
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
BASE=${BASE:-Qwen/Qwen3-1.7B-Base}
N=${N:-circuit-1.7b-v2.2}
REPLAY=${REPLAY:-12000}
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! no GPU; aborting"; exit 3; }
for f in open_tax_train open_tax_eval open_tax_tagger_eval hard_tier_train hard_tier_eval v2_types_train v2_types_eval cf_train cf_eval inj_train inj_eval unseen_eval grid_eval hf_eval cmdiy_eval grounded_tools_eval water_calls; do
  [ -f data/$f.jsonl ] || { echo "!!! data/$f.jsonl missing; add it to STAGE"; exit 4; }
done

echo "=== fetch v2.0"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('jbarney/circuit-1.7b', revision='v2.0', local_dir='runs/circuit-1.7b-v2.0')" || { echo "!!! could not fetch v2.0"; exit 5; }
[ -s runs/circuit-1.7b-v2.0/head.pt ] || { echo "!!! v2.0 has no head.pt"; exit 5; }

# The replay slice: v2.0's own question kinds only, so its head loads as it is.
uv run python - "$REPLAY" <<'PY' > data/v22_mix_train.jsonl
import json, random, sys
kinds = set(json.load(open("runs/circuit-1.7b-v2.0/config.json"))["question_types"])
old = [l for f in ("publish_train_v2", "hard_tier_train", "v2_types_train", "cf_train", "inj_train") for l in open(f"data/{f}.jsonl") if json.loads(l)["question"]["type"] in kinds]
rows = open("data/open_tax_train.jsonl").readlines() + random.Random(0).sample(old, min(int(sys.argv[1]), len(old)))
random.Random(1).shuffle(rows)
sys.stdout.writelines(rows)
PY
echo "mix rows: $(wc -l < data/v22_mix_train.jsonl)"

echo "=== train $N"
uv run python -u scripts/train_lora.py $BASE data/v22_mix_train.jsonl --out runs/$N --init runs/circuit-1.7b-v2.0 --head pointer --epochs 1 --batch 4 --micro 2 --lr 5e-5 --head-lr 5e-4 --max-length 4096 --eval-every 400 --grad-checkpoint --parallel-options --wandb s1proto --run-name $N > results/train_$N.log 2>&1 &
TRAIN=$!
# Training has to be visibly stepping within 3 minutes of the model loading, or the pod is not worth paying for.
for _ in $(seq 36); do grep -q "ep0 step" results/train_$N.log && break; kill -0 $TRAIN 2>/dev/null || break; sleep 5; done
grep -q "ep0 step" results/train_$N.log || { echo "!!! no training step within 3 min"; tail -n 20 results/train_$N.log; kill $TRAIN; exit 6; }
grep -m1 "ep0 step" results/train_$N.log
tail -n +1 -f results/train_$N.log --pid=$TRAIN | grep --line-buffered -E "train=|initialised|step [0-9]+000/|val:|kept step|done in|View run|Traceback|Error"
wait $TRAIN
[ -f runs/$N/head.pt ] || { echo "!!! $N did not train"; exit 2; }

echo "=== open taxonomy $N"; uv run python scripts/eval_open_taxonomy.py lora:runs/$N data/open_tax_eval.jsonl data/open_tax_tagger_eval.jsonl --batch 8 --out results/open_tax_$N.json 2>&1 | grep -E "^all|^tagger|^mmlu|^no_robots|^dbpedia|^dolly|Traceback|Error"
echo "=== permutations $N"; uv run python scripts/eval_permutations.py lora:runs/$N --batch 4 --out results/perm_$N.json 2>&1 | grep -E "^  all|Traceback|Error"
for pair in "cf_eval cf" "inj_eval inj" "v2_types_eval v2" "hard_tier_eval hard" "unseen_eval unseen" "grounded_tools_eval gt" "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
  set -- $pair
  echo "=== $2 $N"; $E lora:runs/$N data/$1.jsonl --batch 4 --out results/${2}_$N.json 2>&1 | grep -E "^timing|Traceback|Error"
done
echo "=== PIPELINE37 $N DONE"
