#!/usr/bin/env bash
# Base-model comparison: the publish recipe on Qwen3.5 (2B, 9B) next to the shipped
# Qwen3 circuits (1.7B, 8B). Same data, same head, same evals; only the base changes.
# The shipped adapters are re-timed on this box so latency is like for like.
set -u
cd /workspace/s1-proto
# UV_NO_SYNC: a sync would put the lockfile's torch back, which is built for a newer CUDA than the driver.
export UV_NO_SYNC=1 UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_set.py"
need_gpu () { uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "!!! GPU LOST at $1; aborting"; exit 3; }; }
run_evals () {  # $1 = run dir, $2 = tag, $3 = batch
  for pair in "grid_eval grid" "hf_eval hf" "water_calls water" "cmdiy_eval cmdiy"; do
    set -- "$1" "$2" "$3" $pair
    need_gpu "$5 $2"
    echo "=== $5 $2"; $E lora:$1 data/$4.jsonl --batch $3 --out results/${5}_$2.json 2>&1 | grep -E "^timing|^overall|Traceback|Error"
  done
  echo "=== timing $2"; $E lora:$1 data/hf_eval_timing.jsonl --batch 1 --out results/timing26_$2.json 2>&1 | grep -E "^timing|Traceback|Error"
}
need_gpu "start"

# usage: pipeline26.sh 2b | 9b | timing   (2b and 9b share one 80 GB card; run them side by side)
# --micro 1: Qwen3.5's backward pass NaNs on a left-padded batch, so rows go through unpadded and accumulate.
case "${1:?2b|9b|timing}" in
  2b) N=circuit35-2b; M=Qwen/Qwen3.5-2B-Base; X="--epochs 2"; B=16 ;;
  9b) N=circuit35-9b; M=Qwen/Qwen3.5-9B-Base; X="--epochs 1 --grad-checkpoint"; B=8 ;;
  timing)
    for r in circuit-1.7b circuit-8b; do
      echo "=== timing $r (same box)"; $E lora:runs/$r data/hf_eval_timing.jsonl --batch 1 --out results/timing26_$r.json 2>&1 | grep -E "^timing|Traceback|Error"
    done; echo "=== PIPELINE26 timing DONE"; exit 0 ;;
esac
echo "=== train $N"
uv run python -u scripts/train_lora.py $M data/publish_train.jsonl --out runs/$N --head pointer $X --batch 4 --micro 1 --max-length 1024 --eval-every 400 --wandb s1proto --run-name $N 2>&1 | grep --line-buffered -E "train=|step [0-9]+00/|val:|saved|done in|View run|Traceback|Error"
[ -f runs/$N/head.pt ] && run_evals runs/$N $N $B || echo "!!! $N did not train"
echo "=== PIPELINE26 $1 DONE"
