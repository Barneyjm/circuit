#!/usr/bin/env bash
# Pod job: a pointer head on frozen Qwen3-0.6B, masked-diffusion vs causal, same rows, then the
# open-taxonomy evals. Runs in diffusion/'s environment (transformers 4.57) for both, so the only
# difference is the model. A 40-row smoke check first: a crash costs cents, not the run.
set -u
cd /workspace/s1-proto
export UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf
for f in v22_head_mix_train open_tax_eval open_tax_tagger_eval; do
  [ -f data/$f.jsonl ] || { echo "!!! data/$f.jsonl missing; add it to STAGE"; exit 4; }
done
echo "=== env"
(cd diffusion && uv sync -q) || { echo "!!! diffusion env did not install"; exit 3; }
R="uv run --project diffusion --no-sync"
cuda_ok () { $R python -c "import torch, transformers, sys; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); sys.exit(0 if torch.cuda.is_available() else 1)"; }
if ! cuda_ok; then  # the lock's torch targets a newer CUDA than this host's driver
  uv pip install -q --python diffusion/.venv/bin/python --reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision  # together, or torchvision's ops do not load
  cuda_ok || { echo "!!! no usable CUDA torch"; exit 3; }
fi
# The dLLM model files import `dllm` in their __main__ demo only, but transformers checks every
# import in remote code before loading it; an empty module satisfies the check.
$R python -c "import site, pathlib; d = pathlib.Path(site.getsitepackages()[0]) / 'dllm'; d.mkdir(exist_ok=True); (d / '__init__.py').touch()"
COMMON="data/v22_head_mix_train.jsonl --freeze-adapter --head pointer --epochs 1 --batch 8 --head-lr 5e-4 --max-length 2048 --val-frac 0.03"
echo "=== smoke"
$R python -u scripts/train_lora.py dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 $COMMON --masked --limit 40 --eval-every 1000 --out runs/smoke > results/smoke_d1.log 2>&1 \
  || { echo "!!! smoke failed"; tail -25 results/smoke_d1.log; exit 5; }
echo "smoke ok: $(grep -m1 'ep0 step' results/smoke_d1.log || tail -1 results/smoke_d1.log)"
for pair in "circuit-0.6b-dlm-head dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --masked" "circuit-0.6b-causal-head Qwen/Qwen3-0.6B --parallel-options"; do
  set -- $pair
  echo "=== train $1"
  $R python -u scripts/train_lora.py $2 $COMMON $3 --eval-every 800 --out runs/$1 > results/train_$1.log 2>&1 || { echo "!!! $1 failed"; tail -20 results/train_$1.log; exit 2; }
  grep -E "kept step|done in" results/train_$1.log | tail -2
  echo "=== eval $1"
  $R python scripts/eval_open_taxonomy.py lora:runs/$1 data/open_tax_eval.jsonl data/open_tax_tagger_eval.jsonl --batch 8 --out results/open_tax_$1.json 2>&1 | grep -E "^all|^tagger|^mmlu|^no_robots|^dbpedia|^dolly|Traceback|Error"
done
echo "=== PIPELINE38 d1 DONE"
