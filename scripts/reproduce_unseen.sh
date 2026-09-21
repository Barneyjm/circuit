#!/usr/bin/env bash
# Rebuild the unseen-dataset benchmark and score every model on it. See REPRODUCE.md.
#   scripts/reproduce_unseen.sh                 # open models only, no keys needed
#   TYPESAFE_API_KEY=... scripts/reproduce_unseen.sh   # also scores Jev (about a minute, a few cents)
#   MODELS="circuit-1.7b" scripts/reproduce_unseen.sh  # a subset; the 8B wants ~20 GB of GPU or unified memory
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-circuit-1.7b circuit-8b}"
OUT="${OUT:-results/repro}"
mkdir -p "$OUT" runs/hub

uv sync -q
[ -f data/unseen_eval.jsonl ] || uv run python scripts/build_unseen_eval.py --n 300   # seed is fixed: same 1,200 items for everyone

for m in $MODELS; do
  [ -f "runs/hub/$m/head.pt" ] || uv run hf download "jbarney/$m" --local-dir "runs/hub/$m" >/dev/null
  uv run python scripts/eval_set.py "lora:runs/hub/$m" data/unseen_eval.jsonl --batch 4 --out "$OUT/unseen_$m.json" >/dev/null
done
uv run --with laya python scripts/eval_laya.py data/unseen_eval.jsonl --out "$OUT/unseen_laya.json" >/dev/null
if [ -n "${TYPESAFE_API_KEY:-}" ]; then
  uv run python scripts/eval_jev.py data/unseen_eval.jsonl --out "$OUT/unseen_jev.json" --concurrency 4 >/dev/null
fi

uv run python - "$OUT" <<'PY'
import glob, json, sys
fams = ["bfcl_relevance", "halueval_qa", "chaosnli", "hwu64"]
print(f"\n{'model':16s}" + "".join(f"{f:>26s}" for f in fams) + "\n" + " " * 16 + "".join(f"{'acc / ece / kl':>26s}" for _ in fams))
for path in sorted(glob.glob(f"{sys.argv[1]}/unseen_*.json")):
    m = json.load(open(path))["metrics"]
    cells = [m.get(f"family:{f} (heldout)") for f in fams]
    print(f"{path.split('unseen_')[1][:-5]:16s}" + "".join(f"{(f'{c['accuracy']:.3f} / {c['ece']:.3f} / {c['kl_to_ref']:.2f}' if c else '-'):>26s}" for c in cells))
PY
