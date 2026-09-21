"""Run SemIf (TheoLeeCJ/SemIf, MIT) on an eval JSONL with our metrics.

SemIf is not a trained model: it reads the answer-letter logits of a stock instruct
model, by default Qwen/Qwen3.5-4B, which LangSmith hosts as `semif-qwen3.5-4b`. Install
it in its own environment (it pins torch and, on a Mac, MLX) and point this at its CLI:

    git clone https://github.com/TheoLeeCJ/SemIf && cd SemIf
    python3 -m venv .venv && .venv/bin/pip install -e '.[mlx]'       # drop [mlx] on CUDA
    uv run python scripts/eval_semif.py data/unseen_eval.jsonl --semif SemIf/.venv/bin/semif-score \\
        --backend mlx --out results/unseen_semif.json

Its answers are letters A to P, so a question with more than 16 options cannot be asked.
Those are scored as a uniform guess and counted under `could_not_run`, as are rows its
own validation refuses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_set import summarize

from s1proto.data.teachers import option_keys

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # the Qwen/Qwen3.5-4B commit SemIf's docs pin
MAX_OPTIONS = 16


def to_row(it: dict) -> dict:
    q = it["question"]
    crit = q.get("criteria")
    if q["type"] == "noul":
        crit = crit or {}
        options = [{"id": "yes", "description": crit.get("true") or "Yes"}, {"id": "no", "description": crit.get("false") or "No"}]
    elif q["type"] == "choice":
        options = [{"id": k, "description": f"{k}: {v}" if v else k} for k, v in crit.items()]
    else:
        options = [{"id": str(i), "description": level} for i, level in enumerate(crit)]
    return {"id": it["id"], "state": it["state"], "question": q["instructions"], "options": options}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--semif", required=True, help="path to the semif-score executable")
    ap.add_argument("--backend", default=None, help="mlx on Apple silicon; omit for CUDA")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.data)][: args.limit]
    askable = [it for it in items if len(option_keys(it["question"])) <= MAX_OPTIONS]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        src.write_text("".join(json.dumps(to_row(it)) + "\n" for it in askable))
        cmd = [args.semif, "--mode", "direct", "--model", args.model, "--revision", args.revision, "--input", str(src), "--output", str(dst)]
        if args.backend:
            cmd += ["--backend", args.backend]
        subprocess.run(cmd, check=True)
        got = {r["id"]: r for r in map(json.loads, dst.read_text().splitlines())}

    preds, lat = [], []
    for it in items:
        keys = option_keys(it["question"])
        r = got.get(it["id"])
        if r is None:
            preds.append([1.0 / len(keys)] * len(keys))
            continue
        by_id = dict(zip(r["option_ids"], r["probabilities"], strict=True))
        preds.append([by_id[k] for k in keys])
        lat.append(r["total_seconds"] * 1000)

    lat.sort()
    timing = {
        "items": len(items),
        "could_not_run": len(items) - len(got),
        "ms_per_item_p50": round(lat[len(lat) // 2], 1) if lat else None,
        "batch": 1,
        "backend": args.backend or "cuda",
    }
    summary = summarize(items, preds)
    print("timing:", json.dumps(timing))
    for k, v in summary.items():
        if k == "all" or (k.startswith("family:") and "/" not in k):
            print(
                f"  {k.replace('family:', '').replace(' (heldout)', ''):<18} n={v['n']:<5} acc={v['accuracy']:.3f} ece={v['ece']:.3f} kl={v['kl_to_ref']:.3f}"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": f"semif:{args.model}@{args.revision[:8]}", "data": args.data, "timing": timing, "metrics": summary}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
