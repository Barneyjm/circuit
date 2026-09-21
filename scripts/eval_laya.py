"""Run Laya (convaiinnovations/laya, Apache-2.0) on an eval JSONL with our metrics.

    uv run --with laya python scripts/eval_laya.py data/unseen_eval.jsonl --out results/unseen_laya.json

Laya takes the same state-and-typed-questions request we do, so items go in unchanged.
It reads 512 tokens and gives the option list 192 of them; a question whose options do
not fit raises, and is scored as a uniform guess and counted under `could_not_run`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_set import summarize

from s1proto.data.teachers import option_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import laya

    agent = laya.load(args.model, device=args.device)
    items = [json.loads(line) for line in open(args.data)][: args.limit]
    preds, lat, failed = [], [], 0
    for it in items:
        keys = option_keys(it["question"])
        t0 = time.perf_counter()
        try:
            a = agent.system_one(it["state"], {"q": it["question"]})["answers"]["q"]
        except ValueError:
            failed += 1
            preds.append([1.0 / len(keys)] * len(keys))
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        if it["kind"] == "noul":
            preds.append([a["noul"], 1.0 - a["noul"]])
        elif it["kind"] == "score":
            preds.append([a["probabilities"][str(i)] for i in range(len(keys))])
        else:
            preds.append([a["probabilities"][k] for k in keys])

    lat.sort()
    timing = {
        "device": str(agent.device),
        "items": len(items),
        "could_not_run": failed,
        "ms_per_item_p50": round(lat[len(lat) // 2], 1) if lat else None,
        "batch": 1,
    }
    summary = summarize(items, preds)
    print("timing:", json.dumps(timing))
    for k, v in summary.items():
        if k == "all" or k.startswith("family:"):
            print(
                f"  {k.replace('family:', '').replace(' (heldout)', ''):<18} n={v['n']:<5} acc={v['accuracy']:.3f} ece={v['ece']:.3f} kl={v['kl_to_ref']:.3f}"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "data": args.data, "timing": timing, "metrics": summary}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
