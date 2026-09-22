"""The same question with more and more options: where does a model change gear?

CLINC items have 151 intents. Each is asked with the true intent plus 3, 7, 15, 25, 31, 63
and 150 distractors, the same item at every size. Accuracy, confidence and latency by size.
A model that switches to a two-stage path above some option count shows it here as a
step in latency, and often in calibration.

    TYPESAFE_API_KEY=... uv run python scripts/probe_option_count.py --out results/probe_option_count_jev.json
    uv run python scripts/probe_option_count.py --url http://localhost:8903/v1/systemone --model circuit-1.7b --out results/probe_option_count_circuit-1.7b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1proto.data.teachers import TYPESAFE_URL, normalize

SIZES = [4, 8, 16, 26, 32, 64, 151]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--headers", default=None, help="extra request headers as JSON")
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    items = [json.loads(line) for line in open("data/hf_eval.jsonl")]
    items = [it for it in items if it["family"] == "clinc_intent"][: args.n]
    key = args.key or os.environ.get("TYPESAFE_API_KEY" if "typesafe" in args.url else "S1_API_KEY", "x")
    client = httpx.AsyncClient(timeout=180.0, headers={"Authorization": f"Bearer {key}", **(json.loads(args.headers) if args.headers else {})})
    gate = asyncio.Semaphore(args.concurrency)
    rng = random.Random(5)

    async def ask(it, size):
        q = it["question"]
        names = list(q["criteria"])
        truth = max(names, key=lambda k: it["ref"][k])
        others = [k for k in names if k != truth]
        rng.shuffle(others)
        chosen = [truth] + others[: size - 1]
        rng.shuffle(chosen)
        body = {"state": it["state"], "model": args.model, "questions": {"q": {**q, "criteria": {k: q["criteria"][k] for k in chosen}}}}
        async with gate:
            t0 = time.perf_counter()
            for attempt in range(5):
                r = await client.post(args.url, json=body)
                if r.status_code in (429, 503, 529):
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                break
            ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return None
        d = normalize(r.json()["answers"]["q"]["probabilities"], chosen)
        top = max(chosen, key=d.get)
        return {"correct": top == truth, "conf": d[top], "p_truth": d[truth], "ms": ms}

    out = {}
    for size in SIZES:
        rows = [x for x in await asyncio.gather(*[ask(it, size) for it in items]) if x]
        n = len(rows)
        lat = sorted(x["ms"] for x in rows)
        out[size] = {
            "n": n,
            "accuracy": round(sum(x["correct"] for x in rows) / n, 4),
            "mean_conf": round(sum(x["conf"] for x in rows) / n, 4),
            "mean_p_truth": round(sum(x["p_truth"] for x in rows) / n, 4),
            "ms_p50": round(lat[n // 2], 1),
        }
        r = out[size]
        print(
            f"  {size:>3} options  n={n:<3} acc={r['accuracy']:.3f} conf={r['mean_conf']:.3f} p(truth)={r['mean_p_truth']:.3f} p50={r['ms_p50']:.0f} ms",
            flush=True,
        )
    await client.aclose()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "results": out}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
