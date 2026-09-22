"""Is a hosted model's nondeterminism the batch it lands in?

The same request is sent K times one after another (each likely alone in its batch), then
K times at once (likely sharing one). If answers vary more under the second, the variation
is batch composition: bf16 reductions that change with who else is in the batch, the
effect batch_invariant_ops removes. If they vary the same either way, it is something else.

    TYPESAFE_API_KEY=... uv run python scripts/probe_jev_concurrency.py --out results/probe_jev_concurrency.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_permutations import SOURCES, pick

from s1proto.data.teachers import TYPESAFE_URL, normalize, option_keys


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    items = pick(SOURCES, 8)[: args.n]
    key = os.environ.get("TYPESAFE_API_KEY" if "typesafe" in args.url else "S1_API_KEY", "x")
    client = httpx.AsyncClient(timeout=120.0, headers={"Authorization": f"Bearer {key}"})

    async def ask(it):
        body = {"state": it["state"], "model": args.model, "questions": {"q": it["question"]}}
        for attempt in range(5):
            r = await client.post(args.url, json=body)
            if r.status_code in (429, 503, 529):
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if r.status_code != 200:
                return None
            a = r.json()["answers"]["q"]
            keys = option_keys(it["question"])
            return [a["noul"], 1 - a["noul"]] if it["question"]["type"] == "noul" else [normalize(a["probabilities"], keys)[k] for k in keys]
        return None

    def stats(runs):
        flips, tvs, n = 0, [], 0
        for ds in runs:
            ds = [d for d in ds if d]
            if len(ds) < 2:
                continue
            n += 1
            tops = [max(range(len(d)), key=d.__getitem__) for d in ds]
            flips += len(set(tops)) > 1
            tvs += [0.5 * sum(abs(a - b) for a, b in zip(d, ds[0], strict=True)) for d in ds[1:]]
        return {"n": n, "flip": round(flips / max(1, n), 4), "tv": round(sum(tvs) / max(1, len(tvs)), 4), "distinct_answers_mean": None}

    sequential = []
    for it in items:
        sequential.append([await ask(it) for _ in range(args.k)])  # strictly one at a time
    concurrent = []
    for it in items:
        concurrent.append(list(await asyncio.gather(*[ask(it) for _ in range(args.k)])))  # k copies in flight together
    await client.aclose()
    out = {"sequential": stats(sequential), "concurrent": stats(concurrent)}
    for name, r in out.items():
        print(f"  {name:<11} n={r['n']:<4} same-request flip={r['flip']:.3f} tv={r['tv']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "k": args.k, "results": out}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
