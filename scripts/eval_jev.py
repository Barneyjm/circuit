"""Run Jev (TypeSafe) on an eval JSONL and score it with the same metrics
as our models, plus per-call latency measured client-side.

    TYPESAFE_API_KEY=... uv run python scripts/eval_jev.py data/hf_eval.jsonl --out results/hf_jev.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.teachers import JevTeacher, option_keys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_set import summarize


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--url", default=None, help="a System One server other than TypeSafe, e.g. http://localhost:8009/v1/systemone")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.data)]
    if args.limit:
        items = items[: args.limit]
    # Teacher semaphore wide open; our own gate below so the timer starts
    # when the request is actually issued, not while queued.
    jev = JevTeacher(model=args.model, concurrency=10_000, **({"url": args.url} if args.url else {}))
    gate = asyncio.Semaphore(args.concurrency)
    lat: list[float] = []
    preds: list[list[float] | None] = [None] * len(items)
    errors = 0

    async def one(i: int) -> None:
        nonlocal errors
        it = items[i]
        keys = option_keys(it["question"])
        async with gate:
            t0 = time.perf_counter()
            try:
                d = await jev.distribution(it["state"], it["question"])
            except Exception as e:
                errors += 1
                print("error", it["id"], str(e)[:120], file=sys.stderr)
                preds[i] = [1.0 / len(keys)] * len(keys)
                return
            lat.append((time.perf_counter() - t0) * 1000)
        preds[i] = [d[k] for k in keys]

    t0 = time.perf_counter()
    await asyncio.gather(*[one(i) for i in range(len(items))])
    wall = time.perf_counter() - t0
    await jev.aclose()

    summary = summarize(items, preds)  # type: ignore[arg-type]
    lat_sorted = sorted(lat)
    timing = {
        "items": len(items),
        "errors": errors,
        "concurrency": args.concurrency,
        "wall_seconds": round(wall, 1),
        "ms_per_item_p50": round(lat_sorted[len(lat_sorted) // 2], 1) if lat_sorted else None,
        "ms_per_item_p90": round(lat_sorted[int(len(lat_sorted) * 0.9)], 1) if lat_sorted else None,
        "ms_per_item_mean": round(sum(lat) / len(lat), 1) if lat else None,
        "note": f"client-side round trip to {args.url or 'api.typesafe.ai'}" + ("; includes network" if not args.url else ""),
    }
    result = {"model": args.model, "data": args.data, "timing": timing, "metrics": summary}
    print("timing:", json.dumps(timing))
    for k, v in summary.items():
        if k.startswith("family:"):
            print(f"  {k[7:].replace(' (heldout)', ''):<18} acc={v['accuracy']:.3f} ece={v['ece']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
