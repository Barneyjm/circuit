"""Two things a hosted decision model is assumed to do, measured.

    TYPESAFE_API_KEY=... uv run python scripts/probe_jev.py --out results/probe_jev.json

repeat     the same request sent K times. A deterministic model returns the same
           probabilities every time; the flip rate here is a floor under any order or
           wording test, because it is what you get changing nothing.
sibling    a question asked alone, then with an unrelated second question in the same
           request. If questions are isolated branches over a shared state, the first
           answer does not move.

Items are the same choice questions the permutation test uses plus noul items from the
unseen sets, one order each. Works against any /v1/systemone endpoint (--url), so the
same probe runs against our own API for comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_permutations import SOURCES, pick

from s1proto.data.teachers import TYPESAFE_URL, normalize, option_keys

SIBLING = {"type": "noul", "instructions": "Does the state contain at least one number written in digits?"}


def parse(answer: dict, q: dict) -> dict[str, float]:
    if q["type"] == "noul":
        p = float(answer["noul"])
        return {"yes": p, "no": 1.0 - p}
    return normalize(answer["probabilities"], option_keys(q))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--key", default=None, help="bearer; default TYPESAFE_API_KEY, or S1_API_KEY when --url is ours")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--headers", default=None, help='extra request headers as JSON, e.g. {"x-circuit-reproducible":"1"}')
    args = ap.parse_args()

    choice = pick(SOURCES, 25)
    noul = [json.loads(line) for line in open("data/unseen_eval.jsonl")]
    noul = [it for it in noul if it["kind"] == "noul"][:: max(1, len(noul) // 150)][:150]
    items = (choice + noul)[: args.n]
    key = args.key or os.environ.get("TYPESAFE_API_KEY" if "typesafe" in args.url else "S1_API_KEY", "x")
    headers = {"Authorization": f"Bearer {key}", **(json.loads(args.headers) if args.headers else {})}
    client = httpx.AsyncClient(timeout=120.0, headers=headers)
    gate = asyncio.Semaphore(args.concurrency)

    async def ask(state, questions: dict) -> dict | None:
        body = {"state": state, "model": args.model, "questions": questions}
        async with gate:
            for attempt in range(5):
                try:
                    r = await client.post(args.url, json=body)
                except httpx.HTTPError:
                    await asyncio.sleep(2**attempt)
                    continue
                if r.status_code in (429, 503, 529):
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                if r.status_code != 200:
                    print("error", r.status_code, r.text[:120], file=sys.stderr)
                    return None
                return r.json()["answers"]
        return None

    async def probe(it):
        q = it["question"]
        alone = [await ask(it["state"], {"q": q}) for _ in range(args.repeats)]
        together = await ask(it["state"], {"q": q, "other": SIBLING})
        return alone, together

    results = await asyncio.gather(*[probe(it) for it in items])
    await client.aclose()

    per_family: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "repeat_flip": 0, "repeat_tv": [], "sibling_flip": 0, "sibling_tv": [], "zero_true": 0, "two_decimals": 0, "values": 0}
    )
    dump = []
    for it, (alone, together) in zip(items, results, strict=True):
        q = it["question"]
        dists = [parse(a["q"], q) for a in alone if a]
        if len(dists) < 2:
            continue
        keys = option_keys(q)
        fam = it["family"].split("/")[0]
        s = per_family[fam]
        s["n"] += 1
        tops = [max(keys, key=d.get) for d in dists]
        s["repeat_flip"] += len(set(tops)) > 1
        s["repeat_tv"] += [0.5 * sum(abs(d[k] - dists[0][k]) for k in keys) for d in dists[1:]]
        truth = max(keys, key=lambda k: it["ref"][k])
        s["zero_true"] += dists[0][truth] == 0.0
        for d in dists:
            for v in d.values():
                s["values"] += 1
                s["two_decimals"] += abs(v * 100 - round(v * 100)) < 1e-6
        if together and "q" in together:
            dt = parse(together["q"], q)
            s["sibling_flip"] += max(keys, key=dt.get) != tops[0]
            s["sibling_tv"].append(0.5 * sum(abs(dt[k] - dists[0][k]) for k in keys))
        dump.append(
            {
                "id": it["id"],
                "family": fam,
                "alone": [[round(d[k], 4) for k in keys] for d in dists],
                "with_sibling": [round(dt[k], 4) for k in keys] if together and "q" in together else None,
                "keys": keys,
            }
        )

    def fin(s):
        return {
            "n": s["n"],
            "repeat_flip": round(s["repeat_flip"] / s["n"], 4),
            "repeat_tv": round(sum(s["repeat_tv"]) / max(1, len(s["repeat_tv"])), 4),
            "sibling_flip": round(s["sibling_flip"] / max(1, len(s["sibling_tv"])), 4),
            "sibling_tv": round(sum(s["sibling_tv"]) / max(1, len(s["sibling_tv"])), 4),
            "true_label_reported_as_zero": round(s["zero_true"] / s["n"], 4),
            "values_on_two_decimals": round(s["two_decimals"] / max(1, s["values"]), 4),
        }

    allstats = {"n": 0, "repeat_flip": 0, "repeat_tv": [], "sibling_flip": 0, "sibling_tv": [], "zero_true": 0, "two_decimals": 0, "values": 0}
    for s in per_family.values():
        for k, v in list(allstats.items()):
            allstats[k] = v + s[k]
    report = {"all": fin(allstats), **{f: fin(s) for f, s in sorted(per_family.items())}}
    for name, r in report.items():
        print(
            f"  {name:<18} n={r['n']:<4} same-request flip={r['repeat_flip']:.3f} tv={r['repeat_tv']:.3f} | +sibling flip={r['sibling_flip']:.3f} tv={r['sibling_tv']:.3f} | truth=0.00 {r['true_label_reported_as_zero']:.3f} | 2dp {r['values_on_two_decimals']:.2f}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "repeats": args.repeats, "results": report, "items": dump}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
