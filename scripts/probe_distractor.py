"""Does text that has nothing to do with the question move the answer?

Each item is asked as written, then with a paragraph of unrelated prose appended to the
state, then prepended. The paragraph never mentions anything an option describes. A model
that reads for the question ignores it; the flip rate and probability shift say how much
it does not.

    TYPESAFE_API_KEY=... uv run python scripts/probe_distractor.py --out results/probe_distractor_jev.json
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

FILLER = (
    "The regional library has extended its weekend opening hours through the autumn, and the "
    "reading room on the second floor now stays open until six. Volunteers repainted the "
    "community garden fence in a pale green over two Saturdays, and the tomato beds along the "
    "south wall did well this year despite a dry August. A local walking group meets at the "
    "old station car park on the first Sunday of each month for a route of about seven miles."
)


def with_filler(state, where: str):
    if isinstance(state, str):
        return state + "\n\n" + FILLER if where == "after" else FILLER + "\n\n" + state
    if isinstance(state, dict):
        return {**state, "note": FILLER} if where == "after" else {"note": FILLER, **state}
    return state + [{"note": FILLER}] if where == "after" else [{"note": FILLER}] + state


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    items = pick(SOURCES, args.per_family)
    key = os.environ.get("TYPESAFE_API_KEY" if "typesafe" in args.url else "S1_API_KEY", "x")
    client = httpx.AsyncClient(timeout=120.0, headers={"Authorization": f"Bearer {key}"})
    gate = asyncio.Semaphore(args.concurrency)

    async def ask(state, q):
        body = {"state": state, "model": args.model, "questions": {"q": q}}
        async with gate:
            for attempt in range(5):
                r = await client.post(args.url, json=body)
                if r.status_code in (429, 503, 529):
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                break
        if r.status_code != 200:
            return None
        keys = option_keys(q)
        return [normalize(r.json()["answers"]["q"]["probabilities"], keys)[k] for k in keys]

    async def probe(it):
        q = it["question"]
        return [await ask(it["state"], q), await ask(with_filler(it["state"], "after"), q), await ask(with_filler(it["state"], "before"), q)]

    results = await asyncio.gather(*[probe(it) for it in items])
    await client.aclose()
    tally = defaultdict(lambda: {"n": 0, "flip_after": 0, "flip_before": 0, "tv_after": [], "tv_before": [], "acc": [0, 0, 0]})
    for it, (a, b, c) in zip(items, results, strict=True):
        if not (a and b and c):
            continue
        keys = option_keys(it["question"])
        truth = keys.index(max(keys, key=lambda k: it["ref"][k]))
        t = tally[it["family"]]
        t["n"] += 1
        ta, tb, tc = (max(range(len(d)), key=d.__getitem__) for d in (a, b, c))
        t["flip_after"] += ta != tb
        t["flip_before"] += ta != tc
        t["tv_after"].append(0.5 * sum(abs(x - y) for x, y in zip(a, b, strict=True)))
        t["tv_before"].append(0.5 * sum(abs(x - y) for x, y in zip(a, c, strict=True)))
        for j, tt in enumerate((ta, tb, tc)):
            t["acc"][j] += tt == truth
    tot = {"n": 0, "flip_after": 0, "flip_before": 0, "tv_after": [], "tv_before": [], "acc": [0, 0, 0]}
    for t in tally.values():
        for k in ("n", "flip_after", "flip_before"):
            tot[k] += t[k]
        tot["tv_after"] += t["tv_after"]
        tot["tv_before"] += t["tv_before"]
        tot["acc"] = [x + y for x, y in zip(tot["acc"], t["acc"], strict=True)]

    def fin(t):
        n = t["n"]
        return {
            "n": n,
            "flip_after": round(t["flip_after"] / n, 3),
            "flip_before": round(t["flip_before"] / n, 3),
            "tv_after": round(sum(t["tv_after"]) / n, 3),
            "tv_before": round(sum(t["tv_before"]) / n, 3),
            "acc": [round(x / n, 3) for x in t["acc"]],
        }

    report = {"all": fin(tot), **{f: fin(t) for f, t in sorted(tally.items()) if t["n"]}}
    for name, r in report.items():
        print(
            f"  {name:<24} n={r['n']:<4} filler after: flip={r['flip_after']:.3f} tv={r['tv_after']:.3f} | before: flip={r['flip_before']:.3f} tv={r['tv_before']:.3f} | acc {r['acc'][0]:.3f}/{r['acc'][1]:.3f}/{r['acc'][2]:.3f}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "results": report}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
