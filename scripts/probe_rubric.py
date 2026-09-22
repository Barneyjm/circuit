"""Does the model read the criteria, or the option names?

A choice question gives each option a name and a description. Three versions of the
same item separate what the model is using:

  named        as written: names and descriptions agree
  swapped      names kept, descriptions rotated one option along. The description-reader's
               answer is now a different option from the name-reader's. Whichever the model
               picks says what it read.
  anonymous    names replaced by opt_1..opt_k, descriptions kept. Rubric reading alone.
  names_only   descriptions removed. Name prior alone.

Items are choice questions whose options all carry descriptions: water calls (11-way) and
the DIY tier and tool questions. Works against any /v1/systemone endpoint.

    TYPESAFE_API_KEY=... uv run python scripts/probe_rubric.py --out results/probe_rubric_jev.json
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
from s1proto.data.teachers import TYPESAFE_URL, normalize

SOURCES = ["data/water_calls.jsonl", "data/cmdiy_eval.jsonl"]


def variants(q: dict) -> dict[str, tuple[dict, dict[str, str]]]:
    """Each variant: the question, and a map from the served option key back to the
    original option whose *description* it carries (the rubric truth)."""
    names = list(q["criteria"])
    descs = [q["criteria"][n] for n in names]
    k = len(names)
    out = {}
    out["named"] = ({**q, "criteria": dict(zip(names, descs, strict=True))}, {n: n for n in names})
    rotated = descs[-1:] + descs[:-1]  # option i now carries option i-1's description
    out["swapped"] = ({**q, "criteria": dict(zip(names, rotated, strict=True))}, {names[i]: names[(i - 1) % k] for i in range(k)})
    anon = [f"opt_{i + 1}" for i in range(k)]
    out["anonymous"] = ({**q, "criteria": dict(zip(anon, descs, strict=True))}, dict(zip(anon, names, strict=True)))
    out["names_only"] = ({**q, "criteria": dict.fromkeys(names)}, {n: n for n in names})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    by = defaultdict(list)
    for f in SOURCES:
        for line in open(f):
            it = json.loads(line)
            c = it["question"].get("criteria")
            if it["kind"] == "choice" and isinstance(c, dict) and len(c) >= 3 and all(isinstance(v, str) and v for v in c.values()):
                by[it["family"]].append(it)
    items = [it for fam in sorted(by) for it in by[fam][: args.per_family]]
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
        keys = list(q["criteria"])
        return normalize(r.json()["answers"]["q"]["probabilities"], keys)

    async def probe(it):
        res = {}
        for name, (q, back) in variants(it["question"]).items():
            d = await ask(it["state"], q)
            res[name] = (d, back)
        return res

    results = await asyncio.gather(*[probe(it) for it in items])
    await client.aclose()

    tally = defaultdict(lambda: defaultdict(lambda: {"n": 0, "by_description": 0, "by_name": 0, "neither": 0, "conf": 0.0}))
    for it, res in zip(items, results, strict=True):
        truth = max(it["ref"], key=it["ref"].get)
        for variant, (d, back) in res.items():
            if not d:
                continue
            top = max(d, key=d.get)
            t = tally[it["family"]][variant]
            t["n"] += 1
            t["conf"] += d[top]
            if back[top] == truth:
                t["by_description"] += 1  # picked the option carrying the true description
            elif top == truth:
                t["by_name"] += 1  # picked the option carrying the true name (only differs when swapped)
            else:
                t["neither"] += 1
    report = {}
    for fam, vs in tally.items():
        report[fam] = {}
        for variant, t in vs.items():
            n = t["n"]
            report[fam][variant] = {
                "n": n,
                "by_description": round(t["by_description"] / n, 3),
                "by_name": round(t["by_name"] / n, 3),
                "neither": round(t["neither"] / n, 3),
                "mean_conf": round(t["conf"] / n, 3),
            }
    for fam, vs in report.items():
        print(f"  {fam}")
        for variant, r in vs.items():
            print(
                f"    {variant:<11} n={r['n']:<4} follows description {r['by_description']:.3f}  follows name {r['by_name']:.3f}  neither {r['neither']:.3f}  conf {r['mean_conf']:.3f}"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "results": report}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
