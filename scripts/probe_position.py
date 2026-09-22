"""Where in a long state the evidence sits: start, middle or end.

Each water-utility call transcript (the whole evidence for an 11-way classification) is
placed among five paragraphs of unrelated prose at the start, in the middle, or at the
end, so the state is about 900 tokens with the answer in one place. Accuracy and mean
confidence by position. A model that reads the whole state scores the same everywhere.

    TYPESAFE_API_KEY=... uv run python scripts/probe_position.py --out results/probe_position_jev.json
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
from s1proto.data.teachers import TYPESAFE_URL, normalize

PARAS = [
    "The regional library has extended its weekend opening hours through the autumn, and the reading room on the second floor now stays open until six. Volunteers repainted the community garden fence in a pale green over two Saturdays.",
    "A local walking group meets at the old station car park on the first Sunday of each month for a route of about seven miles, with a shorter loop for anyone who wants to turn back at the mill.",
    "The bakery on the corner has started selling rye loaves on Thursdays, and the queue on the first morning went past the pharmacy. They said they would bake twice as many the following week.",
    "The school's autumn concert is on the last Friday of the month in the main hall, doors at half past six. Parking is at the leisure centre, with a marshal at the gate until the interval.",
    "Repairs to the footbridge over the stream were finished ahead of schedule, and the path along the north bank has reopened. The bench near the weir was replaced with a longer one.",
]


def placed(call: str, where: str) -> str:
    block = "Call transcript:\n" + call
    if where == "start":
        parts = [block] + PARAS
    elif where == "end":
        parts = PARAS + [block]
    else:
        parts = PARAS[:3] + [block] + PARAS[3:]
    return "Notes and records from the day.\n\n" + "\n\n".join(parts)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=TYPESAFE_URL)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    items = [json.loads(line) for line in open("data/water_calls.jsonl") if '"choice"' in line]
    key = os.environ.get("TYPESAFE_API_KEY" if "typesafe" in args.url else "S1_API_KEY", "x")
    client = httpx.AsyncClient(timeout=180.0, headers={"Authorization": f"Bearer {key}"})
    gate = asyncio.Semaphore(args.concurrency)

    async def ask(it, where):
        q = it["question"]
        state = it["state"] if where == "alone" else placed(it["state"] if isinstance(it["state"], str) else json.dumps(it["state"]), where)
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
        d = normalize(r.json()["answers"]["q"]["probabilities"], keys)
        truth = max(keys, key=lambda k: it["ref"][k])
        top = max(keys, key=d.get)
        return {"correct": top == truth, "conf": d[top], "p_truth": d[truth]}

    out = {}
    for where in ("alone", "start", "middle", "end"):
        rows = [x for x in await asyncio.gather(*[ask(it, where) for it in items]) if x]
        n = len(rows)
        out[where] = {
            "n": n,
            "accuracy": round(sum(x["correct"] for x in rows) / n, 3),
            "mean_conf": round(sum(x["conf"] for x in rows) / n, 3),
            "mean_p_truth": round(sum(x["p_truth"] for x in rows) / n, 3),
        }
        r = out[where]
        print(f"  {where:<7} n={n:<4} acc={r['accuracy']:.3f} conf={r['mean_conf']:.3f} p(truth)={r['mean_p_truth']:.3f}", flush=True)
    await client.aclose()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "url": args.url, "results": out}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
