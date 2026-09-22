"""Score a /v1/systemone endpoint on data/hard_tier_eval.jsonl: accuracy and Brier per family."""

import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from s1proto.data.teachers import JevTeacher

name, url, key, model = sys.argv[1], (sys.argv[2] or None), sys.argv[3], sys.argv[4]
items = [json.loads(l) for l in open("data/hard_tier_eval.jsonl")]


async def main():
    jev = JevTeacher(api_key=key, model=model, concurrency=4, **({"url": url} if url else {}))
    out = [None] * len(items)

    async def one(i):
        try:
            out[i] = await jev.distribution(items[i]["state"], items[i]["question"])
        except Exception as e:
            print("error", items[i]["id"], str(e)[:120], file=sys.stderr)

    await asyncio.gather(*[one(i) for i in range(len(items))])
    await jev.aclose()
    return out


dists = asyncio.run(main())
fam = collections.defaultdict(lambda: [0, 0, 0.0, 0])
with open(f"results/hard_baseline_{name}.dump.jsonl", "w") as f:
    for it, d in zip(items, dists):
        f.write(json.dumps({"id": it["id"], "family": it["family"], "ref": it["ref"], "dist": d}) + "\n")
        a = fam[it["family"]]
        if d is None:
            a[3] += 1
            continue
        gold = max(it["ref"], key=it["ref"].get)
        a[0] += max(d, key=d.get) == gold
        a[1] += 1
        a[2] += sum((d.get(k, 0) - it["ref"][k]) ** 2 for k in it["ref"])
summ = {}
for f_, (c, n, b, e) in sorted(fam.items()):
    summ[f_] = {"acc": round(c / n, 3), "brier": round(b / n, 3), "n": n, "errors": e}
    print(f"{f_:14s} acc {c / n:.3f}  brier {b / n:.3f}  n={n} errors={e}")
tc = sum(v[0] for v in fam.values())
tn = sum(v[1] for v in fam.values())
summ["all"] = {"acc": round(tc / tn, 3), "n": tn}
print(f"{'all':14s} acc {tc / tn:.3f}  n={tn}")
json.dump(summ, open(f"results/hard_baseline_{name}.json", "w"), indent=1)
