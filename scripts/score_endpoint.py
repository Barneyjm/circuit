"""Score a /v1/systemone endpoint on an eval JSONL: accuracy and Brier per family.

uv run python scripts/score_endpoint.py <name> <url or ""> <key> <model> [data.jsonl]
# results/endpoint_<name>.json and .dump.jsonl; url "" means TypeSafe (Jev)
"""

import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from s1proto.data.teachers import JevTeacher

name, url, key, model = sys.argv[1], (sys.argv[2] or None), sys.argv[3], sys.argv[4]
data = sys.argv[5] if len(sys.argv) > 5 else "data/hard_tier_eval.jsonl"
items = [json.loads(l) for l in open(data)]


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


def ece15(confs, correct, bins=15):
    n, e = len(confs), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0.0)]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return e


rows = collections.defaultdict(list)  # family -> [(conf, correct, brier)]
errors = collections.Counter()
with open(f"results/endpoint_{name}.dump.jsonl", "w") as f:
    for it, d in zip(items, dists, strict=True):
        f.write(json.dumps({"id": it["id"], "family": it["family"], "ref": it["ref"], "dist": d}) + "\n")
        if d is None:
            errors[it["family"]] += 1
            continue
        gold = max(it["ref"], key=it["ref"].get)
        top = max(d, key=d.get)
        rows[it["family"]].append((d[top], top == gold, sum((d.get(k, 0) - it["ref"][k]) ** 2 for k in it["ref"])))


def stats(rs):
    n = len(rs)
    return {
        "acc": round(sum(r[1] for r in rs) / n, 4),
        "ece": round(ece15([r[0] for r in rs], [r[1] for r in rs]), 4),
        "brier": round(sum(r[2] for r in rs) / n, 4),
        "n": n,
    }


summ = {f_: {**stats(rs), "errors": errors[f_]} for f_, rs in sorted(rows.items())}
summ["all"] = {**stats([r for rs in rows.values() for r in rs]), "errors": sum(errors.values())}
for f_, v in summ.items():
    print(f"{f_:24s} acc {v['acc']:.3f}  ece {v['ece']:.3f}  brier {v['brier']:.3f}  n={v['n']} errors={v['errors']}")
json.dump(summ, open(f"results/endpoint_{name}.json", "w"), indent=1)
