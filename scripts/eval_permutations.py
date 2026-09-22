"""Does the answer survive reordering the options?

The same choice question is asked K times: as written, reversed, then shuffled. Nothing
else changes. A model that reads the options gives the same distribution every time; a
model that reads the *position* does not.

    uv run python scripts/eval_permutations.py lora:runs/circuit-8b-v1.1 --out results/perm_circuit-8b-v1.1.json
    TYPESAFE_API_KEY=... uv run python scripts/eval_permutations.py jev --out results/perm_jev.json
    uv run python scripts/eval_permutations.py semif --semif SemIf/.venv/bin/semif-score --backend mlx --out results/perm_semif.json

Items are the choice questions from the eval sets already in the repo, capped per family,
picked by hash so every model gets the same ones in the same orders. Reported:

  flip          share of items whose top answer is not the same under every order
  tv            mean total-variation distance from the as-written distribution (0 = identical)
  acc range     lowest and highest accuracy over the K orders

Jev's API reports two decimals, which puts a floor of about 0.005 x options under its tv
and cannot cause a flip unless two options are within 0.01.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1proto.data.teachers import option_keys

SOURCES = ["data/unseen_eval.jsonl", "data/hf_eval.jsonl", "data/water_calls.jsonl", "data/cmdiy_eval.jsonl", "data/grid_eval.jsonl"]


def pick(files: list[str], per_family: int) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        if not Path(f).exists():
            continue
        for line in open(f):
            it = json.loads(line)
            if it["kind"] == "choice" and len(it["question"]["criteria"]) >= 3:
                fam = it["family"].split("/")[0] if it.get("source") == "grid" or "/" in it["family"] else it["family"]
                by[f"grid:{fam}" if "/" in it["family"] else fam].append(it)
    out = []
    for fam, rows in sorted(by.items()):
        rows.sort(key=lambda it: hashlib.sha1(it["id"].encode()).hexdigest())
        out += [{**it, "family": fam} for it in rows[:per_family]]
    return out


def orders(it: dict, k: int) -> list[list[str]]:
    keys = option_keys(it["question"])
    rng = random.Random(int(hashlib.sha1(it["id"].encode()).hexdigest()[:8], 16))
    out = [keys[:], keys[::-1]]
    while len(out) < k:
        o = keys[:]
        rng.shuffle(o)
        out.append(o)
    return out[:k]


def reordered(q: dict, order: list[str]) -> dict:
    return {**q, "criteria": {key: q["criteria"][key] for key in order}}


def run_lora(spec: str, jobs: list[tuple[dict, list[str]]], batch: int) -> list[dict[str, float] | None]:
    from eval_set import parse_question

    from s1proto.scorer import load_scorer, softmax
    from s1proto.template import render

    scorer = load_scorer(spec)
    layout = getattr(scorer, "layout", "letters")
    out: list[dict[str, float] | None] = []
    for start in range(0, len(jobs), batch):
        part = jobs[start : start + batch]
        prompts = [render(it["state"], parse_question(reordered(it["question"], order)), layout=layout) for it, order in part]
        for (_, order), r in zip(part, scorer.score(prompts, None), strict=True):
            out.append(dict(zip(order, softmax(list(r.logits)), strict=True)))
    return out


async def run_jev(jobs: list[tuple[dict, list[str]]], concurrency: int) -> list[dict[str, float] | None]:
    from s1proto.data.teachers import JevTeacher

    jev = JevTeacher(concurrency=concurrency)
    out: list[dict[str, float] | None] = [None] * len(jobs)

    async def one(i: int) -> None:
        it, order = jobs[i]
        try:
            out[i] = await jev.distribution(it["state"], reordered(it["question"], order))
        except Exception as e:
            print("error", it["id"], str(e)[:100], file=sys.stderr)

    await asyncio.gather(*[one(i) for i in range(len(jobs))])
    await jev.aclose()
    return out


def run_semif(jobs: list[tuple[dict, list[str]]], semif: str, backend: str | None) -> list[dict[str, float] | None]:
    from eval_semif import MAX_OPTIONS, REVISION, to_row

    rows = {}
    for i, (it, order) in enumerate(jobs):
        if len(order) <= MAX_OPTIONS:
            rows[str(i)] = {**to_row({**it, "question": reordered(it["question"], order)}), "id": str(i)}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        cmd = [semif, "--mode", "direct", "--model", "Qwen/Qwen3.5-4B", "--revision", REVISION, "--input", str(src), "--output", str(dst)] + (
            ["--backend", backend] if backend else []
        )
        subprocess.run(cmd, check=True)
        got = {r["id"]: dict(zip(r["option_ids"], r["probabilities"], strict=True)) for r in map(json.loads, dst.read_text().splitlines())}
    return [got.get(str(i)) for i in range(len(jobs))]


def report(items: list[dict], k: int, dists: list[dict[str, float] | None]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups["all"].append(i)
        groups[it["family"]].append(i)
    out = {}
    for name, idx in groups.items():
        flips, tvs, hits, ran = 0, [], [0] * k, 0
        for i in idx:
            ds = dists[i * k : (i + 1) * k]
            if any(d is None for d in ds):
                continue
            ran += 1
            keys = option_keys(items[i]["question"])
            truth = max(keys, key=lambda key: items[i]["ref"][key])
            tops = [max(keys, key=lambda key, d=d: d[key]) for d in ds]
            flips += len(set(tops)) > 1
            tvs += [0.5 * sum(abs(d[key] - ds[0][key]) for key in keys) for d in ds[1:]]
            for j, t in enumerate(tops):
                hits[j] += t == truth
        if ran:
            acc = [h / ran for h in hits]
            out[name] = {
                "n": ran,
                "could_not_run": len(idx) - ran,
                "flip": round(flips / ran, 4),
                "tv": round(sum(tvs) / len(tvs), 4),
                "acc_as_written": round(acc[0], 4),
                "acc_reversed": round(acc[1], 4),
                "acc_min": round(min(acc), 4),
                "acc_max": round(max(acc), 4),
            }
        else:
            out[name] = {"n": 0, "could_not_run": len(idx)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="lora:<run dir> | jev | semif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="orders per item: as written, reversed, then shuffles")
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dump", default=None, help="also write every distribution, per item and order, to this JSONL")
    ap.add_argument("--ids", default=None, help="comma-separated item ids: ask only these")
    ap.add_argument("--semif", default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    items = pick(SOURCES, args.per_family)
    if args.ids:
        want = set(args.ids.split(","))
        items = [it for it in items if it["id"] in want]
    jobs = [(it, order) for it in items for order in orders(it, args.k)]
    print(f"{len(items)} items x {args.k} orders = {len(jobs)} questions", flush=True)
    if args.model == "jev":
        dists = asyncio.run(run_jev(jobs, args.concurrency))
    elif args.model == "semif":
        dists = run_semif(jobs, args.semif, args.backend)
    else:
        dists = run_lora(args.model, jobs, args.batch)
    if args.dump:
        with open(args.dump, "w") as f:
            for j, (it, order) in enumerate(jobs):
                f.write(json.dumps({"id": it["id"], "family": it["family"], "order": order, "dist": dists[j]}) + "\n")
    rep = report(items, args.k, dists)
    for name, r in rep.items():
        if r["n"]:
            print(
                f"  {name:<18} n={r['n']:<4} flip={r['flip']:.3f} tv={r['tv']:.3f} acc {r['acc_as_written']:.3f} written, {r['acc_reversed']:.3f} reversed, range {r['acc_min']:.3f}-{r['acc_max']:.3f}"
            )
        else:
            print(f"  {name:<18} could not run ({r['could_not_run']} items)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "k": args.k, "per_family": args.per_family, "results": rep}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
