"""Build the eval / training sets: synthetic states per family, reference
distributions from two teachers (Jev + Gemini), averaged.

    TYPESAFE_API_KEY=... uv run python scripts/gen_data.py --split eval  --per-family 100 --out data/eval.jsonl
    TYPESAFE_API_KEY=... uv run python scripts/gen_data.py --split train --per-family 400 --out data/train.jsonl

`--split eval` covers every family (held-out ones included); `train`
covers only non-held-out families. State generation and the eval set
use different seeds so train/eval states are disjoint; a hash check
drops any exact duplicate anyway.

Output: JSONL, one item per line:
  {id, family, kind, heldout, state, question, refs: {jev: {...}, gemini: {...}}, ref: {...}}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.families import FAMILIES, TRAIN_FAMILIES, Family
from s1proto.data.teachers import GeminiTeacher, JevTeacher, average, option_keys

GEN_BATCH = 20
GEN_BATCH_JSON = 10  # JSON states are long; keep responses well under the output cap

SEED_TOPICS = [
    "a small e-commerce shop", "a hospital scheduling system", "a fintech app", "a university IT desk", "a video game studio",
    "a logistics company", "a SaaS analytics product", "a city parking authority", "a dental clinic", "a music streaming service",
    "a home-improvement retailer", "a nonprofit fundraising platform", "a ride-hailing app", "a law firm", "a food delivery service",
    "an airline", "a cloud hosting provider", "a coworking space", "a language-learning app", "a pet insurance company",
]


def _gen_prompt(fam: Family, n: int, rng: random.Random) -> str:
    topics = rng.sample(SEED_TOPICS, 4)
    fmt = "a JSON object" if fam.state_format == "json" else "a plain string"
    return (
        f"Generate {n} diverse, realistic examples for a decision task. Each example is a STATE: {fam.state_brief}.\n"
        f"Each STATE must be {fmt}. Vary length, register, difficulty, and domain; draw some from these settings: {', '.join(topics)}. "
        "Include easy, hard, ambiguous, and edge cases in roughly equal measure. Do not include the answer or any labels. "
        "Never use real people's personal details; invent names. "
        f"The question that will be asked about each state is: \"{fam.instructions}\" — make sure the states exercise every plausible answer.\n"
        'Output a JSON object {"states": [ ... ]} and nothing else.'
    )


async def gen_states(gem: GeminiTeacher, fam: Family, n: int, rng: random.Random) -> list:
    """Fire generation batches concurrently (the teacher's semaphore
    bounds the actual parallelism); dedupe by content hash."""
    out: list = []
    seen: set[str] = set()
    bsz = GEN_BATCH_JSON if fam.state_format == "json" else GEN_BATCH
    rounds = 0
    while len(out) < n and rounds < 4:
        rounds += 1
        n_batches = max(1, -(-(n - len(out)) // bsz))  # ceil
        prompts = [_gen_prompt(fam, min(bsz, n - len(out) + 2), rng) for _ in range(n_batches)]
        results = await asyncio.gather(*[gem.generate_json(pr) for pr in prompts], return_exceptions=True)
        for obj in results:
            if isinstance(obj, BaseException):
                print(f"  [{fam.key}] gen error: {obj}", file=sys.stderr)
                continue
            states = obj.get("states") if isinstance(obj, dict) else obj
            if not isinstance(states, list):
                continue
            for st in states:
                if fam.state_format == "json" and not isinstance(st, dict):
                    continue
                if fam.state_format == "text" and not isinstance(st, str):
                    continue
                h = hashlib.sha1(json.dumps(st, sort_keys=True).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                out.append(st)
    return out[:n]


async def label(jev: JevTeacher, gem: GeminiTeacher, fam: Family, state, idx: int, rng: random.Random) -> dict | None:
    q = fam.question()
    keys = option_keys(q)
    try:
        dj, dg = await asyncio.gather(jev.distribution(state, q), gem.distribution(state, q))
    except Exception as e:
        print(f"  [{fam.key}#{idx}] label error: {e}", file=sys.stderr)
        return None
    return {
        "id": f"{fam.key}-{hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()[:10]}",
        "family": fam.key,
        "kind": fam.kind,
        "heldout": fam.heldout,
        "state": state,
        "question": q,
        "refs": {"jev": dj, "gemini": dg},
        "ref": average([dj, dg], keys),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], required=True)
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--families", default=None, help="comma-separated subset")
    ap.add_argument("--gemini-model", default="gemini-3.5-flash")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (11 if args.split == "eval" else 23)
    rng = random.Random(seed)
    fams = FAMILIES if args.split == "eval" else TRAIN_FAMILIES
    if args.families:
        want = set(args.families.split(","))
        fams = [f for f in fams if f.key in want]

    jev = JevTeacher(concurrency=args.concurrency)
    gem = GeminiTeacher(model=args.gemini_model, concurrency=args.concurrency)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                existing.add(json.loads(line)["id"])
            except Exception:
                pass
        print(f"resuming: {len(existing)} items already in {out_path}")

    t0 = time.time()
    total = 0
    with out_path.open("a") as fh:
        for fam in fams:
            have = sum(1 for i in existing if i.startswith(fam.key + "-"))
            need = max(0, args.per_family - have)
            if need == 0:
                print(f"[{fam.key}] complete ({have})")
                continue
            print(f"[{fam.key}] generating {need} states ...", flush=True)
            states = await gen_states(gem, fam, need, rng)
            print(f"[{fam.key}] labeling {len(states)} ...", flush=True)
            items = await asyncio.gather(*[label(jev, gem, fam, s, i, rng) for i, s in enumerate(states)])
            n_ok = 0
            for it in items:
                if it and it["id"] not in existing:
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                    existing.add(it["id"])
                    n_ok += 1
            fh.flush()
            total += n_ok
            print(f"[{fam.key}] wrote {n_ok}  (elapsed {time.time() - t0:.0f}s)", flush=True)
    await jev.aclose()
    print(f"done: {total} new items -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
