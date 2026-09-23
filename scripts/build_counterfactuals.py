"""Counterfactual groups: the same item with one fact changed, so the answer must move with
the fact that decides it and stay put when something else changes.

A model can score well on a family by leaning on cues that travel with the answer in that
family and nowhere else. A group asks one question of several versions of the same state,
and only the deciding fact separates them, so the cue cannot answer it. Rows carry a
`group` id; `eval_set.py` reports how many groups the model gets entirely right.

    uv run python scripts/build_counterfactuals.py
    # data/cf_train.jsonl, data/cf_eval.jsonl

  cf_snli   Kaushik et al.      an SNLI pair and four human edits (premise or hypothesis),   Apache-2.0 edits
            (2020)              each rewritten to one of the other two labels                 on SNLI, CC BY-SA 4.0
  cf_wiki2  2WikiMultihopQA     paragraphs + a proposed answer, four versions: as written     Apache-2.0
                                (true); the answer swapped for another of the same relation
                                everywhere in the text (now false; the swapped-in answer now
                                true); a paragraph that no hop uses removed (still true)

The SNLI rows use the wide mix's `snli` question wording, so an edited pair is the same task
the model already knows. Its eval groups come from the edit set's test split and skip any
original that the wide mix trains on. 2Wiki train groups come from its train split, eval
groups from its validation split; the substitute answer is another question's answer under
the same final relation ("award received", "place of birth"), so the edited text still reads
as true of some world.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_hard_tier import MAX_CHARS, _lit

CAD = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/{part}/{split}.tsv"
SNLI_Q = "Given `premise`, what is the status of `hypothesis`?"  # as build_wide_mix.py's snli
SNLI_CRIT = {
    "entailed": "Must be true if the premise is true",
    "neutral": "The premise does not settle it",
    "contradicted": "Must be false if the premise is true",
}
SNLI_LABEL = {"entailment": "entailed", "neutral": "neutral", "contradiction": "contradicted"}
WIKI2_Q = "Going only on `paragraphs`, is `proposed_answer` the correct answer to `question`?"  # as build_hard_tier.py's wiki2_hops
WIKI2_CRIT = {"true": "The paragraphs establish it", "false": "The paragraphs establish something else"}


def row(family: str, kind: str, group: str, state: dict, question: dict, ref: dict[str, float], heldout: bool) -> dict:
    key = json.dumps([state, question], sort_keys=True, ensure_ascii=False)
    return {
        "id": f"{family}-{hashlib.sha1(key.encode()).hexdigest()[:10]}",
        "family": family,
        "kind": kind,
        "group": group,
        "heldout": heldout,
        "state": state,
        "question": question,
        "refs": {"jev": ref, "gemini": ref},  # no teachers; the label stands in for both
        "ref": ref,
    }


# ---------------------------------------------------------------- SNLI edits
def _tsv(part: str, split: str) -> list[dict]:
    with urllib.request.urlopen(CAD.format(part=part, split=split), timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))


def cf_snli(split: str, n: int, rng: random.Random, heldout: bool, trained: set[tuple[str, str]]) -> list[dict]:
    """Each original is followed in the edit files by its two premise edits and its two
    hypothesis edits, in order; the grouping is checked on the text, not assumed."""
    orig, prem, hyp = (_tsv(p, split) for p in ("original", "revised_premise", "revised_hypothesis"))
    out = []
    order = list(range(len(orig)))
    rng.shuffle(order)
    for i in order:
        if len(out) >= 5 * n:
            break
        o = orig[i]
        if heldout and (o["sentence1"], o["sentence2"]) in trained:
            continue
        ps, hs = prem[2 * i : 2 * i + 2], hyp[2 * i : 2 * i + 2]
        if len(ps) < 2 or len(hs) < 2 or any(p["sentence2"] != o["sentence2"] for p in ps) or any(h["sentence1"] != o["sentence1"] for h in hs):
            continue  # misaligned: skip rather than guess
        group = f"cf_snli-{hashlib.sha1((o['sentence1'] + o['sentence2']).encode()).hexdigest()[:10]}"
        q = {"type": "choice", "instructions": SNLI_Q, "criteria": SNLI_CRIT}
        members = (o, *ps, *hs)
        if any(r["gold_label"] not in SNLI_LABEL for r in members):
            continue
        for r in members:
            label = SNLI_LABEL[r["gold_label"]]
            ref = {k: float(k == label) for k in SNLI_CRIT}
            out.append(row("cf_snli", "choice", group, {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, q, ref, heldout))
    return out


# ---------------------------------------------------------------- 2Wiki swaps
def _wiki2_rows(split: str) -> list[dict]:
    d = load_dataset("xanhho/2WikiMultihopQA", revision="refs/convert/parquet", split=split)
    rows = []
    for r in d:
        if r["type"] not in ("compositional", "inference"):
            continue  # comparison answers are yes/no or one of two names in the question: nothing to swap
        ev = _lit(r["evidences"])
        if not ev or len(r["answer"]) < 4:
            continue
        rows.append({**r, "relation": ev[-1][1], "context": _lit(r["context"]), "supporting_facts": _lit(r["supporting_facts"])})
    return rows


def cf_wiki2(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    rows = _wiki2_rows(split)
    by_rel = collections.defaultdict(set)
    for r in rows:
        by_rel[r["relation"]].add(r["answer"])
    rng.shuffle(rows)
    q = {"type": "noul", "instructions": WIKI2_Q, "criteria": WIKI2_CRIT}
    out = []
    for r in rows:
        if len(out) >= 4 * n:
            break
        gold = r["answer"]
        text = "\n\n".join(f"[{t}] {' '.join(s)}" for t, s in r["context"])
        if len(text) > MAX_CHARS or gold not in text:
            continue
        others = sorted(a for a in by_rel[r["relation"]] if a.lower() != gold.lower() and a.lower() not in text.lower() and gold.lower() not in a.lower())
        if not others:
            continue
        swap = rng.choice(others)
        swapped = re.sub(rf"(?<!\w){re.escape(gold)}(?!\w)", lambda _m, swap=swap: swap, text)  # whole words; a function so \1 in a name is literal
        if gold.lower() in swapped.lower():
            continue  # another spelling of the answer survives the swap; the label would be wrong
        used = {t for t, _ in r["supporting_facts"]}
        spare = [j for j, (t, _) in enumerate(r["context"]) if t not in used]
        if not spare:
            continue
        drop = rng.choice(spare)
        fewer = "\n\n".join(f"[{t}] {' '.join(s)}" for j, (t, s) in enumerate(r["context"]) if j != drop)
        group = f"cf_wiki2-{r['_id']}"
        for paras, proposed, ok in ((text, gold, True), (swapped, gold, False), (swapped, swap, True), (fewer, gold, True)):
            state = {"paragraphs": paras, "question": r["question"], "proposed_answer": proposed}
            out.append(row("cf_wiki2", "noul", group, state, q, {"yes": float(ok), "no": float(not ok)}, heldout))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/cf_train.jsonl")
    ap.add_argument("--eval", default="data/cf_eval.jsonl")
    ap.add_argument("--mix", default="data/publish_train_v2.jsonl", help="the wide mix: eval groups skip SNLI originals it trains on")
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    trained = set()
    if Path(args.mix).exists():
        for line in open(args.mix):
            r = json.loads(line)
            if r["family"] == "snli":
                trained.add((r["state"]["premise"], r["state"]["hypothesis"]))
    for path, which, heldout in ((args.train, 0, False), (args.eval, 1, True)):
        rng = random.Random(args.seed + which)
        rows = cf_snli(("train", "test")[which], (600, 100)[which], rng, heldout, trained)
        rows += cf_wiki2(("train", "validation")[which], (600, 100)[which], rng, heldout)
        for fam, c in sorted(collections.Counter(r["family"] for r in rows).items()):
            groups = len({r["group"] for r in rows if r["family"] == fam})
            print(f"{'train' if which == 0 else 'eval'} {fam:9s} {c} rows in {groups} groups", flush=True)
        rng.shuffle(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} to {path}")


if __name__ == "__main__":
    main()
