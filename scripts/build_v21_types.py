"""Training and eval rows for the circuit v2.1 question types: `rank` and `match`.

`rank` orders the options, best first; its reference is a grade per option (higher is
better, ties allowed) and it trains on the Plackett-Luce likelihood of that order. `match`
pairs each of several items with one option or "none"; its reference is, per item, the
option it matches.

    uv run python scripts/build_v21_types.py
    # data/v21_types_train.jsonl, data/v21_types_eval.jsonl

  esci_rank      ESCI (Amazon)   a shopper's search + its products -> order by relevance    Apache-2.0
  stackx_rank    Stack Exchange  a question + its answers -> order by readers' votes         CC BY-SA 4.0
  esci_match     ESCI (Amazon)   searches x products -> the exact product each wants, or none Apache-2.0
  cuad_match     CUAD            clause types x a contract's clauses -> which is which, or none CC BY 4.0
  musique_match  MuSiQue         a question's steps x 20 paragraphs -> the one each step needs CC BY 4.0

ESCI grades are Amazon's annotators' labels (Exact 3, Substitute 2, Complement 1,
Irrelevant 0); a match's hard negatives are other searches' Substitutes, which are close
to the product a search wants and are not it. Stack Exchange grades are the answers' vote
scores as the dataset gives them (pm_score); rows keep only questions whose answers span at
least two points, since votes on a close call are noise. Its eval uses sites the train rows
never saw. About one match item in five has its answer removed from the options, so "none"
is learned on the same data. Train rows come from each set's train split, eval rows from
its test or validation split. Option and item keys are opaque ids (ASINs, s1/s2, p0/p1),
so a key never gives the answer away.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_hard_tier import MAX_CHARS, _lit
from build_v2_types import MAX_TOKENS, _cuad_windows

NONE_EVERY = 5
ESCI_GRADE = {"Exact": 3.0, "Substitute": 2.0, "Complement": 1.0, "Irrelevant": 0.0}
STACKX_TRAIN = ["cooking", "diy", "travel", "gardening", "bicycles", "money", "law", "workplace", "parenting", "pets", "mechanics"]
STACKX_EVAL = ["outdoors", "fitness", "academia"]
TAG = re.compile(r"<[^>]+>")


def row(family: str, kind: str, state, question: dict, ref: dict, heldout: bool) -> dict:
    """A JSONL row. Refs are kept as given: grades for rank, {item: {option: 1}} for match."""
    key = json.dumps([state, question], sort_keys=True, ensure_ascii=False)
    return {
        "id": f"{family}-{hashlib.sha1(key.encode()).hexdigest()[:10]}",
        "family": family,
        "kind": kind,
        "heldout": heldout,
        "state": state,
        "question": question,
        "refs": {"jev": ref, "gemini": ref},  # no teachers; the human label stands in for both
        "ref": ref,
    }


# ---------------------------------------------------------------- ESCI
def _esci(split: str, shards: int) -> dict[str, list[dict]]:
    """US-locale ESCI rows grouped by query, read from the first `shards` parquet files."""
    files = sorted(f for f in HfApi().list_repo_files("tasksource/esci", repo_type="dataset") if f.startswith(f"data/{split}-"))
    cols = ["query_id", "query", "product_id", "product_locale", "esci_label", "product_title", "product_brand"]
    by_q = collections.defaultdict(list)
    for f in files[:shards]:
        for r in pq.read_table(hf_hub_download("tasksource/esci", f, repo_type="dataset"), columns=cols).to_pylist():
            if r["product_locale"] == "us" and r["product_title"]:
                by_q[r["query_id"]].append(r)
    return by_q


def _product(r: dict) -> str:
    brand = f" (brand: {r['product_brand']})" if r.get("product_brand") and r["product_brand"] != "None" else ""
    return f"{r['product_title'][:200]}{brand}"


def esci_rank(by_q, n, rng, heldout):
    qids = [q for q, rs in by_q.items() if len(rs) >= 4 and len({r["esci_label"] for r in rs}) >= 2]
    rng.shuffle(qids)
    out = []
    for q in qids[:n]:
        rs = rng.sample(by_q[q], min(8, len(by_q[q])))
        if len({r["esci_label"] for r in rs}) < 2:
            continue
        criteria = {r["product_id"]: _product(r) for r in rs}
        ref = {r["product_id"]: ESCI_GRADE[r["esci_label"]] for r in rs}
        question = {"type": "rank", "instructions": "Order these products by how well each one matches what the shopper searched for.", "criteria": criteria}
        out.append(row("esci_rank", "rank", {"search": rs[0]["query"]}, question, ref, heldout))
    return out


def esci_match(by_q, n, rng, heldout):
    """Four searches against their Exact products and each one's Substitute; one search in
    five loses its Exact product and should match none."""
    ok = [q for q, rs in by_q.items() if any(r["esci_label"] == "Exact" for r in rs) and any(r["esci_label"] == "Substitute" for r in rs)]
    rng.shuffle(ok)
    out = []
    for g in range(0, len(ok) - 3, 4):
        if len(out) >= n:
            break
        group = ok[g : g + 4]
        items, criteria, ref = {}, {}, {}
        for j, q in enumerate(group):
            rs = by_q[q]
            exact = rng.choice([r for r in rs if r["esci_label"] == "Exact"])
            sub = rng.choice([r for r in rs if r["esci_label"] == "Substitute"])
            name = f"search{j + 1}"
            items[name] = rs[0]["query"]
            criteria[sub["product_id"]] = _product(sub)
            if rng.randrange(NONE_EVERY) == 0:
                ref[name] = {"none": 1.0}
            else:
                criteria[exact["product_id"]] = _product(exact)
                ref[name] = {exact["product_id"]: 1.0}
        if any(len(set(ref[i]) & set(criteria)) == 0 and "none" not in ref[i] for i in ref):
            continue
        keys = list(criteria)
        rng.shuffle(keys)
        question = {
            "type": "match",
            "instructions": "Which listed product is exactly what each search is looking for? A close substitute or an accessory is not a match.",
            "items": items,
            "criteria": {k: criteria[k] for k in keys},
            "none": "none of the products is what this search wants",
        }
        out.append(row("esci_match", "match", "A shop's search log and part of its catalogue.", question, ref, heldout))
    return out


# ---------------------------------------------------------------- Stack Exchange
def _text(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", html.unescape(TAG.sub("", s))).strip()


def stackx_rank(sites, n, rng, heldout):
    out = []
    per_site = -(-n // len(sites))
    repo = "HuggingFaceH4/stack-exchange-preferences"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    for site in sites:
        first = min(f for f in files if f.startswith(f"data/{site}.stackexchange.com/") and f.endswith(".parquet"))  # big sites are split
        f = hf_hub_download(repo, first, repo_type="dataset")
        rows = pq.read_table(f, columns=["qid", "question", "answers"]).to_pylist()
        rng.shuffle(rows)
        got = 0
        for r in rows:
            if got >= per_site:
                break
            answers = [a for a in r["answers"] if len(a["text"]) < 3000]
            if not 3 <= len(answers) <= 6:
                continue
            scores = [a["pm_score"] for a in answers]
            if max(scores) - min(scores) < 2:
                continue
            q = _text(r["question"])
            texts = {f"a{a['answer_id']}": _text(a["text"]) for a in answers}
            if len(q) + sum(map(len, texts.values())) > MAX_CHARS:
                continue
            question = {
                "type": "rank",
                "instructions": "Order these answers from most to least useful to the person asking, as the site's readers judged them.",
                "criteria": texts,
            }
            ref = {f"a{a['answer_id']}": float(a["pm_score"]) for a in answers}
            out.append(row("stackx_rank", "rank", {"site": site, "question": q}, question, ref, heldout))
            got += 1
    return out[:n]


# ---------------------------------------------------------------- CUAD
def cuad_match(split, n, rng, heldout):
    """Clause types against clauses cut from one stretch of a contract: each present type's
    first clause, a clause of a type not asked about as a distractor, and absent types, which
    match none."""
    out = []
    for window, clauses in _cuad_windows(split, rng):
        if len(out) >= n:
            break
        present = [c for c, (_, spans) in clauses.items() if spans]
        absent = [c for c, (_, spans) in clauses.items() if not spans]
        if len(present) < 3 or not absent:
            continue
        rng.shuffle(present)
        rng.shuffle(absent)
        asked = present[:3] + absent[:1]
        snippets = {}
        for c in present[:4]:  # the fourth, if there is one, is a clause nobody asks about
            a, b = clauses[c][1][0]
            snippets[c] = window[a:b][:700]
        if len(set(snippets.values())) < len(snippets):
            continue  # two clause types marked on the same text; the match would be ambiguous
        keys = [f"s{j + 1}" for j in range(len(snippets))]
        rng.shuffle(keys)
        key_of = dict(zip(snippets, keys, strict=True))
        rng.shuffle(asked)
        items = {c.lower(): clauses[c][0] for c in asked}
        ref = {c.lower(): ({key_of[c]: 1.0} if c in snippets else {"none": 1.0}) for c in asked}
        question = {
            "type": "match",
            "instructions": "Which of these clauses is each kind of provision? A kind the clauses do not include matches none.",
            "items": items,
            "criteria": {key_of[c]: snippets[c] for c in sorted(snippets, key=key_of.get)},
            "none": "none of these clauses is this kind of provision",
        }
        out.append(row("cuad_match", "match", "Clauses from one commercial contract.", question, ref, heldout))
    return out


# ---------------------------------------------------------------- MuSiQue
def musique_match(split, n, rng, heldout):
    """A multi-hop question's steps against its 20 paragraphs; one question in five loses a
    step's supporting paragraph, and that step matches none. A step that refers to an earlier
    step's answer ("#1") gets that answer written in. Questions with a step in MuSiQue's
    "entity >> relation" notation (two in three) are skipped, so items read as questions."""
    rows = list(load_dataset("dgslibisey/MuSiQue", split=split))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        if len(out) >= n:
            break
        if not r["answerable"]:
            continue
        paras = _lit(r["paragraphs"])
        steps = _lit(r["question_decomposition"])
        if any(">>" in st["question"] for st in steps):
            continue  # MuSiQue's "entity >> relation" lookup notation: no caller writes items like that
        drop = steps[rng.randrange(len(steps))]["paragraph_support_idx"] if i % NONE_EVERY == 0 else None
        keep = [p for p in paras if p["idx"] != drop]
        if sum(len(p["paragraph_text"]) + len(p["title"]) for p in keep) > MAX_CHARS:
            continue
        order = list(range(len(keep)))
        rng.shuffle(order)
        key_of = {keep[j]["idx"]: f"p{k}" for k, j in enumerate(order)}
        items, ref = {}, {}
        for s, step in enumerate(steps):
            text = step["question"]
            for k in range(s):
                text = text.replace(f"#{k + 1}", steps[k]["answer"])
            items[f"step{s + 1}"] = text
            idx = step["paragraph_support_idx"]
            ref[f"step{s + 1}"] = {key_of[idx]: 1.0} if idx in key_of else {"none": 1.0}
        criteria = {key_of[keep[j]["idx"]]: f"[{keep[j]['title']}] {keep[j]['paragraph_text']}" for j in order}
        question = {
            "type": "match",
            "instructions": "Which paragraph answers each step of the question?",
            "items": items,
            "criteria": criteria,
            "none": "no paragraph answers this step",
        }
        out.append(row("musique_match", "match", {"question": r["question"]}, question, ref, heldout))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v21_types_train.jsonl")
    ap.add_argument("--eval", default="data/v21_types_eval.jsonl")
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from s1proto.schema import parse_question
    from s1proto.template import render

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    for path, which, heldout in ((args.train, 0, False), (args.eval, 1, True)):
        rng = random.Random(args.seed + which)
        esci = _esci("train" if which == 0 else "test", 2 if which == 0 else 1)
        plan = [  # (builder, what it reads, rows)
            (esci_rank, esci, (1500, 150)[which]),
            (stackx_rank, (STACKX_TRAIN, STACKX_EVAL)[which], (1200, 120)[which]),
            (esci_match, esci, (1200, 120)[which]),
            (cuad_match, ("train", "test")[which], (1000, 100)[which]),
            (musique_match, ("train", "validation")[which], (1200, 120)[which]),
        ]
        rows = []
        for build, source, n in plan:
            got = build(source, n, rng, heldout)
            before = len(got)
            got = [r for r in got if len(tok(render(r["state"], parse_question(r["question"]), layout="pointer").text).input_ids) <= MAX_TOKENS]
            print(f"{'train' if which == 0 else 'eval'} {build.__name__:14s} {len(got)} (dropped {before - len(got)} over {MAX_TOKENS} tokens)", flush=True)
            rows += got
        rng.shuffle(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} to {path}")


if __name__ == "__main__":
    main()
