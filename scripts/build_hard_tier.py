"""Long policies and multi-hop evidence: the two families the JevBench hard tier found missing.

On JevBench's public hard items circuit-8b loses to Jev mostly on long policies (7 of 19
against 12) and multi-hop reasoning (10 of 18 against 16). Nothing in the mix asks either,
and the mix was trained at 1,024 tokens while a third of those items run past 1,300. These
families are real documents with human or code labels, sized to fit 4,096 tokens.

    uv run python scripts/build_hard_tier.py
    # data/hard_tier_train.jsonl, data/hard_tier_eval.jsonl

  maud_deal       MAUD      merger-agreement excerpt -> the lawyer-labelled answer      choice   CC BY 4.0
  sharc_policy    ShARC     a whole rules page + a caller's situation -> the outcome    choice   CC BY-SA 3.0
  cuad_clause     CUAD      a stretch of a contract -> does it contain this clause?     noul     CC BY 4.0
  musique_hops    MuSiQue   20 paragraphs + a 2-4 hop question + an answer -> status    choice   CC BY 4.0
  wiki2_hops      2Wiki     10 paragraphs + a comparison or bridge question -> answer   noul     Apache-2.0

Hard negatives throughout: ShARC policies are every snippet from the same source page, so
the right clause has to be found; a MuSiQue wrong answer is the answer to one of its own
intermediate hops; an unanswerable MuSiQue item removes one hop's supporting paragraph,
which is how MuSiQue's authors built theirs. Train rows come from each set's train split,
eval rows from its validation or test split. No JevBench item is used.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf_eval import item, onehot

MAX_CHARS = 13000  # about 3,300 tokens of state, leaving room for the question in 4,096


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


# ---------------------------------------------------------------- MAUD
def maud(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    train = load_dataset("theatticusproject/maud", split="train")
    options = collections.defaultdict(set)
    for q, a in zip(train["question"], train["answer"], strict=True):
        if q.endswith("-Answer") and a:
            options[q].add(a)
    d = load_dataset("theatticusproject/maud", split=split)
    rows = [
        r
        for r in d
        if r["question"].endswith("-Answer") and r["data_type"] != "abridged" and len(r["text"]) <= MAX_CHARS and r["answer"] in options[r["question"]]
    ]
    rng.shuffle(rows)
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)
    picked, i = [], 0
    while len(picked) < n and any(by_q.values()):  # round-robin so no question type dominates
        q = list(by_q)[i % len(by_q)]
        if by_q[q]:
            picked.append(by_q[q].pop())
        i += 1
    out = []
    for r in picked:
        topic = r["question"].removesuffix("-Answer")
        keys = sorted(options[r["question"]])
        q = {"type": "choice", "instructions": f"Reading this excerpt of a merger agreement, which describes it: {topic}?", "criteria": dict.fromkeys(keys)}
        out.append(item("maud_deal", "choice", {"agreement_excerpt": r["text"]}, q, onehot(keys, r["answer"]), heldout))
    return out


# ---------------------------------------------------------------- ShARC
SHARC_KEYS = {"Yes": "yes", "No": "no", "Irrelevant": "not_covered"}


def sharc(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    d = load_dataset("UCLNLP/sharc", revision="refs/convert/parquet", split=split)
    pages = collections.defaultdict(set)
    for r in d:
        pages[r["source_url"]].add(r["snippet"])
    rows = list(d)
    rng.shuffle(rows)
    buckets, seen = collections.defaultdict(list), set()
    crit = {
        "yes": "The rules say yes for this person",
        "no": "The rules say no for this person",
        "not_covered": "The rules on this page do not answer this question",
        "need_more_information": "The rules could answer it, but the situation does not yet say enough",
    }
    for r in rows:
        key = (r["source_url"], r["question"], r["scenario"], json.dumps(r["history"], sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        label = SHARC_KEYS.get(r["answer"], "need_more_information")
        snippets = sorted(pages[r["source_url"]])
        rng.shuffle(snippets)
        policy = "\n\n".join(snippets)
        if len(policy) > MAX_CHARS:
            continue
        history = _lit(r["history"]) or []
        facts = [f"Asked: {h['follow_up_question']} Answered: {h['follow_up_answer']}" for h in history]
        state = {"policy": policy, "question": r["question"], "situation": r["scenario"] or "(none given)", "already_established": facts}
        q = {
            "type": "choice",
            "instructions": "Apply `policy` to this person's `situation` and what is `already_established`. What is the answer to their `question`?",
            "criteria": crit,
        }
        buckets[label].append(item("sharc_policy", "choice", state, q, onehot(list(crit), label), heldout))
    # Round-robin by label: "not_covered" is ~5% of ShARC as shipped.
    out, queues = [], [buckets[k] for k in crit]
    while len(out) < n and any(queues):
        for qu in queues:
            if qu and len(out) < n:
                out.append(qu.pop())
    return out


# ---------------------------------------------------------------- CUAD
def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def cuad(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    d = load_dataset("theatticusproject/cuad-qa", revision="refs/convert/parquet", split=split)
    by_doc = collections.defaultdict(list)
    for r in d:
        by_doc[r["title"]].append(r)
    docs = list(by_doc)
    rng.shuffle(docs)
    pos, neg = [], []
    for title in docs:
        rows = by_doc[title]
        text = rows[0]["context"]
        if len(text) < 3000:
            continue
        start = rng.randrange(0, max(1, len(text) - MAX_CHARS))
        window = text[start : start + MAX_CHARS]
        for r in rows:
            clause = r["question"].split('related to "')[1].split('"')[0] if 'related to "' in r["question"] else None
            detail = r["question"].split("Details: ", 1)[-1]
            if not clause or clause in ("Document Name", "Parties", "Agreement Date", "Effective Date"):
                continue
            ans = _lit(r["answers"])
            inside = any(start <= s and s + len(t) <= start + MAX_CHARS for s, t in zip(ans["answer_start"], ans["text"], strict=True))
            q = {
                "type": "noul",
                "instructions": f"Does this part of the contract contain {article(clause)} {clause.lower()} provision? ({detail})",
                "criteria": {"true": "It does", "false": "It does not, at least not in this part"},
            }
            it = item("cuad_clause", "noul", {"contract_excerpt": window}, q, {"yes": float(inside), "no": float(not inside)}, heldout)
            (pos if inside else neg).append(it)
        if len(pos) >= n // 2 and len(neg) >= n:
            break
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[: n // 2] + neg[: n - n // 2]


# ---------------------------------------------------------------- MuSiQue
MUS_CRIT = {
    "supported": "The paragraphs establish this answer, following each step",
    "wrong": "The paragraphs establish a different answer",
    "not_enough_information": "A step needed to reach any answer is missing from the paragraphs",
}


def musique(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    d = load_dataset("dgslibisey/MuSiQue", split=split)
    rows = list(d)
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        if len(out) >= n:
            break
        paras = _lit(r["paragraphs"])
        decomp = _lit(r["question_decomposition"])
        kind = ["supported", "wrong", "not_enough_information"][i % 3]
        answer = r["answer"]
        keep = list(paras)
        if kind == "wrong":
            inter = [h["answer"] for h in decomp[:-1] if h["answer"].lower() != answer.lower()]
            if not inter:
                continue
            answer = rng.choice(inter)  # the answer to an intermediate hop: right entity, wrong question
        elif kind == "not_enough_information":
            drop = rng.choice([h["paragraph_support_idx"] for h in decomp])
            keep = [p for p in paras if p["idx"] != drop]
        text = "\n\n".join(f"[{p['title']}] {p['paragraph_text']}" for p in keep)
        if len(text) > MAX_CHARS:
            continue
        state = {"paragraphs": text, "question": r["question"], "proposed_answer": answer}
        q = {"type": "choice", "instructions": "Going only on `paragraphs`, what is the status of `proposed_answer` to `question`?", "criteria": MUS_CRIT}
        out.append(item("musique_hops", "choice", state, q, onehot(list(MUS_CRIT), kind), heldout))
    return out


# ---------------------------------------------------------------- 2WikiMultihopQA
def wiki2(split: str, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    d = load_dataset("xanhho/2WikiMultihopQA", revision="refs/convert/parquet", split=split)
    idx = list(range(len(d)))
    rng.shuffle(idx)
    answers = [d[i]["answer"] for i in idx[:5000]]
    out = []
    for j, i in enumerate(idx):
        if len(out) >= n:
            break
        r = d[i]
        ctx = _lit(r["context"])
        text = "\n\n".join(f"[{t}] {' '.join(s)}" for t, s in ctx)
        if len(text) > MAX_CHARS:
            continue
        gold = r["answer"]
        if j % 2 == 0:
            answer, ok = gold, True
        elif gold.lower() in ("yes", "no"):
            answer, ok = ("no" if gold.lower() == "yes" else "yes"), False
        else:
            same_type = [a for a in answers if a.lower() not in ("yes", "no") and a != gold]
            answer, ok = rng.choice(same_type), False
        state = {"paragraphs": text, "question": r["question"], "proposed_answer": answer}
        q = {
            "type": "noul",
            "instructions": "Going only on `paragraphs`, is `proposed_answer` the correct answer to `question`?",
            "criteria": {"true": "The paragraphs establish it", "false": "The paragraphs establish something else"},
        }
        out.append(item("wiki2_hops", "noul", state, q, {"yes": float(ok), "no": float(not ok)}, heldout))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/hard_tier_train.jsonl")
    ap.add_argument("--eval", default="data/hard_tier_eval.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    plan = [  # (builder, train split, eval split, train n, eval n)
        (maud, "train", "test", 1500, 150),
        (sharc, "train", "validation", 1500, 150),
        (cuad, "train", "test", 1000, 100),
        (musique, "train", "validation", 1500, 150),
        (wiki2, "train", "validation", 800, 100),
    ]
    for path, which, heldout in ((args.train, 0, False), (args.eval, 1, True)):
        rng = random.Random(args.seed + which)
        rows = []
        for build, tr, ev, ntr, nev in plan:
            got = build(tr if which == 0 else ev, ntr if which == 0 else nev, rng, heldout)
            print(f"{'train' if which == 0 else 'eval'} {build.__name__:8s} {len(got)}", flush=True)
            rows += got
        rng.shuffle(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} to {path}")


if __name__ == "__main__":
    main()
