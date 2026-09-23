"""Training and eval rows for the circuit v2 question types: `locate` and `multi`.

`locate` points at the part of the state that answers the question (or "none"); every
string value in the state is a candidate, so each source lays its text out as a list at
the grain the answer lives at (paragraphs, sentences, rules). `multi` gives every option
its own probability; references are 1/0 per option, or the share of raters where there
were several.

    uv run python scripts/build_v2_types.py
    # data/v2_types_train.jsonl, data/v2_types_eval.jsonl

  musique_locate  MuSiQue    20 paragraphs -> the ones a 2-4 hop answer needs          CC BY 4.0
  wiki2_locate    2Wiki      10 passages as sentences -> the supporting sentences      Apache-2.0
  sharc_locate    ShARC      every rule on a gov.uk page -> the one that decides it    CC BY-SA 3.0
  cuad_locate     CUAD       a contract stretch as sentences -> the clause's sentence  CC BY 4.0
  squad_locate    SQuAD 2.0  a passage as sentences -> the answer's sentence, or none  CC BY-SA 4.0
  cuad_multi      CUAD       a contract stretch -> which of six provisions it has      CC BY 4.0
  goemo_multi     GoEmotions a Reddit comment -> which of 28 emotions, share of raters Apache-2.0
  sharc_asknext   ShARC      rules + situation -> the follow-up that settles it, or none CC BY-SA 3.0

sharc_asknext is a plain `choice`: the follow-up ShARC's annotators asked next, against
questions already asked (asking again learns nothing), follow-ups from other rules pages,
and "nothing more is needed". The page's other unasked follow-ups are left out: ShARC
records one order of asking, and on many pages several of them are equally good next
questions, so they would be wrong labels, not hard negatives.

About one row in six of each locate source has its answer removed (the supporting
paragraphs, passages or clause are not in the state), so "none" is learned from the same
documents. Train rows come from each set's train split, eval rows from its validation or
test split (GoEmotions has one split; its eval is a fixed 10% by comment id).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_hard_tier import MAX_CHARS, SHARC_KEYS, _lit
from build_hf_eval import item

from s1proto.template import locate_candidates

MAX_TOKENS = 4000  # rendered prompt, Qwen3 tokenizer; training runs at 4,096
SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'0-9])")
NONE_EVERY = 6


def sentences_with_offsets(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for part in SENTENCE.split(text):
        start = text.find(part, pos)
        if part.strip():
            out.append((start, part.strip()))
        pos = start + len(part)
    return out


def locate_item(family: str, state: dict, instructions: str, gold: list[str], heldout: bool, none_means: str | None = None) -> dict | None:
    """gold: candidate paths that answer it; empty means "none". Checked against the
    renderer's own candidate list so a path typo cannot produce a row that never scores."""
    paths = {p for p, _ in locate_candidates(state)}
    if any(g not in paths for g in gold) or len(paths) > 500:
        return None
    q = {"type": "locate", "instructions": instructions}
    if none_means:
        q["criteria"] = none_means
    ref = {g: 1.0 for g in gold} if gold else {"none": 1.0}
    return item(family, "locate", state, q, ref, heldout)


def multi_item(family: str, state: dict, instructions: str, criteria: dict, ref: dict[str, float], heldout: bool) -> dict:
    it = item(family, "multi", state, {"type": "multi", "instructions": instructions, "criteria": criteria}, {k: 1.0 for k in ref}, heldout)
    it["ref"] = dict(ref)  # independent per option: not normalized
    it["refs"] = {"jev": dict(ref), "gemini": dict(ref)}
    return it


# ---------------------------------------------------------------- MuSiQue
def musique_locate(split, n, rng, heldout):
    rows = list(load_dataset("dgslibisey/MuSiQue", split=split))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        if len(out) >= n:
            break
        if not r["answerable"]:
            continue
        paras = _lit(r["paragraphs"])
        drop = i % NONE_EVERY == 0
        keep = [p for p in paras if not (drop and p["is_supporting"])]
        texts = [f"[{p['title']}] {p['paragraph_text']}" for p in keep]
        if sum(map(len, texts)) > MAX_CHARS:
            continue
        state = {"question": r["question"], "paragraphs": texts}
        gold = [f"paragraphs[{j}]" for j, p in enumerate(keep) if p["is_supporting"]]
        it = locate_item("musique_locate", state, "Which paragraph gives a fact needed to answer `question`?", gold, heldout)
        if it:
            out.append(it)
    return out


# ---------------------------------------------------------------- 2WikiMultihopQA
def wiki2_locate(split, n, rng, heldout):
    d = load_dataset("xanhho/2WikiMultihopQA", revision="refs/convert/parquet", split=split)
    idx = list(range(len(d)))
    rng.shuffle(idx)
    out = []
    for j, i in enumerate(idx):
        if len(out) >= n:
            break
        r = d[i]
        ctx = [(t, s) for t, s in _lit(r["context"])]
        facts = _lit(r["supporting_facts"])
        titles = [t for t, _ in ctx]
        if len(set(titles)) != len(titles) or any(t not in titles for t, _ in facts):
            continue
        drop = j % NONE_EVERY == 0
        support = {t for t, _ in facts}
        passages = {t: s for t, s in ctx if not (drop and t in support)}
        if sum(len(x) for s in passages.values() for x in s) > MAX_CHARS:
            continue
        state = {"question": r["question"], "passages": passages}
        gold = [] if drop else [f"passages.{t}[{k}]" for t, k in facts if k < len(passages[t])]
        if not drop and len(gold) != len(facts):
            continue
        it = locate_item("wiki2_locate", state, "Which sentence gives a fact needed to answer `question`?", gold, heldout)
        if it:
            out.append(it)
    return out


# ---------------------------------------------------------------- ShARC
def sharc_locate(split, n, rng, heldout):
    d = load_dataset("UCLNLP/sharc", revision="refs/convert/parquet", split=split)
    pages = collections.defaultdict(set)
    for r in d:
        pages[r["source_url"]].add(r["snippet"])
    rows = list(d)
    rng.shuffle(rows)
    covered, uncovered, seen = [], [], set()
    for r in rows:
        key = (r["source_url"], r["question"], r["snippet"])
        if key in seen:
            continue
        seen.add(key)
        rules = sorted(pages[r["source_url"]])
        rng.shuffle(rules)
        if sum(map(len, rules)) > MAX_CHARS or len(rules) < 2:
            continue
        state = {"policy": rules, "question": r["question"], "situation": r["scenario"] or "(none given)"}
        irrelevant = SHARC_KEYS.get(r["answer"]) == "not_covered"
        gold = [] if irrelevant else [f"policy[{rules.index(r['snippet'])}]"]
        it = locate_item("sharc_locate", state, "Which rule in `policy` decides this person's `question`?", gold, heldout, "no rule on this page covers it")
        if it:
            (uncovered if irrelevant else covered).append(it)
    k = min(len(uncovered), n // NONE_EVERY)
    return covered[: n - k] + uncovered[:k]


ASK_NONE = "nothing more is needed; it can be answered now"


def sharc_asknext(split, n, rng, heldout):
    d = load_dataset("UCLNLP/sharc", revision="refs/convert/parquet", split=split)
    pages, followups = collections.defaultdict(set), collections.defaultdict(set)
    for r in d:
        pages[r["source_url"]].add(r["snippet"])
        if r["answer"] not in ("Yes", "No", "Irrelevant"):
            followups[r["source_url"]].add(r["answer"].strip())
    rows = list(d)
    rng.shuffle(rows)
    ask, settled, seen = [], [], set()
    for r in rows:
        url = r["source_url"]
        if r["answer"] == "Irrelevant" or len(followups[url]) < 2:
            continue
        history = _lit(r["history"]) or []
        key = (url, r["question"], r["scenario"], json.dumps(history, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        rules = sorted(pages[url])
        rng.shuffle(rules)
        if sum(map(len, rules)) > MAX_CHARS:
            continue
        gold = ASK_NONE if r["answer"] in ("Yes", "No") else r["answer"].strip()
        asked = [h["follow_up_question"].strip() for h in history]
        others = [u for u in followups if u != url]
        elsewhere = [q for u in rng.sample(others, 3) for q in rng.sample(sorted(followups[u]), 1)]
        opts = list(dict.fromkeys([*([gold] if gold != ASK_NONE else []), *asked, *elsewhere]))
        rng.shuffle(opts)
        opts.append(ASK_NONE)
        if len(opts) < 3:
            continue
        facts = [f"Asked: {h['follow_up_question']} Answered: {h['follow_up_answer']}" for h in history]
        state = {"policy": "\n\n".join(rules), "question": r["question"], "situation": r["scenario"] or "(none given)", "already_established": facts}
        q = {
            "type": "choice",
            "instructions": "To answer this person's `question` under `policy`, what should they be asked next?",
            "criteria": {o: None for o in opts},
        }
        (settled if gold == ASK_NONE else ask).append(item("sharc_asknext", "choice", state, q, {o: float(o == gold) for o in opts}, heldout))
    k = min(len(settled), n // 5)
    return ask[: n - k] + settled[:k]


# ---------------------------------------------------------------- CUAD
def _cuad_windows(split, rng):
    d = load_dataset("theatticusproject/cuad-qa", revision="refs/convert/parquet", split=split)
    by_doc = collections.defaultdict(list)
    for r in d:
        by_doc[r["title"]].append(r)
    docs = list(by_doc)
    rng.shuffle(docs)
    for title in docs:
        rows = by_doc[title]
        text = rows[0]["context"]
        if len(text) < 3000:
            continue
        # up to three stretches of each contract: there are only ~400 contracts
        for start in sorted(rng.sample(range(max(1, len(text) - MAX_CHARS)), min(3, max(1, len(text) - MAX_CHARS)))):
            yield from _cuad_window(rows, text, start)


def _cuad_window(rows, text, start):
    """One stretch of a contract, with each clause type's answer spans inside it."""
    window = text[start : start + MAX_CHARS]
    clauses = {}
    for r in rows:
        if 'related to "' not in r["question"]:
            continue
        clause = r["question"].split('related to "')[1].split('"')[0]
        if clause in ("Document Name", "Parties", "Agreement Date", "Effective Date"):
            continue
        ans = _lit(r["answers"])
        spans = [
            (s - start, s - start + len(t)) for s, t in zip(ans["answer_start"], ans["text"], strict=True) if start <= s and s + len(t) <= start + MAX_CHARS
        ]
        clauses[clause] = (r["question"].split("Details: ", 1)[-1], spans)
    yield window, clauses


def cuad_locate(split, n, rng, heldout):
    out = []
    for w, (window, clauses) in enumerate(_cuad_windows(split, rng)):
        if len(out) >= n:
            break
        sents = sentences_with_offsets(window)
        present = [c for c, (_, spans) in clauses.items() if spans]
        absent = [c for c, (_, spans) in clauses.items() if not spans]
        pick = rng.choice(absent) if (w % NONE_EVERY == 0 and absent) else (rng.choice(present) if present else None)
        if pick is None:
            continue
        detail, spans = clauses[pick]
        gold = [f"contract_excerpt[{j}]" for j, (s, t) in enumerate(sents) if any(s < b and a < s + len(t) for a, b in spans)]
        state = {"contract_excerpt": [t for _, t in sents]}
        it = locate_item(
            "cuad_locate",
            state,
            f"Which sentence contains the {pick.lower()} provision? ({detail})",
            gold,
            heldout,
            "this part of the contract has no such provision",
        )
        if it:
            out.append(it)
    return out


def cuad_multi(split, n, rng, heldout):
    out = []
    for window, clauses in _cuad_windows(split, rng):
        if len(out) >= n:
            break
        present = [c for c, (_, spans) in clauses.items() if spans]
        absent = [c for c, (_, spans) in clauses.items() if not spans]
        rng.shuffle(present)
        rng.shuffle(absent)
        opts = present[:3] + absent[: 6 - min(3, len(present))]
        if len(opts) < 6:
            continue
        rng.shuffle(opts)
        criteria = {c.lower(): clauses[c][0] for c in opts}
        ref = {c.lower(): float(c in present) for c in opts}
        out.append(
            multi_item("cuad_multi", {"contract_excerpt": window}, "Which of these provisions does this part of the contract contain?", criteria, ref, heldout)
        )
    return out


# ---------------------------------------------------------------- SQuAD 2.0
def squad_locate(split, n, rng, heldout):
    rows = list(load_dataset("rajpurkar/squad_v2", split=split))
    rng.shuffle(rows)
    answerable, unanswerable = [], []
    for r in rows:
        if len(answerable) >= n and len(unanswerable) >= n // NONE_EVERY:
            break
        sents = sentences_with_offsets(r["context"])
        ans = r["answers"]
        if ans["text"]:
            a = ans["answer_start"][0]
            gold = [f"passage[{j}]" for j, (s, t) in enumerate(sents) if s <= a < s + len(t)]
            if len(gold) != 1:
                continue
        else:
            gold = []
        it = locate_item(
            "squad_locate", {"question": r["question"], "passage": [t for _, t in sents]}, "Which sentence of `passage` answers `question`?", gold, heldout
        )
        if it:
            (answerable if gold else unanswerable).append(it)
    k = n // NONE_EVERY
    return answerable[: n - k] + unanswerable[:k]


# ---------------------------------------------------------------- GoEmotions
EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def goemo_multi(split, n, rng, heldout):
    d = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    by_id = collections.defaultdict(list)
    for r in d:
        if not r["example_very_unclear"]:
            by_id[r["id"]].append(r)
    ids = [i for i, rs in by_id.items() if len(rs) >= 3 and (int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0) == (split == "eval")]
    rng.shuffle(ids)
    out = []
    for i in ids[:n]:
        rs = by_id[i]
        ref = {e: round(sum(r[e] for r in rs) / len(rs), 3) for e in EMOTIONS}
        criteria = {e: None for e in EMOTIONS}
        out.append(multi_item("goemo_multi", {"text": rs[0]["text"]}, "Which emotions does the writer of `text` express?", criteria, ref, heldout))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v2_types_train.jsonl")
    ap.add_argument("--eval", default="data/v2_types_eval.jsonl")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    plan = [  # (builder, train split, eval split, train n, eval n)
        (musique_locate, "train", "validation", 1200, 120),
        (wiki2_locate, "train", "validation", 1000, 100),
        (sharc_locate, "train", "validation", 1000, 100),
        (cuad_locate, "train", "test", 800, 80),
        (squad_locate, "train", "validation", 1000, 100),
        (cuad_multi, "train", "test", 1000, 100),
        (goemo_multi, "train", "eval", 2000, 200),
        (sharc_asknext, "train", "validation", 1000, 100),
    ]
    for path, which, heldout in ((args.train, 0, False), (args.eval, 1, True)):
        rng = random.Random(args.seed + which)
        rows = []
        for build, tr, ev, ntr, nev in plan:
            got = build(tr if which == 0 else ev, ntr if which == 0 else nev, rng, heldout)
            print(f"{'train' if which == 0 else 'eval'} {build.__name__:15s} {len(got)}", flush=True)
            rows += got
        # A truncated locate prompt loses candidates, and a truncated anything loses state;
        # measure the rendered prompt, not the characters.
        from transformers import AutoTokenizer

        from s1proto.schema import parse_question
        from s1proto.template import render

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
        before = len(rows)
        rows = [r for r in rows if len(tok(render(r["state"], parse_question(r["question"]), layout="pointer").text).input_ids) <= MAX_TOKENS]
        print(f"dropped {before - len(rows)} rows over {MAX_TOKENS} tokens")
        rng.shuffle(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} to {path}")


if __name__ == "__main__":
    main()
