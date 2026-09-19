"""Build a cold-eval set from public, human-labeled Hugging Face datasets
in our JSONL format (state, question, ref). No teachers: the reference
is the human label (one-hot), or the annotator fraction where the
dataset provides one (civil_comments toxicity).

    uv run python scripts/build_hf_eval.py --n 200 --out data/hf_eval.jsonl

Families:
  clinc_intent      151-way intent incl. `oos` (out of scope)   choice  (needs the trained head; >26 options)
  ticket_queue      support ticket -> one of 12 queues            choice
  ticket_priority   support ticket -> very_low..critical          score
  mnli              premise/hypothesis -> entailment status       choice
  civil_toxic       comment -> toxic? (soft label = annotator %)  noul
  sms_spam          sms -> spam?                                  noul
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

from datasets import load_dataset


def item(family, kind, state, question, ref, heldout=True):
    keys = list(ref.keys())
    z = sum(ref.values()) or 1.0
    ref = {k: v / z for k, v in ref.items()}
    return {
        "id": f"{family}-{hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()[:10]}",
        "family": family,
        "kind": kind,
        "heldout": heldout,
        "state": state,
        "question": question,
        "refs": {"jev": ref, "gemini": ref},  # no teachers here; human label stands in for both
        "ref": {k: ref[k] for k in keys},
        "source": "hf",
    }


def onehot(keys, k):
    return {x: (1.0 if x == k else 0.0) for x in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="data/hf_eval.jsonl")
    ap.add_argument("--split", choices=["eval", "train"], default="eval", help="train draws from the datasets' train splits (disjoint from eval)")
    ap.add_argument("--exclude", default=None, help="eval JSONL whose ids must not appear (safety net for sources without a train split)")
    args = ap.parse_args()
    rng = random.Random(args.seed if args.split == "eval" else args.seed + 1000)
    out = []
    train = args.split == "train"
    excluded = set()
    if args.exclude:
        excluded = {json.loads(line)["id"] for line in open(args.exclude)}

    # --- CLINC OOS (151 intents incl. oos)
    d = load_dataset("clinc/clinc_oos", "plus", split="train" if train else "validation")
    names = d.features["intent"].names
    crit = {n: ("The request does not fit any listed intent" if n == "oos" else None) for n in names}
    q = {"type": "choice", "instructions": "Which intent does this user utterance express? Pick `oos` if none fits.", "criteria": crit}
    idx = rng.sample(range(len(d)), min(args.n, len(d)))
    for i in idx:
        out.append(item("clinc_intent", "choice", d[i]["text"], q, onehot(names, names[d[i]["intent"]])))

    # --- support tickets (English only): queue (choice) and priority (score)
    t = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    en = [i for i in range(len(t)) if t[i]["language"] == "en" and t[i]["subject"] and t[i]["body"]]
    if train:
        en = []  # noisy labels; not worth training on
    queues = [
        "Technical Support",
        "Product Support",
        "Customer Service",
        "IT Support",
        "Billing and Payments",
        "Returns and Exchanges",
        "Service Outages and Maintenance",
        "Sales and Pre-Sales",
        "Human Resources",
        "General Inquiry",
    ]
    prios = ["very_low", "low", "medium", "high", "critical"]
    qq = {"type": "choice", "instructions": "Which support queue should this ticket go to?", "criteria": {k: None for k in queues}}
    qp = {
        "type": "score",
        "instructions": "What priority should this ticket have?",
        "criteria": [
            "Very low: cosmetic, no impact",
            "Low: minor, can wait",
            "Medium: normal request",
            "High: significant impact, needs prompt attention",
            "Critical: outage, security, or data loss",
        ],
    }
    rng.shuffle(en)
    nq = npr = 0
    for i in en:
        row = t[i]
        state = {"subject": row["subject"], "body": row["body"][:1500]}
        if nq < args.n and row["queue"] in queues:
            out.append(item("ticket_queue", "choice", state, qq, onehot(queues, row["queue"])))
            nq += 1
        elif npr < args.n and row["priority"] in prios:
            out.append(item("ticket_priority", "score", state, qp, onehot([str(k) for k in range(5)], str(prios.index(row["priority"])))))
            npr += 1
        if nq >= args.n and npr >= args.n:
            break

    # --- MultiNLI
    m = load_dataset("nyu-mll/multi_nli", split="train" if train else "validation_matched")
    keys = ["entailed", "neutral", "contradicted"]
    lab = {0: "entailed", 1: "neutral", 2: "contradicted"}
    qm = {
        "type": "choice",
        "instructions": "Given `premise`, what is the status of `hypothesis`?",
        "criteria": {
            "entailed": "The hypothesis must be true if the premise is true",
            "neutral": "The premise does not settle it",
            "contradicted": "The hypothesis must be false if the premise is true",
        },
    }
    for i in rng.sample(range(len(m)), args.n):
        if m[i]["label"] < 0:
            continue
        out.append(item("mnli", "choice", {"premise": m[i]["premise"], "hypothesis": m[i]["hypothesis"]}, qm, onehot(keys, lab[m[i]["label"]])))

    # --- Civil Comments: soft label from annotator fraction; balance toxic/non-toxic
    c = load_dataset("google/civil_comments", split="train" if train else "test")
    scan = range(min(len(c), 200_000))
    tox = [i for i in scan if c[i]["toxicity"] >= 0.5]
    non = [i for i in scan if c[i]["toxicity"] < 0.2]
    qc = {
        "type": "noul",
        "instructions": "Would most people consider this comment toxic (rude, disrespectful, or likely to make someone leave the discussion)?",
        "criteria": {"true": "Toxic", "false": "Not toxic"},
    }
    for i in rng.sample(tox, min(args.n // 2, len(tox))) + rng.sample(non, min(args.n - args.n // 2, len(non))):
        p = float(c[i]["toxicity"])
        out.append(item("civil_toxic", "noul", c[i]["text"][:1500], qc, {"yes": p, "no": 1 - p}))

    # --- SMS spam
    s = load_dataset("ucirvine/sms_spam", split="train")
    qs = {
        "type": "noul",
        "instructions": "Is this SMS spam?",
        "criteria": {"true": "Unsolicited promotion, scam, or bulk message", "false": "A normal personal or transactional message"},
    }
    spam = [i for i in range(len(s)) if s[i]["label"] == 1]
    ham = [i for i in range(len(s)) if s[i]["label"] == 0]
    for i in rng.sample(spam, min(args.n // 2, len(spam))) + rng.sample(ham, min(args.n - args.n // 2, len(ham))):
        out.append(item("sms_spam", "noul", s[i]["sms"], qs, {"yes": 1.0 if s[i]["label"] == 1 else 0.0, "no": 0.0 if s[i]["label"] == 1 else 1.0}))

    rng.shuffle(out)
    if excluded:
        before = len(out)
        out = [it for it in out if it["id"] not in excluded]
        print(f"excluded {before - len(out)} items that appear in {args.exclude}")
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter

    print(len(out), "items", dict(Counter(i["family"] for i in out)))


if __name__ == "__main__":
    main()
