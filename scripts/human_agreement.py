"""Score human labels against the teachers and against model predictions.

Input: a JSON list of label records as written by the labeling page:
  {item, family, kind, answer, unsure, ts}
(export them from the artifact db into a file first).

Reports, over labeled items (and separately over the subset where the
two teachers disagree):
  - human agreement with Jev, with Gemini, with the averaged reference
  - for each --preds file (JSONL of {id, pred:[...]} in canonical key
    order, e.g. results/logits_*.jsonl or a preds dump): agreement with
    the human, i.e. accuracy against a person instead of a teacher
  - teacher calibration against the human: ECE of Jev's and Gemini's
    stated max-probability vs whether they matched the human

    uv run python scripts/human_agreement.py results/human_labels.json --preds results/logits_8B_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.teachers import option_keys


def ece15(confs, correct, bins=15):
    n = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0.0)]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return e


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--preds", action="append", default=[], help="JSONL with id + pred (probs) or logits")
    args = ap.parse_args()

    raw = open(args.labels).read()
    if raw.lstrip().startswith("S1LABELS"):
        # compact export from the labeling page: "<id>\t<answer>[\tunsure]"
        labels = []
        for line in raw.splitlines()[1:]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            labels.append({"item": parts[0].split("@")[0], "v": int(parts[0].split("@")[1]) if "@" in parts[0] else 1, "answer": None if parts[1] == "-" else parts[1], "unsure": len(parts) > 2 and parts[2] == "unsure"})
    else:
        labels = json.loads(raw)
        if isinstance(labels, dict):
            labels = list(labels.values())
    human = {r["item"]: r for r in labels if r.get("answer")}
    items = {}
    for line in open(args.eval):
        it = json.loads(line)
        if it["id"] in human:
            items[it["id"]] = it
    print(f"labeled: {len(human)}  (unsure: {sum(1 for r in human.values() if r.get('unsure'))})")

    def am(d):
        return max(d, key=d.get)

    def report(ids, title):
        if not ids:
            return
        aj = sum(am(items[i]["refs"]["jev"]) == human[i]["answer"] for i in ids) / len(ids)
        ag = sum(am(items[i]["refs"]["gemini"]) == human[i]["answer"] for i in ids) / len(ids)
        ar = sum(am(items[i]["ref"]) == human[i]["answer"] for i in ids) / len(ids)
        both = sum(am(items[i]["refs"]["jev"]) == am(items[i]["refs"]["gemini"]) == human[i]["answer"] for i in ids) / len(ids)
        print(f"\n== {title} (n={len(ids)})")
        print(f"  human agrees with: Jev {aj:.3f}   Gemini {ag:.3f}   averaged ref {ar:.3f}   both-teachers-when-they-agree {both:.3f}")
        for name in ("jev", "gemini"):
            confs = [max(items[i]["refs"][name].values()) for i in ids]
            corr = [am(items[i]["refs"][name]) == human[i]["answer"] for i in ids]
            print(f"  {name:6} calibration vs human: mean stated conf {sum(confs) / len(confs):.3f}  hit rate {sum(corr) / len(corr):.3f}  ECE {ece15(confs, corr):.3f}")
        for kind in ("noul", "choice", "score"):
            k_ids = [i for i in ids if items[i]["kind"] == kind]
            if k_ids:
                a = sum(am(items[i]["ref"]) == human[i]["answer"] for i in k_ids) / len(k_ids)
                print(f"  by type {kind:6} n={len(k_ids):<3} human vs ref {a:.3f}")

    all_ids = list(items)
    dis_ids = [i for i in all_ids if am(items[i]["refs"]["jev"]) != am(items[i]["refs"]["gemini"])]
    agr_ids = [i for i in all_ids if i not in set(dis_ids)]
    report(all_ids, "all labeled")
    report(agr_ids, "teachers agree")
    report(dis_ids, "teachers disagree")
    if dis_ids:
        jw = sum(am(items[i]["refs"]["jev"]) == human[i]["answer"] for i in dis_ids)
        gw = sum(am(items[i]["refs"]["gemini"]) == human[i]["answer"] for i in dis_ids)
        print(f"  tiebreaks: human sided with Jev {jw}, with Gemini {gw}, with neither {len(dis_ids) - jw - gw}")

    for pf in args.preds:
        preds = {}
        for line in open(pf):
            d = json.loads(line)
            if "pred" in d:
                preds[d["id"]] = d["pred"]
            elif "logits" in d:
                v = d["logits"][0] if isinstance(d["logits"][0], list) else d["logits"]
                preds[d["id"]] = softmax(v)
        ids = [i for i in all_ids if i in preds]
        if not ids:
            continue
        keys = {i: option_keys(items[i]["question"]) for i in ids}
        corr = [keys[i][max(range(len(preds[i])), key=preds[i].__getitem__)] == human[i]["answer"] for i in ids]
        confs = [max(preds[i]) for i in ids]
        print(f"\n== model {pf} (n={len(ids)})")
        print(f"  agreement with human {sum(corr) / len(corr):.3f}   mean conf {sum(confs) / len(confs):.3f}   ECE vs human {ece15(confs, corr):.3f}")
        d_ids = [k for k, i in enumerate(ids) if i in set(dis_ids)]
        if d_ids:
            print(f"  on teacher-disagreement items: agreement with human {sum(corr[k] for k in d_ids) / len(d_ids):.3f}")


if __name__ == "__main__":
    main()
