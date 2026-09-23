"""Evaluate a scorer against a JSONL set with reference distributions.

Metrics per question type and per family (all vs the averaged reference):
  - kl_to_ref     : mean KL(ref || model)
  - brier         : mean squared error between model and ref distributions
  - accuracy      : model argmax == ref argmax
  - ece           : 15-bin ECE using ref argmax as the label
  - sel_acc_90    : accuracy on the 90% most confident items
  - agree_jev     : model argmax == Jev argmax (distillation view)

Calibration knobs applied at eval time (composable):
  --temps "noul=1.0,choice=0.7,score=0.9"   per-type temperature
  --fit-temps                               grid-search per-type T on half the
                                            items (by id hash), report on the
                                            other half; prints the fitted temps
  --permutations K                          score K option orders, average probs
  --debias                                  subtract content-free label prior

    uv run python scripts/eval_set.py Qwen/Qwen3-8B-Base data/eval.jsonl --out results/set_8B_raw.json
    uv run python scripts/eval_set.py Qwen/Qwen3-8B-Base data/eval.jsonl --permutations 3 --fit-temps
    uv run python scripts/eval_set.py lora:runs/8b-r16 data/eval.jsonl        # Phase 3 checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.teachers import item_keys, option_keys
from s1proto.schema import parse_question
from s1proto.scorer import load_scorer, readout, softmax
from s1proto.service import parse_temperatures
from s1proto.template import render

CONTENT_FREE = ["N/A", "", "[MASK]"]
T_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]


def kl(p: list[float], q: list[float]) -> float:
    eps = 1e-9
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q, strict=True))


def ece15(confs, correct, bins=15):
    n = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0.0)]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return e


def permuted_question(q: dict, rng: random.Random) -> tuple[dict, list[str]]:
    """Choice and multi: shuffle option order. Score levels are ordered by meaning,
    Noul is fixed yes/no, and locate's candidates are the state itself, so those keep
    their order."""
    if q["type"] not in ("choice", "multi"):
        raise ValueError("only choice and multi options can be reordered")
    keys = option_keys(q)
    order = keys[:]
    rng.shuffle(order)
    return {**q, "criteria": {k: q["criteria"][k] for k in order}}, order


def collect_logits(scorer, items: list[dict], permutations: int, debias: bool, batch: int, seed: int) -> list[list[list[float]]]:
    """Per item: a list (one per permutation pass) of canonical-order
    label logits (debiased if requested). Temperature is applied later
    so fitting is free."""
    rng = random.Random(seed)
    per_item: list[list[list[float]]] = [[] for _ in items]
    for perm in range(max(1, permutations)):
        prompts, meta = [], []
        for i, it in enumerate(items):
            q = it["question"]
            keys = item_keys(it)
            qp, order = permuted_question(q, rng) if perm > 0 and q["type"] in ("choice", "multi") else (q, keys)
            prompts.append(render(it["state"], parse_question(qp), layout=getattr(scorer, "layout", "letters")))
            meta.append((i, keys, order, qp))
        priors: dict[str, list[float]] = {}
        if debias:
            for _, _, order, qp in meta:
                key = json.dumps(qp, sort_keys=True)
                if key not in priors:
                    ps = [render(cf, parse_question(qp), layout=getattr(scorer, "layout", "letters")) for cf in CONTENT_FREE]
                    rs = scorer.score(ps, [1.0] * len(ps))
                    priors[key] = [sum(r.logits[j] for r in rs) / len(rs) for j in range(len(order))]
        for start in range(0, len(prompts), batch):
            rs = scorer.score(prompts[start : start + batch], None)
            for (i, keys, order, qp), r in zip(meta[start : start + batch], rs, strict=True):
                logits = list(r.logits)
                if debias:
                    pr = priors[json.dumps(qp, sort_keys=True)]
                    logits = [x - y for x, y in zip(logits, pr, strict=True)]
                if it_kind(items[i]) == "match":  # never permuted: one row of option logits per item, in order
                    per_item[i].append(logits)
                    continue
                by_key = dict(zip(order, logits, strict=True))
                per_item[i].append([by_key[k] for k in keys])
    return per_item


def it_kind(it: dict) -> str:
    return it["question"]["type"]


def probs_from_logits(per_item, items, temps) -> list[list[float]]:
    out = []
    for vecs, it in zip(per_item, items, strict=True):
        t = temps.get(it["kind"], 1.0)
        ps = [readout(v, it["kind"], t, len(item_keys(it))) for v in vecs]
        out.append([sum(p[j] for p in ps) / len(ps) for j in range(len(ps[0]))])
    return out


def summarize(items: list[dict], preds: list[list[float]]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups["all"].append(i)
        groups[f"type:{it['kind']}"].append(i)
        groups[f"family:{it['family']}" + (" (heldout)" if it.get("heldout") else "")].append(i)

    def metrics(idx: list[int]) -> dict:
        v2 = {k: [i for i in idx if items[i]["kind"] == k] for k in ("multi", "locate", "rank", "match")}
        idx = [i for i in idx if items[i]["kind"] not in v2]
        out = {k: v2_metrics(k, v) for k, v in v2.items() if v}
        if not idx:  # a v2-only group: its headline numbers are the v2 type's own
            return next(iter(out.values()))
        return {**one_answer(idx), **out}

    def v2_metrics(kind: str, idx: list[int]) -> dict:
        if kind in ("rank", "match"):
            return (rank_metrics if kind == "rank" else match_metrics)(idx)
        confs, correct, briers, hits, top3 = [], [], [], [], []
        tp = fp = fn = 0
        for i in idx:
            it, p = items[i], preds[i]
            ref = [it["ref"].get(k, 0.0) for k in item_keys(it)]
            briers.append(sum((a - b) ** 2 for a, b in zip(p, ref, strict=True)) / (len(p) if kind == "multi" else 1))
            if kind == "multi":
                pred, gold = [x >= 0.5 for x in p], [x >= 0.5 for x in ref]
                confs += [max(x, 1 - x) for x in p]
                correct += [a == b for a, b in zip(pred, gold, strict=True)]
                hits.append(pred == gold)
                tp += sum(a and b for a, b in zip(pred, gold, strict=True))
                fp += sum(a and not b for a, b in zip(pred, gold, strict=True))
                fn += sum(b and not a for a, b in zip(pred, gold, strict=True))
            else:
                order = sorted(range(len(p)), key=lambda j: -p[j])
                confs.append(p[order[0]])
                correct.append(ref[order[0]] > 0)
                hits.append(ref[order[0]] > 0)
                top3.append(any(ref[j] > 0 for j in order[:3]))
        n = len(idx)
        out = {"n": n, "accuracy": round(sum(hits) / n, 4), "ece": round(ece15(confs, correct), 4), "brier": round(sum(briers) / n, 4)}
        if kind == "multi":
            out["f1"] = round(2 * tp / max(1, 2 * tp + fp + fn), 4)
            out["accuracy_note"] = "exact set; ece and f1 are per option"
        else:
            out["top3"] = round(sum(top3) / n, 4)
            out["accuracy_note"] = "top pick is a gold candidate"
        return out

    def rank_metrics(idx: list[int]) -> dict:
        """Per pair of differently graded options: is the better one more likely first, and how
        calibrated is P(a above b) = p_a / (p_a + p_b). Top1: the first pick has the top grade."""
        confs, correct, top1 = [], [], []
        for i in idx:
            grade, p = [items[i]["ref"].get(k, 0.0) for k in item_keys(items[i])], preds[i]
            for a in range(len(p)):
                for b in range(len(p)):
                    if grade[a] > grade[b]:
                        pab = p[a] / max(p[a] + p[b], 1e-12)
                        confs.append(max(pab, 1 - pab))
                        correct.append(pab > 0.5)
            top1.append(grade[max(range(len(p)), key=p.__getitem__)] == max(grade))
        n = len(idx)
        return {
            "n": n,
            "accuracy": round(sum(top1) / n, 4),
            "pair_acc": round(sum(correct) / max(1, len(correct)), 4),
            "ece": round(ece15(confs, correct), 4),
            "accuracy_note": "top pick has the top grade; ece is per pair",
        }

    def match_metrics(idx: list[int]) -> dict:
        confs, correct, none_hits = [], [], []
        for i in idx:
            it, p = items[i], preds[i]
            keys = item_keys(it)
            for r, name in enumerate(it["question"]["items"]):
                row = p[r * len(keys) : (r + 1) * len(keys)]
                gold = it["ref"][name]
                pick = keys[max(range(len(row)), key=row.__getitem__)]
                confs.append(max(row))
                correct.append(gold.get(pick, 0.0) >= 0.5)
                if gold.get("none", 0.0) >= 0.5:
                    none_hits.append(correct[-1])
        n = len(correct)
        return {
            "n": len(idx),
            "items": n,
            "accuracy": round(sum(correct) / n, 4),
            "accuracy_none": round(sum(none_hits) / max(1, len(none_hits)), 4),
            "ece": round(ece15(confs, correct), 4),
            "accuracy_note": "per item; ece per item",
        }

    def one_answer(idx: list[int]) -> dict:
        kls, briers, confs, correct, agree = [], [], [], [], []
        for i in idx:
            it = items[i]
            keys = option_keys(it["question"])
            ref = [it["ref"][k] for k in keys]
            jev = [it["refs"].get("jev", it["ref"])[k] for k in keys]  # no Jev reference on code-labeled sets
            p = preds[i]
            kls.append(kl(ref, p))
            briers.append(sum((a - b) ** 2 for a, b in zip(p, ref, strict=True)))
            am_p = max(range(len(p)), key=p.__getitem__)
            am_r = max(range(len(ref)), key=ref.__getitem__)
            am_j = max(range(len(jev)), key=jev.__getitem__)
            confs.append(max(p))
            correct.append(am_p == am_r)
            agree.append(am_p == am_j)
        n = len(idx)
        order = sorted(range(n), key=lambda k: -confs[k])
        top = order[: max(1, int(round(0.9 * n)))]
        return {
            "n": n,
            "kl_to_ref": round(sum(kls) / n, 4),
            "brier": round(sum(briers) / n, 4),
            "accuracy": round(sum(correct) / n, 4),
            "ece": round(ece15(confs, correct), 4),
            "sel_acc_90": round(sum(correct[k] for k in top) / len(top), 4),
            "agree_jev": round(sum(agree) / n, 4),
            "mean_conf": round(sum(confs) / n, 4),
        }

    return {g: metrics(idx) for g, idx in groups.items()}


def fit_temperatures(per_item, items, fit_idx: list[int]) -> dict[str, float]:
    """Per type, the grid temperature minimizing mean KL(ref||p) on fit_idx."""
    temps = {}
    for kind in ("noul", "choice", "score"):
        idx = [i for i in fit_idx if items[i]["kind"] == kind]
        if not idx:
            temps[kind] = 1.0
            continue
        best_t, best = 1.0, float("inf")
        for t in T_GRID:
            tot = 0.0
            for i in idx:
                keys = option_keys(items[i]["question"])
                ref = [items[i]["ref"][k] for k in keys]
                ps = [softmax([x / t for x in v]) for v in per_item[i]]
                p = [sum(q[j] for q in ps) / len(ps) for j in range(len(keys))]
                tot += kl(ref, p)
            if tot < best:
                best, best_t = tot, t
        temps[kind] = best_t
    return temps


def split_by_hash(items: list[dict]) -> tuple[list[int], list[int]]:
    a, b = [], []
    for i, it in enumerate(items):
        (a if int(hashlib.sha1(it["id"].encode()).hexdigest(), 16) % 2 == 0 else b).append(i)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--temps", default=None)
    ap.add_argument("--fit-temps", action="store_true")
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--debias", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--families", default=None, help="comma-separated families to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-logits", default=None)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.data)]
    if args.families:
        keep = set(args.families.split(","))
        items = [it for it in items if it["family"] in keep]
    if args.limit:
        items = items[: args.limit]
    scorer = load_scorer(args.model)
    t0 = time.perf_counter()
    per_item = collect_logits(scorer, items, args.permutations, args.debias, args.batch, args.seed)
    elapsed = time.perf_counter() - t0
    # tokens actually scored (state + question tail) for a throughput figure
    n_tokens = 0
    try:
        tok = scorer.tokenizer
        for it in items:
            n_tokens += len(
                tok.encode(render(it["state"], parse_question(it["question"]), layout=getattr(scorer, "layout", "letters")).text, add_special_tokens=False)
            )
    except AttributeError:
        n_tokens = 0
    timing = {
        "device": getattr(scorer, "device", None),
        "items": len(items),
        "passes": max(1, args.permutations),
        "seconds": round(elapsed, 1),
        "ms_per_item": round(1000 * elapsed / (len(items) * max(1, args.permutations)), 1),
        "input_tokens": n_tokens,
        "tokens_per_s": round(n_tokens * max(1, args.permutations) / elapsed, 1) if elapsed else None,
        "batch": args.batch,
    }
    print("timing:", json.dumps(timing))
    if args.dump_logits:
        with open(args.dump_logits, "w") as f:
            for it, v in zip(items, per_item, strict=True):
                f.write(json.dumps({"id": it["id"], "kind": it["kind"], "family": it["family"], "logits": v}) + "\n")

    temps = parse_temperatures(args.temps)
    if args.temps is None and getattr(scorer, "temperatures", None):
        temps = {**temps, **scorer.temperatures}  # the run's own fitted temperatures, what the API serves
    report_items, report_logits = items, per_item
    fitted = None
    if args.fit_temps:
        fit_idx, test_idx = split_by_hash(items)
        fitted = fit_temperatures(per_item, items, fit_idx)
        temps = fitted
        report_items = [items[i] for i in test_idx]
        report_logits = [per_item[i] for i in test_idx]
        print(f"fitted temps on {len(fit_idx)} items: {fitted}; reporting on the other {len(test_idx)}")

    preds = probs_from_logits(report_logits, report_items, temps)
    summary = summarize(report_items, preds)
    result = {
        "model": getattr(scorer, "name", args.model),
        "data": args.data,
        "temps": temps,
        "fitted": fitted,
        "permutations": args.permutations,
        "debias": args.debias,
        "timing": timing,
        "metrics": summary,
    }
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("family:")}, indent=1))
    print("families:")
    for k, v in summary.items():
        if k.startswith("family:"):
            extra = (
                f"kl={v['kl_to_ref']:.3f} agree_jev={v['agree_jev']:.3f}"
                if "kl_to_ref" in v
                else " ".join(f"{m}={v[m]:.3f}" for m in ("brier", "f1", "top3", "pair_acc", "accuracy_none") if m in v)
            )
            print(f"  {k:<44} n={v['n']:<4} acc={v['accuracy']:.3f} ece={v['ece']:.3f} {extra}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
