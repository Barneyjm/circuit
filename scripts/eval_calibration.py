"""Quick calibration check on public labeled data (Phase 1, step 1:
"confirm with a quick ECE check on 200 items before committing").

Two task families with hard labels:
  - sst2   : Noul  "Is this movie review positive?"      (GLUE SST-2 validation)
  - ag_news: Choice 4-way topic {world, sports, business, sci/tech}

Reports accuracy, ECE (15 bins), Brier, mean confidence, and the
label-position bias (how often each letter wins), per model. Also
runs a permutation check on ag_news: fraction of items whose argmax
changes when option order is reversed.

    uv run python scripts/eval_calibration.py Qwen/Qwen3-0.6B-Base --n 200
    uv run python scripts/eval_calibration.py Qwen/Qwen3-0.6B-Base --n 200 --temperature 1.6
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

from s1proto.schema import ChoiceQuestion, NoulQuestion
from s1proto.scorer import HFScorer
from s1proto.template import render

AG_LABELS = ["world", "sports", "business", "sci_tech"]
AG_CRITERIA = {"world": "World news, politics, international affairs", "sports": "Sports", "business": "Business, markets, companies, economy", "sci_tech": "Science and technology"}


def ece(confs: list[float], correct: list[bool], bins: int = 15) -> float:
    total = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += len(idx) / total * abs(acc - conf)
    return e


def brier(prob_vectors: list[list[float]], truth_idx: list[int]) -> float:
    s = 0.0
    for probs, t in zip(prob_vectors, truth_idx, strict=True):
        s += sum((p - (1.0 if i == t else 0.0)) ** 2 for i, p in enumerate(probs))
    return s / len(prob_vectors)


def batched(xs, size):
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def load_items(n: int, seed: int):
    from datasets import load_dataset

    rng = random.Random(seed)
    sst = load_dataset("nyu-mll/glue", "sst2", split="validation")
    sst_idx = rng.sample(range(len(sst)), min(n, len(sst)))
    sst_items = [(sst[i]["sentence"], int(sst[i]["label"])) for i in sst_idx]  # 1 = positive

    ag = load_dataset("fancyzhx/ag_news", split="test")
    ag_idx = rng.sample(range(len(ag)), min(n, len(ag)))
    ag_items = [(ag[i]["text"], int(ag[i]["label"])) for i in ag_idx]  # 0 world,1 sports,2 business,3 sci/tech
    return sst_items, ag_items


CONTENT_FREE_STATES = ["N/A", "", "[MASK]"]


def content_free_prior(scorer: HFScorer, q) -> list[float]:
    """Mean label logits over content-free states for this question."""
    prompts = [render(cf, q) for cf in CONTENT_FREE_STATES]
    rs = scorer.score(prompts, [1.0] * len(prompts))
    n = len(rs[0].logits)
    return [sum(r.logits[i] for r in rs) / len(rs) for i in range(n)]


def run_noul(scorer: HFScorer, items, temperature: float, batch: int, debias: bool = False):
    q = NoulQuestion(type="noul", instructions="Is this movie review positive?", criteria={"true": "The reviewer liked the film", "false": "The reviewer disliked the film"})
    prompts = [render(text, q) for text, _ in items]
    results = []
    for chunk in batched(prompts, batch):
        results.extend(scorer.score(chunk, [temperature] * len(chunk)))
    truth = [t for _, t in items]
    logits = [r.logits for r in results]
    if debias:
        prior = content_free_prior(scorer, q)
        logits = [[x - pr for x, pr in zip(lg, prior, strict=True)] for lg in logits]
    summary = _noul_metrics(logits, truth, temperature)
    summary["temperature_sweep"] = {str(t): {k: round(v, 4) for k, v in _noul_metrics(logits, truth, t).items() if k in ("ece", "brier", "accuracy")} for t in SWEEP}
    return summary


SWEEP = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]


def _softmax(xs, t):
    m = max(xs)
    es = [math.exp((x - m) / t) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def _noul_metrics(logits, truth, t):
    p_yes = [_softmax(lg, t)[0] for lg in logits]
    pred = [1 if p >= 0.5 else 0 for p in p_yes]
    correct = [p == tr for p, tr in zip(pred, truth, strict=True)]
    confs = [max(p, 1 - p) for p in p_yes]
    return {
        "n": len(truth),
        "accuracy": sum(correct) / len(truth),
        "ece": ece(confs, correct),
        "brier": brier([[p, 1 - p] for p in p_yes], [0 if tr == 1 else 1 for tr in truth]),
        "mean_confidence": sum(confs) / len(confs),
        "yes_rate_pred": sum(pred) / len(pred),
        "yes_rate_true": sum(truth) / len(truth),
    }


def run_choice(scorer: HFScorer, items, temperature: float, batch: int, reverse: bool = False, debias: bool = False):
    labels = list(reversed(AG_LABELS)) if reverse else AG_LABELS
    crit = {k: AG_CRITERIA[k] for k in labels}
    q = ChoiceQuestion(type="choice", instructions="What is the topic of this news article?", criteria=crit)
    prompts = [render(text, q) for text, _ in items]
    results = []
    for chunk in batched(prompts, batch):
        results.extend(scorer.score(chunk, [temperature] * len(chunk)))
    if debias:
        prior = content_free_prior(scorer, q)
        for r in results:
            r.logits = [x - pr for x, pr in zip(r.logits, prior, strict=True)]
    truth = [t for _, t in items]
    # canonical-order logits so a temperature sweep is free
    logits = []
    for r in results:
        by_name = dict(zip(labels, r.logits, strict=True))
        logits.append([by_name[k] for k in AG_LABELS])
    summary, pred = _choice_metrics(logits, truth, temperature)
    # position bias: which *letter* (position) won, in the order shown
    pos_winner = Counter(max(range(4), key=lambda i: r.logits[i]) for r in results)
    summary["position_winner_rate"] = {f"pos{k}": round(v / len(items), 3) for k, v in sorted(pos_winner.items())}
    summary["temperature_sweep"] = {str(t): {k: round(v, 4) for k, v in _choice_metrics(logits, truth, t)[0].items() if k in ("ece", "brier", "accuracy")} for t in SWEEP}
    return summary, pred


def _choice_metrics(logits, truth, t):
    prob_vectors = [_softmax(lg, t) for lg in logits]
    pred = [max(range(4), key=lambda i: pv[i]) for pv in prob_vectors]
    correct = [p == tr for p, tr in zip(pred, truth, strict=True)]
    confs = [max(pv) for pv in prob_vectors]
    return {
        "n": len(truth),
        "accuracy": sum(correct) / len(truth),
        "ece": ece(confs, correct),
        "brier": brier(prob_vectors, truth),
        "mean_confidence": sum(confs) / len(confs),
        "pred_label_rate": {AG_LABELS[k]: round(v / len(truth), 3) for k, v in sorted(Counter(pred).items())},
        "true_label_rate": {AG_LABELS[k]: round(v / len(truth), 3) for k, v in sorted(Counter(truth).items())},
    }, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--debias", action="store_true", help="subtract content-free label prior (contextual calibration)")
    args = ap.parse_args()

    scorer = HFScorer(args.model)
    sst_items, ag_items = load_items(args.n, args.seed)

    noul = run_noul(scorer, sst_items, args.temperature, args.batch, debias=args.debias)
    choice, pred_fwd = run_choice(scorer, ag_items, args.temperature, args.batch, debias=args.debias)
    choice_rev, pred_rev = run_choice(scorer, ag_items, args.temperature, args.batch, reverse=True, debias=args.debias)
    order_flip = sum(a != b for a, b in zip(pred_fwd, pred_rev, strict=True)) / len(pred_fwd)

    out = {
        "model": scorer.name,
        "temperature": args.temperature,
        "debias": args.debias,
        "sst2_noul": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in noul.items()},
        "ag_news_choice": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in choice.items()},
        "ag_news_choice_reversed_order": {"accuracy": round(choice_rev["accuracy"], 4), "ece": round(choice_rev["ece"], 4)},
        "order_sensitivity_argmax_flip_rate": round(order_flip, 4),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    assert math.isfinite(out["sst2_noul"]["ece"])


if __name__ == "__main__":
    main()
