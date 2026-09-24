"""How a model answers user-defined option lists: top-1, and how it uses "other".

    uv run python scripts/eval_open_taxonomy.py lora:runs/circuit-1.7b-v2.0 data/open_tax_eval.jsonl data/open_tax_tagger_eval.jsonl

Per family:
  top1          argmax == the reference label
  other_rate    share of answers that are the catch-all ("other", "none", "not_listed", ...)
  false_other   share of rows whose answer IS in the list where the model said "other" anyway
  other_recall  share of rows whose answer is NOT in the list where the model said "other"
  conf          mean top probability, to read beside top1

tagger_ref_* rows score agreement with Jev's tags: a proxy for a user taxonomy, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_set import collect_logits, probs_from_logits

from s1proto.data.teachers import item_keys
from s1proto.scorer import load_scorer
from s1proto.service import parse_temperatures

OTHER = {"other", "none", "none_of_these", "not_listed"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="rows per file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = []
    for path in args.data:
        rows = [json.loads(line) for line in open(path)]
        items += rows[: args.limit] if args.limit else rows
    items = [it for it in items if it["kind"] == "choice"]
    scorer = load_scorer(args.model)
    temps = {**parse_temperatures(None), **(getattr(scorer, "temperatures", None) or {})}
    probs = probs_from_logits(collect_logits(scorer, items, 1, False, args.batch, 0), items, temps)

    by: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for it, p in zip(items, probs, strict=True):
        keys = item_keys(it)
        pred = keys[max(range(len(p)), key=p.__getitem__)]
        gold = max(it["ref"], key=it["ref"].get)
        for group in {"all", "tagger" if it["family"].startswith("tagger") else it["family"], it["family"]}:
            by[group].append((pred, gold, max(p)))
    result = {}
    for g, rs in sorted(by.items()):
        present = [r for r in rs if r[1] not in OTHER]
        absent = [r for r in rs if r[1] in OTHER]
        result[g] = {
            "n": len(rs),
            "top1": round(sum(p == y for p, y, _ in rs) / len(rs), 3),
            "other_rate": round(sum(p in OTHER for p, _, _ in rs) / len(rs), 3),
            "false_other": round(sum(p in OTHER for p, _, _ in present) / len(present), 3) if present else None,
            "other_recall": round(sum(p in OTHER for p, _, _ in absent) / len(absent), 3) if absent else None,
            "conf": round(sum(c for _, _, c in rs) / len(rs), 3),
        }
        print(f"{g:26} " + "  ".join(f"{k} {v}" for k, v in result[g].items()))
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "families": result}, indent=1))


if __name__ == "__main__":
    main()
