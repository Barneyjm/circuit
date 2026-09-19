"""Shuffle the `candidates` list inside pick_* states and remap the
reference keys accordingly.

The generator listed candidates in document order, so for pick_total
the answer was the last candidate in ~85% of items (totals come last on
invoices). The question refers to candidates by index, so permuting the
list and renaming c<i> -> c<perm(i)> in every distribution is exact; no
re-labeling needed.

    uv run python scripts/fix_candidate_order.py data/train.jsonl data/eval.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

FAMILIES = {"pick_total", "pick_meeting_time"}


def remap(dist: dict[str, float], perm: dict[str, str]) -> dict[str, float]:
    return {perm.get(k, k): v for k, v in dist.items()}


def main() -> None:
    rng = random.Random(1234)
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path)]
        before, after = Counter(), Counter()
        changed = 0
        for it in rows:
            if it["family"] not in FAMILIES or not isinstance(it["state"], dict):
                continue
            cands = it["state"].get("candidates")
            if not isinstance(cands, list) or len(cands) < 2:
                continue
            n = len(cands)
            order = list(range(n))
            rng.shuffle(order)  # new position j holds old index order[j]
            perm = {f"c{old}": f"c{new}" for new, old in enumerate(order)}
            a = max(it["ref"], key=it["ref"].get)
            before["last" if a == f"c{n - 1}" else ("none" if a == "none" else "other")] += 1
            it["state"]["candidates"] = [cands[old] for old in order]
            it["ref"] = remap(it["ref"], perm)
            it["refs"] = {t: remap(d, perm) for t, d in it["refs"].items()}
            a = max(it["ref"], key=it["ref"].get)
            after["last" if a == f"c{n - 1}" else ("none" if a == "none" else "other")] += 1
            changed += 1
        with open(path, "w") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{path}: reshuffled {changed} items; answer position before {dict(before)} -> after {dict(after)}")


if __name__ == "__main__":
    main()
