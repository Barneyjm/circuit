"""Build the generalization grid: (operation x format) cells with
code-computed labels. See s1proto/data/grid.py for the taxonomy.

    uv run python scripts/build_grid.py --per-cell 150 --eval-per-cell 40
    uv run python scripts/build_grid.py --holdout "temporal/*,*/thread,consistency/document"

`--holdout` names cells (globs on op/format) that go to the eval file
only, for leave-cells-out training. Everything is always in eval.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.grid import cell_of, cells, generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--eval-per-cell", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--holdout", default="", help="comma-separated cell globs kept out of training, e.g. 'temporal/*,*/thread'")
    ap.add_argument("--train", default="data/grid_train.jsonl")
    ap.add_argument("--eval", default="data/grid_eval.jsonl")
    args = ap.parse_args()

    holdout = {c for c in cells() if any(fnmatch.fnmatch(c, g.strip()) for g in args.holdout.split(",") if g.strip())}
    train = [r for r in generate(args.per_cell, args.seed, "train") if cell_of(r) not in holdout]
    ev = generate(args.eval_per_cell, args.seed, "eval")
    train_ids = {r["id"] for r in train}
    ev = [r for r in ev if r["id"] not in train_ids]
    for path, rows in ((args.train, train), (args.eval, ev)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train {len(train)} items, {len(Counter(cell_of(r) for r in train))} cells (held out: {sorted(holdout) or 'none'}) -> {args.train}")
    print(f"eval  {len(ev)} items, {len(Counter(cell_of(r) for r in ev))} cells -> {args.eval}")
    print("kinds:", dict(Counter(r["kind"] for r in train)), "| ambiguous:", sum(r["ambiguous"] for r in train))


if __name__ == "__main__":
    main()
