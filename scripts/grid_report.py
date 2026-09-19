"""Operations x formats table from grid eval results.

    uv run python scripts/grid_report.py results/grid_jev.json results/grid_1.7B_raw.json results/grid_lora_1.7B_real.json
    uv run python scripts/grid_report.py results/grid_*.json --metric ece
    uv run python scripts/grid_report.py results/grid_*.json --md >> docs/grid.md

One block per result file: a 9x6 table of accuracy (or ECE) per cell,
with row and column means, so a weakness shows up as a row (an
operation the model can't do) or a column (a format it can't read).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.grid import FORMATS, OPERATIONS


def table(metrics: dict, metric: str) -> tuple[dict[str, dict[str, float | None]], dict[str, float], dict[str, float]]:
    cells: dict[str, dict[str, float | None]] = {op: {} for op in OPERATIONS}
    for op in OPERATIONS:
        for fmt in FORMATS:
            m = metrics.get(f"family:{op}/{fmt} (heldout)") or metrics.get(f"family:{op}/{fmt}")
            cells[op][fmt] = m[metric] if m else None
    row = {op: _mean([v for v in cells[op].values() if v is not None]) for op in OPERATIONS}
    col = {fmt: _mean([cells[op][fmt] for op in OPERATIONS if cells[op][fmt] is not None]) for fmt in FORMATS}
    return cells, row, col


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fmt_v(v: float | None, metric: str) -> str:
    if v is None:
        return "  -  "
    return f"{v * 100:4.0f}%" if metric == "accuracy" else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "ece", "brier", "kl_to_ref"])
    ap.add_argument("--md", action="store_true", help="markdown tables")
    args = ap.parse_args()
    for path in args.results:
        d = json.load(open(path))
        m = d["metrics"]
        cells, row, col = table(m, args.metric)
        overall = m["all"][args.metric]
        t = d.get("timing", {})
        ms = t.get("ms_per_item") or t.get("ms_per_item_p50")
        title = f"{d.get('model', path)}  {args.metric} overall {fmt_v(overall, args.metric).strip()}" + (f", {ms:.0f} ms/item" if ms else "")
        if args.md:
            print(f"\n**{title}**\n")
            print("| operation | " + " | ".join(FORMATS) + " | mean |")
            print("|---|" + "---|" * (len(FORMATS) + 1))
            for op in OPERATIONS:
                print(f"| {op} | " + " | ".join(fmt_v(cells[op][f], args.metric).strip() for f in FORMATS) + f" | {fmt_v(row[op], args.metric).strip()} |")
            print("| **mean** | " + " | ".join(fmt_v(col[f], args.metric).strip() for f in FORMATS) + " | |")
        amb = {k[7 + len("ambiguous/") :]: v for k, v in m.items() if k.startswith("family:ambiguous/")}
        if args.md and amb:
            print(
                "\nAmbiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): "
                + ", ".join(f"{op} conf {v['mean_conf']:.2f} ece {v['ece']:.2f}" for op, v in sorted(amb.items()))
            )
        if not args.md:
            print(f"\n{title}")
            print(f"{'':<12}" + "".join(f"{f:>10}" for f in FORMATS) + f"{'mean':>10}")
            for op in OPERATIONS:
                print(f"{op:<12}" + "".join(f"{fmt_v(cells[op][f], args.metric):>10}" for f in FORMATS) + f"{fmt_v(row[op], args.metric):>10}")
            print(f"{'mean':<12}" + "".join(f"{fmt_v(col[f], args.metric):>10}" for f in FORMATS))
            if amb:
                print("ambiguous (ideal conf ~0): " + "  ".join(f"{op} conf={v['mean_conf']:.2f}" for op, v in sorted(amb.items())))


if __name__ == "__main__":
    main()
