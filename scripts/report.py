"""Tabulate results/set_*.json into markdown for the writeups.

uv run python scripts/report.py results/set_8B_raw.json results/set_8B_temps.json ...
"""

from __future__ import annotations

import json
import sys


def row(label: str, m: dict) -> str:
    return f"| {label} | {m['accuracy']:.3f} | {m['ece']:.3f} | {m['kl_to_ref']:.3f} | {m['brier']:.3f} | {m['sel_acc_90']:.3f} | {m['agree_jev']:.3f} |"


def main() -> None:
    files = sys.argv[1:]
    runs = []
    for f in files:
        d = json.load(open(f))
        tag = d["model"]
        if d.get("permutations", 1) > 1:
            tag += f" perm{d['permutations']}"
        if d.get("debias"):
            tag += " debias"
        if d.get("fitted"):
            tag += " T=" + ",".join(f"{k[0]}{v}" for k, v in d["fitted"].items())
        runs.append((tag, d["metrics"]))

    hdr = "| run | acc | ECE | KL | Brier | sel@90 | agree Jev |\n|---|---|---|---|---|---|---|"
    for group in ["all", "type:noul", "type:choice", "type:score"]:
        print(f"\n### {group}\n\n{hdr}")
        for tag, m in runs:
            print(row(tag, m[group]))

    # held-out vs train families, aggregated by n-weighted mean
    print("\n### held-out families vs training families (n-weighted means)\n")
    print("| run | train-fam acc | train-fam ECE | heldout acc | heldout ECE | heldout KL |\n|---|---|---|---|---|---|")
    for tag, m in runs:
        agg = {"train": [0, 0.0, 0.0, 0.0], "heldout": [0, 0.0, 0.0, 0.0]}
        for k, v in m.items():
            if not k.startswith("family:"):
                continue
            key = "heldout" if "(heldout)" in k else "train"
            a = agg[key]
            a[0] += v["n"]
            a[1] += v["accuracy"] * v["n"]
            a[2] += v["ece"] * v["n"]
            a[3] += v["kl_to_ref"] * v["n"]
        t, h = agg["train"], agg["heldout"]
        print(f"| {tag} | {t[1] / t[0]:.3f} | {t[2] / t[0]:.3f} | {h[1] / h[0]:.3f} | {h[2] / h[0]:.3f} | {h[3] / h[0]:.3f} |")

    print("\n### per family (last run)\n")
    print("| family | n | acc | ECE | KL | agree Jev |\n|---|---|---|---|---|---|")
    for k, v in runs[-1][1].items():
        if k.startswith("family:"):
            print(f"| {k[7:]} | {v['n']} | {v['accuracy']:.3f} | {v['ece']:.3f} | {v['kl_to_ref']:.3f} | {v['agree_jev']:.3f} |")


if __name__ == "__main__":
    main()
