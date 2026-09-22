"""Figures for the option-order finding. Writes docs/figures/order_*.png from the results files.

uv run --with matplotlib python scripts/figures_order.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.family": "Helvetica Neue",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#444",
        "ytick.color": "#444",
        "figure.dpi": 200,
    }
)
INK, MUTED, RED, BLUE, GREY = "#1a1a1a", "#777777", "#c8401f", "#1f4fa0", "#b8b8b8"
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
R = Path("results")


def flip(tag):
    return json.load(open(R / f"perm_{tag}.json"))["results"]["all"]["flip"] * 100


# ---- Figure 1: the headline bar chart -------------------------------------------------
models = [
    ("SemIf\n(stock Qwen3.5-4B)", flip("semif"), GREY),
    ("circuit-8b v1.1", flip("circuit-8b-v1.1"), GREY),
    ("circuit-1.7b v1.1", flip("circuit-1.7b-v1.1"), GREY),
    ("circuit-8b v1.0", flip("circuit-8b"), GREY),
    ("Jev (hosted)", flip("jev"), GREY),
    ("circuit-1.7b,\noptions side by side", flip("circuit-1.7b-par2"), BLUE),
]
fig, ax = plt.subplots(figsize=(9, 5.2))
names = [m[0] for m in models]
vals = [m[1] for m in models]
cols = [m[2] for m in models]
bars = ax.barh(names, vals, color=cols, height=0.62)
for b, v in zip(bars, vals, strict=True):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", color=INK, fontsize=12, fontweight="bold" if v < 1 else "normal")
ax.set_xlim(0, 27)
ax.set_xlabel("Share of questions whose answer changed when only the option order changed")
ax.set_title("Same question, same options, different order: how often the answer changes", loc="left", fontsize=14, color=INK, pad=14)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.text(
    0.01,
    -0.03,
    "981 multiple-choice questions from 16 task families, each asked in 4 option orders. Jev's floor from repeating an identical request is about 4%. SemIf: the 716 items within its 16-option limit.",
    fontsize=9.5,
    color=MUTED,
)
fig.savefig(OUT / "order_flips.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Figure 2: one real question, two orders, four models ---------------------------
# ChaosNLI item: premise "Write, write, and write." / hypothesis "Writing is a waste of time."
# 100 annotators: entailed 1%, neutral 53%, contradicted 46%. Each order was asked twice.
orders = ["as written\nE · N · C", "reversed\nC · N · E"]
K = ["entailed", "neutral", "contradicted"]
data = {  # (entailed, neutral, contradicted) per order; Jev's two asks of each order averaged
    "Jev (hosted)": [(0.00, 0.525, 0.475), (0.00, 0.40, 0.60)],
    "circuit-1.7b v1.1": [(0.001, 0.048, 0.951), (0.002, 0.060, 0.938)],
    "SemIf (Qwen3.5-4B)": [(0.099, 0.077, 0.825), (0.007, 0.029, 0.964)],
    "circuit-1.7b, options side by side": [(0.005, 0.069, 0.926), (0.005, 0.070, 0.925)],
}
kcol = {"entailed": GREY, "neutral": BLUE, "contradicted": RED}
fig, axes = plt.subplots(1, 4, figsize=(13, 4.8), sharey=True)
for ax, (name, rows) in zip(axes, data.items(), strict=True):
    ax.set_xlim(-0.6, 1.6)
    for oi, row in enumerate(rows):
        for ki, (k, v) in enumerate(zip(K, row, strict=True)):
            ax.bar(oi + (ki - 1) * 0.26, v, width=0.24, color=kcol[k])
        top = K[max(range(3), key=row.__getitem__)]
        ax.text(oi, max(row) + 0.03, top, ha="center", fontsize=10.5, color=kcol[top], fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(orders, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_title(name, fontsize=11.5, color=INK, loc="left")
    ax.axhline(0.5, color="#ddd", lw=0.8, zorder=0)
axes[0].set_ylabel("probability")
from matplotlib.patches import Patch

axes[0].legend(handles=[Patch(color=kcol[k], label=k) for k in K], frameon=False, fontsize=9, loc="upper left")
fig.suptitle(
    'Premise "Write, write, and write."  Hypothesis "Writing is a waste of time."  Same question, options listed in two orders.',
    x=0.01,
    ha="left",
    fontsize=12.5,
    color=INK,
)
fig.text(
    0.01,
    -0.03,
    "Options E = entailed, N = neutral, C = contradicted. 100 annotators split 53% neutral / 46% contradicted. Jev answers neutral as written and contradicted reversed, both times each order was asked. SemIf keeps contradicted but moves .83 to .96. The side-by-side model returns the same numbers either way.",
    fontsize=9.3,
    color=MUTED,
)
fig.tight_layout()
fig.savefig(OUT / "order_example.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Figure 3: the same request three times -------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
runs = [(0.43, 0.40, 0.17), (0.57, 0.27, 0.16), (0.36, 0.48, 0.16)]
ours = [(0.43, 0.45, 0.12)] * 3
labels = ["row 0", "row 1", "none"]
cols3 = [BLUE, RED, GREY]
w = 0.12
for r, run in enumerate(runs):
    for j, v in enumerate(run):
        ax.bar(r * 1.0 + (j - 1) * w * 1.15 - 0.55, v, width=w, color=cols3[j])
    for j, v in enumerate(ours[r]):
        ax.bar(r * 1.0 + (j - 1) * w * 1.15 + 0.0, v, width=w, color=cols3[j], alpha=0.45)
    top = labels[max(range(3), key=run.__getitem__)]
    ax.text(r - 0.55, max(run) + 0.03, top, ha="center", fontsize=10, fontweight="bold", color=cols3[labels.index(top)])
    ax.text(r, max(ours[r]) + 0.03, "row 1", ha="center", fontsize=10, color=RED, alpha=0.8)
ax.set_xticks([-0.55, 0, 0.45, 1.0, 1.45, 2.0])
ax.set_xticklabels(["Jev", "circuit-1.7b", "Jev", "circuit-1.7b", "Jev", "circuit-1.7b"], fontsize=9.5)
for xr, t in zip([-0.27, 0.73, 1.73], ["request sent 1st", "request sent 2nd (identical)", "request sent 3rd (identical)"], strict=True):
    ax.text(xr, 0.86, t, ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 1.0)
ax.set_xlim(-0.9, 2.3)
ax.set_ylabel("probability")
ax.set_title('"con rod bolts 1275" — which table row? The byte-identical request, sent three times.', loc="left", fontsize=13, color=INK, pad=12)
from matplotlib.patches import Patch

ax.legend(
    handles=[Patch(color=c, label=l) for c, l in zip(cols3, labels, strict=True)], frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 0.75)
)
ax.text(
    -0.9,
    -0.2,
    "Jev answered row 0, row 0, row 1 (.43, .57, .48). Across 300 questions, 3.7% of identical requests to Jev get a different top answer; on our Mac tier, 0.0%.",
    fontsize=9.5,
    color=MUTED,
)
fig.tight_layout()
fig.savefig(OUT / "order_repeat.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Figure 4: training curve — order stability is the layout, calibration is the epoch --
import glob
import re

steps, flips, eces = [], [], []
for f in sorted(glob.glob(str(R / "steps_circuit-1.7b-par2_unseen_*.json")), key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1))):
    s = int(re.search(r"_(\d+)\.json$", f).group(1))
    pf = R / f"steps_circuit-1.7b-par2_perm_{s}.json"
    if not pf.exists():
        continue
    steps.append(s)
    flips.append(json.load(open(pf))["results"]["all"]["flip"] * 100)
    eces.append(json.load(open(f))["metrics"]["all"]["ece"] * 100)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(steps, flips, color=BLUE, lw=2.2, marker="o", ms=4, label="answers that flip under reordering (%)")
ax.plot(steps, eces, color=RED, lw=2.2, marker="o", ms=4, label="calibration error on unseen datasets (ECE, %)")
ax.axhline(14.5, color=GREY, lw=1, ls="--")
ax.text(steps[-1], 15.3, "circuit-1.7b v1.1, ordinary layout: 14.5% flips", ha="right", fontsize=9.5, color=MUTED)
ax.set_xlabel("training step")
ax.set_ylim(0, 27)
ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.62))
ax.set_title("With options encoded side by side, order stability is there from the first checkpoint", loc="left", fontsize=13, color=INK, pad=12)
ax.text(
    0,
    -6.5,
    "circuit-1.7b, 23 checkpoints over 2 epochs. Flips stay under 2% throughout; calibration off-distribution is best early and drifts with training.",
    fontsize=9.5,
    color=MUTED,
)
fig.tight_layout()
fig.savefig(OUT / "order_training.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("order_*.png")))
