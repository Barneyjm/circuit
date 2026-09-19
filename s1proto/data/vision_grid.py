"""The vision grid: rendered images with code-computed labels.

Same idea as the text grid (s1proto/data/grid.py): span the structure of
visual judgments rather than photo subjects, and know every label by
construction. Each item is an image (PNG on disk), a question, and a
reference distribution. About 8% of items are rendered to be
undecidable (blurred figure, occluded item, cropped chart) with a soft
label of 0.5.

Scenes (columns)                 Operations (rows)
  receipt   itemised bill          extract     what is the total? which store? paid by card?     choice / noul
  chart     bar chart              compare     which bar is tallest / shortest?                 choice
  table     small data table       count       how many rows are flagged / items over $10?      score 0..3+
  form      filled-in form         consistency does the claim match the document?               noul
  scene     shapes on a canvas     classify    what kind of document / chart is this?           choice
                                   read        what value is in field X?                        choice
                                   negation    is the box UNchecked / is the item NOT present?  noul

Not every (operation, scene) pair makes sense; `CELLS` lists the ones
that do. Output goes to data/vision/grid/<split>/<id>.png plus a JSONL
whose `state` is {"image": <relative path>, "text": <optional caption>}.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

STORES = ["Hofer", "Northside Market", "Blue Fern Grocery", "Quickmart", "Harbor Foods", "Cedar & Co", "Daily Basket"]
ITEMS = [
    "Milk 1L",
    "Bread",
    "Eggs 12",
    "Coffee 250g",
    "Bananas",
    "Yogurt",
    "Pasta 500g",
    "Tomatoes",
    "Cheese",
    "Rice 1kg",
    "Apples",
    "Butter",
    "Juice",
    "Chicken",
    "Chocolate",
]
COLORS = {"red": (214, 69, 65), "blue": (52, 120, 210), "green": (60, 160, 90), "yellow": (240, 200, 60), "purple": (140, 90, 190), "orange": (240, 140, 50)}
SHAPES = ["circle", "square", "triangle"]
FIELDS = ["Name", "Date", "Order ID", "Amount", "City", "Status"]
CHART_KINDS = ["bar chart", "line chart", "pie chart", "table"]


def _font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class VItem:
    cell: str
    kind: str
    image: Image.Image
    caption: str | None
    question: dict[str, Any]
    ref: dict[str, float]
    ambiguous: bool = False


def onehot(keys: list[str], k: str) -> dict[str, float]:
    return {x: (1.0 if x == k else 0.0) for x in keys}


def noul(p: float) -> dict[str, float]:
    return {"yes": p, "no": 1 - p}


def blur(im: Image.Image, box: tuple[int, int, int, int], radius: int = 6) -> Image.Image:
    region = im.crop(box).filter(ImageFilter.GaussianBlur(radius))
    im.paste(region, box)
    return im


# ------------------------------------------------------------- renderers
def render_receipt(rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    n = rng.randint(3, 8)
    rows = [(rng.choice(ITEMS), round(rng.uniform(0.5, 24.0), 2)) for _ in range(n)]
    total = round(sum(p for _, p in rows), 2)
    store = rng.choice(STORES)
    card = rng.random() < 0.6
    W, H = 420, 160 + 28 * n + 120
    im = Image.new("RGB", (W, H), (250, 248, 242))
    d = ImageDraw.Draw(im)
    f, fb = _font(20), _font(24)
    d.text((W / 2 - d.textlength(store, font=fb) / 2, 24), store, font=fb, fill=(20, 20, 20))
    d.text((30, 64), f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2026  {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}", font=f, fill=(60, 60, 60))
    y = 110
    for name, price in rows:
        d.text((30, y), name, font=f, fill=(20, 20, 20))
        s = f"{price:.2f}"
        d.text((W - 30 - d.textlength(s, font=f), y), s, font=f, fill=(20, 20, 20))
        y += 28
    d.line([(30, y + 6), (W - 30, y + 6)], fill=(120, 120, 120), width=2)
    y += 18
    d.text((30, y), "TOTAL", font=fb, fill=(20, 20, 20))
    s = f"{total:.2f}"
    d.text((W - 30 - d.textlength(s, font=fb), y), s, font=fb, fill=(20, 20, 20))
    total_box = (W - 200, y - 4, W - 20, y + 30)
    y += 44
    d.text((30, y), "PAID: " + ("VISA ****" + str(rng.randint(1000, 9999)) if card else "CASH"), font=f, fill=(60, 60, 60))
    return im, {"store": store, "rows": rows, "total": total, "card": card, "n": n, "total_box": total_box}


def render_chart(rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    k = rng.randint(3, 6)
    labels = rng.sample(["Q1", "Q2", "Q3", "Q4", "North", "South", "East", "West", "2023", "2024", "2025", "2026"], k)
    vals = rng.sample(range(10, 100), k)
    W, H = 520, 340
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = _font(16)
    d.line([(60, 20), (60, 280), (500, 280)], fill=(40, 40, 40), width=2)
    bw = 380 // k
    for i, (lab, v) in enumerate(zip(labels, vals, strict=True)):
        x0 = 70 + i * bw
        h = int(v * 2.4)
        d.rectangle([x0, 280 - h, x0 + bw - 14, 280], fill=rng.choice(list(COLORS.values())))
        d.text((x0 + (bw - 14) / 2 - d.textlength(lab, font=f) / 2, 288), lab, font=f, fill=(40, 40, 40))
        d.text((x0 + (bw - 14) / 2 - d.textlength(str(v), font=f) / 2, 280 - h - 20), str(v), font=f, fill=(40, 40, 40))
    return im, {"labels": labels, "vals": vals}


def render_table(rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    n = rng.randint(3, 7)
    rows = [(rng.choice(ITEMS), round(rng.uniform(1, 30), 2), rng.random() < 0.4) for _ in range(n)]
    W, H = 460, 60 + 34 * (n + 1)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = _font(17)
    cols = [30, 230, 340]
    for x, h in zip(cols, ["Item", "Price", "Flagged"], strict=True):
        d.text((x, 24), h, font=f, fill=(20, 20, 20))
    d.line([(20, 52), (W - 20, 52)], fill=(90, 90, 90), width=2)
    for i, (name, price, flag) in enumerate(rows):
        y = 64 + i * 34
        d.text((cols[0], y), name, font=f, fill=(20, 20, 20))
        d.text((cols[1], y), f"${price:.2f}", font=f, fill=(20, 20, 20))
        d.text((cols[2], y), "YES" if flag else "no", font=f, fill=(180, 30, 30) if flag else (90, 90, 90))
        d.line([(20, y + 28), (W - 20, y + 28)], fill=(220, 220, 220), width=1)
    return im, {"rows": rows}


def render_form(rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    vals = {
        "Name": rng.choice(["Priya Raman", "Marcus Hale", "Elena Vidal", "Kenji Sato", "Nora Flynn"]),
        "Date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "Order ID": f"ORD-{rng.randint(10000, 99999)}",
        "Amount": f"${rng.randint(10, 900)}.{rng.randint(0, 99):02d}",
        "City": rng.choice(["Austin", "Lisbon", "Osaka", "Denver", "Dublin"]),
        "Status": rng.choice(["Delivered", "Shipped", "Processing"]),
    }
    checks = {"Gift wrap": rng.random() < 0.5, "Express": rng.random() < 0.5, "Insured": rng.random() < 0.5}
    W, H = 480, 380
    im = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(im)
    f, fb = _font(17), _font(20)
    d.text((30, 20), "ORDER FORM", font=fb, fill=(20, 20, 20))
    boxes = {}
    for i, k in enumerate(FIELDS):
        y = 64 + i * 38
        d.text((30, y), k + ":", font=f, fill=(90, 90, 90))
        d.rectangle([160, y - 4, 440, y + 24], outline=(170, 170, 170))
        d.text((168, y), vals[k], font=f, fill=(20, 20, 20))
        boxes[k] = (160, y - 4, 440, y + 24)
    for i, (k, on) in enumerate(checks.items()):
        y = 300 + i * 26
        d.rectangle([30, y, 48, y + 18], outline=(60, 60, 60), width=2)
        if on:
            d.line([(33, y + 9), (38, y + 15), (46, y + 3)], fill=(20, 20, 20), width=3)
        d.text((58, y - 2), k, font=f, fill=(20, 20, 20))
    return im, {"vals": vals, "checks": checks, "boxes": boxes}


def render_scene(rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    W, H = 480, 360
    im = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(im)
    n = rng.randint(2, 9)
    placed = []
    counts: dict[tuple[str, str], int] = {}
    for _ in range(n):
        for _try in range(30):
            x, y, r = rng.randint(40, W - 40), rng.randint(40, H - 40), rng.randint(18, 34)
            if all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 6) ** 2 for px, py, pr in placed):
                break
        else:
            continue
        placed.append((x, y, r))
        color, shape = rng.choice(list(COLORS)), rng.choice(SHAPES)
        counts[(color, shape)] = counts.get((color, shape), 0) + 1
        c = COLORS[color]
        if shape == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=c)
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=c)
    return im, {"counts": counts, "n": len(placed)}


# ------------------------------------------------------------- items
def cell_receipt_extract(rng: random.Random) -> VItem:
    im, meta = render_receipt(rng)
    which = rng.choice(["total", "store", "card"])
    amb = rng.random() < 0.08
    if which == "total":
        wrong = [f"{meta['total'] + rng.choice([-5.0, 3.1, 10.0]):.2f}", f"{meta['total'] * 0.9:.2f}", f"{meta['rows'][0][1]:.2f}"]
        opts = [f"{meta['total']:.2f}", *wrong, "Cannot tell"]
        rng.shuffle(opts)
        if amb:
            im = blur(im, meta["total_box"], 8)
        q = {"type": "choice", "instructions": "What is the total on the receipt?", "criteria": {o: None for o in opts}}
        ref = {o: 0.0 for o in opts}
        ref["Cannot tell" if amb else f"{meta['total']:.2f}"] = 1.0
        return VItem("extract/receipt", "choice", im, None, q, ref, amb)
    if which == "store":
        opts = [meta["store"], *rng.sample([s for s in STORES if s != meta["store"]], 3)]
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": "Which store issued this receipt?", "criteria": {o: None for o in opts}}
        return VItem("extract/receipt", "choice", im, None, q, onehot(opts, meta["store"]))
    q = {"type": "noul", "instructions": "Was this receipt paid by card?", "criteria": {"true": "A card payment line is shown", "false": "Paid in cash"}}
    return VItem("extract/receipt", "noul", im, None, q, noul(1.0 if meta["card"] else 0.0))


def cell_chart_compare(rng: random.Random) -> VItem:
    im, meta = render_chart(rng)
    want_max = rng.random() < 0.5
    vals, labels = meta["vals"], meta["labels"]
    amb = rng.random() < 0.08
    if amb:  # make the two extremes equal so the strict question is undecidable
        i, j = sorted(range(len(vals)), key=lambda k: vals[k])[-2:] if want_max else sorted(range(len(vals)), key=lambda k: vals[k])[:2]
        vals[j] = vals[i]
        im = render_chart_with(rng, labels, vals)
    q = {"type": "choice", "instructions": f"Which bar is the {'tallest' if want_max else 'shortest'}?", "criteria": {lab: None for lab in labels}}
    target = max(range(len(vals)), key=lambda k: vals[k]) if want_max else min(range(len(vals)), key=lambda k: vals[k])
    ref = onehot(labels, labels[target])
    if amb:
        ties = [k for k in range(len(vals)) if vals[k] == vals[target]]
        ref = {lab: (1.0 / len(ties) if k in ties else 0.0) for k, lab in enumerate(labels)}
    return VItem("compare/chart", "choice", im, None, q, ref, amb)


def render_chart_with(rng: random.Random, labels: list[str], vals: list[int]) -> Image.Image:
    W, H = 520, 340
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = _font(16)
    d.line([(60, 20), (60, 280), (500, 280)], fill=(40, 40, 40), width=2)
    bw = 380 // len(labels)
    for i, (lab, v) in enumerate(zip(labels, vals, strict=True)):
        x0 = 70 + i * bw
        h = int(v * 2.4)
        d.rectangle([x0, 280 - h, x0 + bw - 14, 280], fill=rng.choice(list(COLORS.values())))
        d.text((x0 + (bw - 14) / 2 - d.textlength(lab, font=f) / 2, 288), lab, font=f, fill=(40, 40, 40))
        d.text((x0 + (bw - 14) / 2 - d.textlength(str(v), font=f) / 2, 280 - h - 20), str(v), font=f, fill=(40, 40, 40))
    return im


def cell_table_count(rng: random.Random) -> VItem:
    im, meta = render_table(rng)
    if rng.random() < 0.5:
        k = sum(1 for _, _, flag in meta["rows"] if flag)
        q_text = "How many rows are flagged?"
    else:
        k = sum(1 for _, price, _ in meta["rows"] if price > 10)
        q_text = "How many items cost more than $10?"
    q = {"type": "score", "instructions": q_text, "criteria": ["0", "1", "2", "3 or more"]}
    return VItem("count/table", "score", im, None, q, onehot(["0", "1", "2", "3"], str(min(k, 3))))


def cell_scene_count(rng: random.Random) -> VItem:
    im, meta = render_scene(rng)
    color = rng.choice(list(COLORS))
    k = sum(v for (c, _), v in meta["counts"].items() if c == color)
    q = {"type": "score", "instructions": f"How many {color} shapes are there?", "criteria": ["0", "1", "2", "3 or more"]}
    return VItem("count/scene", "score", im, None, q, onehot(["0", "1", "2", "3"], str(min(k, 3))))


def cell_form_read(rng: random.Random) -> VItem:
    im, meta = render_form(rng)
    field = rng.choice(FIELDS)
    truth = meta["vals"][field]
    amb = rng.random() < 0.08
    if amb:
        im = blur(im, meta["boxes"][field], 7)
    decoys = {
        "Name": ["Priya Raman", "Marcus Hale", "Elena Vidal", "Kenji Sato", "Nora Flynn"],
        "Date": [f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" for _ in range(6)],
        "Order ID": [f"ORD-{rng.randint(10000, 99999)}" for _ in range(6)],
        "Amount": [f"${rng.randint(10, 900)}.{rng.randint(0, 99):02d}" for _ in range(6)],
        "City": ["Austin", "Lisbon", "Osaka", "Denver", "Dublin"],
        "Status": ["Delivered", "Shipped", "Processing"],
    }[field]
    opts = [truth, *[x for x in decoys if x != truth][:3], "Cannot tell"]
    rng.shuffle(opts)
    q = {"type": "choice", "instructions": f"What is written in the {field} field?", "criteria": {o: None for o in opts}}
    ref = {o: 0.0 for o in opts}
    ref["Cannot tell" if amb else truth] = 1.0
    return VItem("read/form", "choice", im, None, q, ref, amb)


def cell_form_consistency(rng: random.Random) -> VItem:
    im, meta = render_form(rng)
    field = rng.choice(FIELDS)
    consistent = rng.random() < 0.5
    claim_val = (
        meta["vals"][field]
        if consistent
        else {"Name": "Sam Okoro", "Date": "2025-11-30", "Order ID": "ORD-00001", "Amount": "$1.00", "City": "Perth", "Status": "Cancelled"}[field]
    )
    caption = f"Customer claim: the {field.lower()} on my order was {claim_val}."
    q = {
        "type": "noul",
        "instructions": "Is the customer's claim consistent with the form?",
        "criteria": {"true": "The form shows the claimed value", "false": "The form shows something else"},
    }
    return VItem("consistency/form", "noul", im, caption, q, noul(1.0 if consistent else 0.0))


def cell_receipt_consistency(rng: random.Random) -> VItem:
    im, meta = render_receipt(rng)
    consistent = rng.random() < 0.5
    item = rng.choice(meta["rows"])[0] if consistent else rng.choice([i for i in ITEMS if i not in {r[0] for r in meta["rows"]}])
    caption = f"Customer claim: I bought {item} at {meta['store']}."
    q = {
        "type": "noul",
        "instructions": "Does the receipt support the customer's claim?",
        "criteria": {"true": "The item and store both appear", "false": "The item is not on the receipt"},
    }
    return VItem("consistency/receipt", "noul", im, caption, q, noul(1.0 if consistent else 0.0))


def cell_classify(rng: random.Random) -> VItem:
    kind = rng.choice(["receipt", "chart", "table", "form"])
    im = {"receipt": render_receipt, "chart": render_chart, "table": render_table, "form": render_form}[kind](rng)[0]
    opts = ["a receipt", "a bar chart", "a data table", "a form", "a photo"]
    truth = {"receipt": "a receipt", "chart": "a bar chart", "table": "a data table", "form": "a form"}[kind]
    q = {"type": "choice", "instructions": "What kind of document is this image?", "criteria": {o: None for o in opts}}
    return VItem(f"classify/{kind}", "choice", im, None, q, onehot(opts, truth))


def cell_form_negation(rng: random.Random) -> VItem:
    im, meta = render_form(rng)
    box = rng.choice(list(meta["checks"]))
    on = meta["checks"][box]
    ask_unchecked = rng.random() < 0.5
    q = {"type": "noul", "instructions": f"Is the '{box}' box {'left unchecked' if ask_unchecked else 'checked'}?", "criteria": {"true": "Yes", "false": "No"}}
    return VItem("negation/form", "noul", im, None, q, noul(1.0 if (on != ask_unchecked) else 0.0))


def cell_scene_negation(rng: random.Random) -> VItem:
    im, meta = render_scene(rng)
    color, shape = rng.choice(list(COLORS)), rng.choice(SHAPES)
    present = meta["counts"].get((color, shape), 0) > 0
    q = {
        "type": "noul",
        "instructions": f"Is there NO {color} {shape} in the image?",
        "criteria": {"true": f"No {color} {shape} anywhere", "false": f"At least one {color} {shape}"},
    }
    return VItem("negation/scene", "noul", im, None, q, noul(0.0 if present else 1.0))


CELLS = {
    "extract/receipt": cell_receipt_extract,
    "compare/chart": cell_chart_compare,
    "count/table": cell_table_count,
    "count/scene": cell_scene_count,
    "read/form": cell_form_read,
    "consistency/form": cell_form_consistency,
    "consistency/receipt": cell_receipt_consistency,
    "classify/any": cell_classify,
    "negation/form": cell_form_negation,
    "negation/scene": cell_scene_negation,
}


def generate(per_cell: int, seed: int, split: str, out_dir: Path) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}-{split}-vision")
    img_dir = out_dir / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cell, fn in CELLS.items():
        for _ in range(per_cell):
            it = fn(rng)
            digest = hashlib.sha1(it.image.tobytes()).hexdigest()[:12]
            rel = f"{split}/{it.cell.replace('/', '-')}-{digest}.png"
            it.image.save(out_dir / rel, optimize=True)
            keys = list(it.ref)
            rows.append(
                {
                    "id": f"vgrid-{it.cell.replace('/', '-')}-{digest}",
                    "family": (f"ambiguous/{it.cell.split('/')[0]}" if it.ambiguous else it.cell),
                    "operation": it.cell.split("/")[0],
                    "format": it.cell.split("/")[1],
                    "kind": it.kind,
                    "heldout": split == "eval",
                    "ambiguous": it.ambiguous,
                    "state": {"image": rel, **({"text": it.caption} if it.caption else {})},
                    "question": it.question,
                    "refs": {"code": {k: it.ref[k] for k in keys}},
                    "ref": {k: it.ref[k] for k in keys},
                    "source": "vision_grid",
                }
            )
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=120)
    ap.add_argument("--eval-per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="data/vision/grid")
    args = ap.parse_args()
    out = Path(args.out)
    for split, n in (("train", args.per_cell), ("eval", args.eval_per_cell)):
        rows = generate(n, args.seed, split, out)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"{split}: {len(rows)} items, cells {len(Counter(r['family'] for r in rows if not r['ambiguous']))}, ambiguous {sum(r['ambiguous'] for r in rows)} -> {out / f'{split}.jsonl'}"
        )
