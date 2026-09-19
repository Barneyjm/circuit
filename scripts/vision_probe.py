"""Phase-1 style probing of a raw vision-language model: image + question +
lettered options, read the next-token distribution over the letters.

    uv run python scripts/vision_probe.py photo.jpg --model Qwen/Qwen3-VL-4B-Instruct
    uv run python scripts/vision_probe.py x --set data/vision/probe_set.json          # a question set over several images
    uv run python scripts/vision_probe.py x --frames frames/                          # a video, as one frame per second

Asks a few questions about the image, then crops the image to a region
and asks again, to see whether the answer comes from text in the image
or from what is pictured.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from PIL import Image

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load(model_id: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    return proc, model


def letter_ids(proc) -> list[list[int]]:
    """Every single-token spelling of each letter ("A", " A"); mass is summed across them."""
    tok = proc.tokenizer
    out = []
    for letter in LETTERS:
        variants = set()
        for text in (letter, " " + letter):
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                variants.add(ids[0])
        out.append(sorted(variants))
    return out


@torch.inference_mode()
def ask(proc, model, image: Image.Image | list[Image.Image], question: str, options: list[str], device: str) -> dict[str, float]:
    """`image` may be one PIL image or a list of frames, which is passed as a video."""
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    text = f"{question}\n\nOptions:\n{lines}\n\nAnswer with the single letter of the best option."
    video = isinstance(image, list)
    messages = [{"role": "user", "content": [{"type": "video" if video else "image"}, {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], **({"videos": [image]} if video else {"images": [image]}), return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()
    full = torch.log_softmax(logits, dim=-1)
    lp = torch.stack([torch.logsumexp(full[ids], dim=0) for ids in letter_ids(proc)[: len(options)]])
    p = torch.softmax(lp, dim=-1).tolist()  # renormalize over the letters only
    mass = math.exp(torch.logsumexp(lp, dim=0).item())  # how much of the vocab mass landed on letters at all
    top_id = int(full.argmax())
    return {"probs": dict(zip(options, p, strict=True)), "letter_mass": mass, "top_token": proc.tokenizer.decode([top_id])}


def run_set(args) -> None:
    import json
    from pathlib import Path

    items = json.loads(Path(args.set).read_text())
    proc, model = load(args.model, args.device)
    hits = total = 0
    for it in items:
        img = Image.open(Path(args.set).with_name(it["image"])).convert("RGB")
        img.thumbnail((896, 896))
        print(f"\n== {it['image']}")
        for q in it["questions"]:
            r = ask(proc, model, img, q["q"], q["options"], args.device)
            top = max(r["probs"], key=r["probs"].get)
            expect = q.get("expect")
            ok = None if expect is None else top in (expect if isinstance(expect, list) else [expect])
            if ok is not None:
                total += 1
                hits += ok
            mark = "  ?" if ok is None else (" ok" if ok else " XX")
            dist = "  ".join(f"{o}={p:.2f}" for o, p in r["probs"].items())
            print(f" {mark} {q['q']:<48} -> {top:<18} [{dist}]")
    print(f"\n{hits}/{total} decidable questions right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--crop", default=None, help="left,top,right,bottom as fractions, e.g. 0.1,0.55,0.9,1.0")
    ap.add_argument(
        "--set", default=None, help="JSON: [{image, questions: [{q, options, expect}]}]; expect may be a list for acceptable answers or null for ambiguous"
    )
    ap.add_argument("--frames", default=None, help="directory of frame images (e.g. from `ffmpeg -vf fps=1`) to pass as one video instead of an image")
    args = ap.parse_args()
    if args.set:
        run_set(args)
        return

    t0 = time.perf_counter()
    proc, model = load(args.model, args.device)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.0f}s")
    if args.frames:
        import glob

        frames = [Image.open(f).convert("RGB") for f in sorted(glob.glob(args.frames.rstrip("/") + "/*.jpg"))]
        views = {f"video ({len(frames)} frames)": frames}
    else:
        img = Image.open(args.image).convert("RGB")
        img.thumbnail((896, 896))
        views = {"full image": img}
    if args.crop and not args.frames:
        l, t, r, b = (float(x) for x in args.crop.split(","))
        w, h = img.size
        views["cropped"] = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))

    questions = [
        ("Is the food shown spicy?", ["Yes", "No"]),
        ("Is this a beverage?", ["Yes", "No"]),
        ("What kind of food is this?", ["Pasta", "Soup", "Salad", "Bread", "Dessert"]),
        ("How spicy would you expect this to be?", ["Not spicy", "Mild", "Medium", "Hot", "Extremely hot"]),
        ("Which brand is this?", ["Knorr", "Kraft", "Barilla", "Annie's", "Cannot tell"]),
    ]
    for name, view in views.items():
        print(f"\n== {name} {view[0].size if isinstance(view, list) else view.size}")
        for q, opts in questions:
            t0 = time.perf_counter()
            r = ask(proc, model, view, q, opts, args.device)
            top = max(r["probs"], key=r["probs"].get)
            dist = "  ".join(f"{o}={p:.2f}" for o, p in r["probs"].items())
            print(f"  {q:<42} -> {top:<14} [{dist}]  letters={r['letter_mass']:.2f} top={r['top_token']!r}  {1000 * (time.perf_counter() - t0):.0f} ms")


if __name__ == "__main__":
    main()
