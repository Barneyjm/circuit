"""Build an eval set from datasets the circuits have never seen in any split.

`hf_eval.jsonl` is held-out *rows*: four of its six families (CLINC, MNLI, Civil
Comments, SMS spam) also supply training rows. This is held-out *datasets*, so a
score here says something about a task nobody prepared the model for.

    uv run python scripts/build_unseen_eval.py --n 300
    # writes data/unseen_eval.jsonl and data/vision/unseen/eval.jsonl (+ images)

Families:
  bfcl_relevance   request + tool list -> can any tool do it?        noul    Apache-2.0
  halueval_qa      passage + question + answer -> supported?         noul    Apache-2.0 (mirror; original MIT)
  chaosnli         premise/hypothesis -> status, 100 human labels    choice  CC BY-NC 4.0, eval only
  hwu64            assistant utterance -> one of 64 intents          choice  CC BY 4.0
  pope             image -> is the named object present?             noul    MIT; COCO val2014 images

The reference is the human label, or for ChaosNLI the distribution of 100 of them,
which is the one place a probability can be checked against people rather than
against a single answer. Outputs are not committed: ChaosNLI is non-commercial and
the COCO images carry their photographers' licences. The seed makes them reproducible.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf_eval import item, onehot

BFCL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_{}.json"


def bfcl_rows(name: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(BFCL.format(name), headers={"User-Agent": "s1proto-eval/1.0"})) as r:
        return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]


def bfcl_state(row: dict) -> dict | None:
    """The user's request and what each tool is for. Parameter schemas are dropped:
    whether a tool is relevant is settled by its description, and they run to pages."""
    turns = [m["content"] for m in row["question"][0] if m["role"] == "user"]
    if len(row["question"]) != 1 or not turns:
        return None
    tools = [{"name": f["name"], "description": f["description"], "parameters": sorted(f.get("parameters", {}).get("properties", {}))} for f in row["function"]]
    state = {"request": turns[-1], "tools": tools}
    return state if len(json.dumps(state)) <= 2500 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="items per family")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/unseen_eval.jsonl")
    ap.add_argument("--vision-out", default="data/vision/unseen/eval.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out: list[dict] = []
    half = args.n // 2

    # --- BFCL: is any listed tool relevant to the request?
    q = {
        "type": "noul",
        "instructions": "Can at least one of the listed `tools` carry out the user's `request`?",
        "criteria": {"true": "A listed tool does what the request asks for", "false": "None of the listed tools fits; calling any of them would be wrong"},
    }
    yes = [s for r in bfcl_rows("live_simple") + bfcl_rows("live_relevance") if (s := bfcl_state(r))]
    no = [s for r in bfcl_rows("live_irrelevance") if (s := bfcl_state(r))]
    # same number of tools on both sides, so the count of tools cannot give the answer away
    no = [s for s in no if len(s["tools"]) == 1] or no
    yes = [s for s in yes if len(s["tools"]) == 1] or yes
    for s in rng.sample(yes, min(half, len(yes))):
        out.append(item("bfcl_relevance", "noul", s, q, {"yes": 1.0, "no": 0.0}))
    for s in rng.sample(no, min(args.n - half, len(no))):
        out.append(item("bfcl_relevance", "noul", s, q, {"yes": 0.0, "no": 1.0}))

    # --- HaluEval QA: is the answer supported by the passage?
    q = {
        "type": "noul",
        "instructions": "Is `answer` a correct answer to `question` according to `passage`?",
        "criteria": {"true": "The passage supports the answer", "false": "The answer states something the passage does not support"},
    }
    h = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    for i in rng.sample(range(len(h)), args.n):
        r = h[i]
        ok = r["hallucination"] == "no"
        out.append(
            item(
                "halueval_qa", "noul", {"passage": r["knowledge"], "question": r["question"], "answer": r["answer"]}, q, {"yes": float(ok), "no": float(not ok)}
            )
        )

    # --- ChaosNLI (MNLI part): the distribution of 100 annotators
    keys = ["entailed", "neutral", "contradicted"]
    q = {
        "type": "choice",
        "instructions": "Given `premise`, what is the status of `hypothesis`?",
        "criteria": {
            "entailed": "The hypothesis must be true if the premise is true",
            "neutral": "The premise does not settle it",
            "contradicted": "The hypothesis must be false if the premise is true",
        },
    }
    c = load_dataset("metaeval/chaos-mnli-ambiguity", split="train")
    for i in rng.sample(range(len(c)), min(args.n, len(c))):
        dist = c[i]["label_dist"]  # entailment, neutral, contradiction
        dist = ast.literal_eval(dist) if isinstance(dist, str) else dist
        out.append(item("chaosnli", "choice", {"premise": c[i]["premise"], "hypothesis": c[i]["hypothesis"]}, q, dict(zip(keys, dist, strict=True))))

    # --- HWU64: 64 intents, none shared with CLINC, banking77 or MASSIVE by name or source
    w = load_dataset("FastFit/hwu_64", split="test")
    names = sorted(set(w["label"]))
    q = {"type": "choice", "instructions": "Which intent does this request to a home assistant express?", "criteria": dict.fromkeys(names)}
    for i in rng.sample(range(len(w)), min(args.n, len(w))):
        out.append(item("hwu64", "choice", w[i]["text"], q, onehot(names, w[i]["label"])))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} text items to {args.out}")

    # --- POPE: object presence, a third each random / popular / adversarial
    vout = Path(args.vision_out)
    (vout.parent / "eval").mkdir(parents=True, exist_ok=True)
    p = load_dataset("lmms-lab-encoder/POPE", "default", split="test")
    cats = p["category"]
    vis = []
    for cat in ("random", "popular", "adversarial"):
        idx = [i for i, x in enumerate(cats) if x == cat]
        for i in rng.sample(idx, min(args.n // 3, len(idx))):
            r = p[i]
            name = f"eval/pope-{cat}-{r['question_id']}.jpg"
            r["image"].convert("RGB").save(vout.parent / name, quality=90)
            q = {"type": "noul", "instructions": r["question"], "criteria": {"true": "Yes, it is visible", "false": "No, it is not in the image"}}
            ok = r["answer"].strip().lower() == "yes"
            it = item(f"pope_{cat}", "noul", {"image": name}, q, {"yes": float(ok), "no": float(not ok)})
            it["id"] = f"pope-{cat}-{r['question_id']}"  # the state is only a filename; keep ids unique per question
            vis.append(it)
    with open(vout, "w") as f:
        for it in vis:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(vis)} image items to {vout}")


if __name__ == "__main__":
    main()
