"""Score an eval JSONL with Bespoke-Nimble-9B (bespokelabs/Bespoke-Nimble-9B),
using its own prompt builder verbatim so the comparison is fair, and
report the same metrics as every other row.

    uv run python scripts/eval_nimble.py data/hf_eval.jsonl --nimble /path/to/nimble-model --out results/hf_nimble.json

Nimble's reference runner insists on CUDA; this reproduces its scoring
(one prompt per field, next-token logits gathered at the letter codes,
softmax over the letters) on whatever device is available. Its schema
has `boolean` and `enum` fields with at most 26 choices; questions
with more options are counted as errors with a uniform prediction.
Prompts over its 2,048-token limit are rejected by its builder and
counted the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_set import summarize

from s1proto.data.teachers import option_keys


def to_schema(q: dict) -> tuple[dict, bool]:
    """Our question -> one Nimble schema field; returns (schema, is_score)."""
    kind = q["type"]
    crit = q.get("criteria")
    if kind == "noul":
        field = {"type": "boolean", "description": str(q["instructions"])}
        if isinstance(crit, dict) and (crit.get("true") or crit.get("false")):
            field["choice_descriptions"] = {k: str(v) for k, v in (("true", crit.get("true")), ("false", crit.get("false"))) if v}
        return {"q": field}, False
    if kind == "choice":
        keys = list(crit.keys())
        field = {"type": "enum", "choices": keys, "description": str(q["instructions"])}
        desc = {k: str(v) for k, v in crit.items() if v}
        if desc:
            field["choice_descriptions"] = desc
        return {"q": field}, False
    levels = [str(i) for i in range(len(crit))]
    return {"q": {"type": "enum", "choices": levels, "description": str(q["instructions"]), "choice_descriptions": {str(i): str(t) for i, t in enumerate(crit)}}}, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--nimble", required=True, help="directory with the adapter, schema_config.json, parallel_schema.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    sys.path.insert(0, args.nimble)
    from parallel_schema import prepare_prompts  # Nimble's own prompt code
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = json.loads((Path(args.nimble) / "schema_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.nimble)
    t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(contract["model"], revision=contract["revision"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.nimble).to(args.device).eval()
    print(f"loaded {contract['model']} + adapter in {time.perf_counter() - t0:.0f}s on {args.device}", flush=True)

    items = [json.loads(line) for line in open(args.data)]
    if args.limit:
        items = items[: args.limit]
    preds: list[list[float]] = []
    lat: list[float] = []
    skipped = 0
    with torch.inference_mode():
        for i, it in enumerate(items):
            keys = option_keys(it["question"])
            state = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], ensure_ascii=False)
            try:
                schema, _ = to_schema(it["question"])
                prepared = prepare_prompts(tok, state, schema, contract["max_length"])
            except ValueError:
                skipped += 1
                preds.append([1.0 / len(keys)] * len(keys))
                continue
            ids = torch.tensor([prepared.full_ids[0]], device=args.device)
            cand = torch.tensor([prepared.candidate_ids[0]], device=args.device)
            t1 = time.perf_counter()
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            sel = logits.gather(1, cand)[0][: len(keys)]
            p = torch.softmax(sel, dim=-1).tolist()
            lat.append((time.perf_counter() - t1) * 1000)
            if it["question"]["type"] == "noul":  # Nimble's boolean order is [false, true]; ours is [yes, no]
                p = [p[1], p[0]]
            preds.append(p)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(items)}  {sum(lat) / len(lat):.0f} ms/item", flush=True)

    summary = summarize(items, preds)
    lat_sorted = sorted(lat)
    timing = {"device": args.device, "items": len(items), "skipped": skipped, "ms_per_item": round(sum(lat) / len(lat), 1) if lat else None, "ms_per_item_p50": round(lat_sorted[len(lat_sorted) // 2], 1) if lat else None, "batch": 1, "note": "one forward per question, Nimble's own prompt builder"}
    result = {"model": "bespokelabs/Bespoke-Nimble-9B", "data": args.data, "timing": timing, "metrics": summary}
    print("timing:", json.dumps(timing))
    for k, v in summary.items():
        if k.startswith("family:"):
            print(f"  {k[7:].replace(' (heldout)', ''):<28} acc={v['accuracy']:.3f} ece={v['ece']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
