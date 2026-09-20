"""What moved a probability: token attribution and sentence ablation.

    uv run python scripts/explain.py runs/circuit-1.7b claim.json
    uv run python scripts/explain.py runs/circuit-1.7b claim.json --option yes

The item is {"state": ..., "question": {...}} — the same request body the
service takes, one question. Two readings come out:

  gradient x input   how much each input token contributed to the option's
                     pointer logit, from one backward pass. An internal
                     measurement: it says where the model looked, which is
                     not the same as why the answer is right.

  sentence ablation  the same question re-asked with one sentence removed at
                     a time. A counterfactual on the actual input, so it is
                     checkable by anyone with the endpoint and needs no
                     access to the weights. This is the reading to put in
                     front of someone who has to trust the result.

Neither is a reason the model holds. They are measurements of a function.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1proto.scorer import LoRAScorer, softmax
from s1proto.template import render


def option_logits(sc: LoRAScorer, text: str, n_options: int, grad: bool = False):
    """The pointer head's logits for one rendered prompt, optionally with a
    graph back to the input embeddings so they can be attributed."""
    enc = sc.tokenizer([text], return_tensors="pt", truncation=True, max_length=sc.max_length)
    ids = enc["input_ids"].to(sc.device)
    mask = enc["attention_mask"].to(sc.device)
    decoder = sc.model.get_base_model().model
    embeds = decoder.get_input_embeddings()(ids)
    if grad:
        embeds = embeds.detach().requires_grad_(True)
    hs = decoder(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    pos = (ids[0] == sc.opt_end_id).nonzero(as_tuple=True)[0]
    if len(pos) != n_options:
        raise ValueError(f"found {len(pos)} option delimiters for {n_options} options (truncated?)")
    q = sc.q(hs[:, -1, :].float())[0]
    k = sc.k(hs[0, pos, :].float())
    return (q * k).sum(-1) * sc.scale, ids[0], embeds


def attribute(sc: LoRAScorer, state, question, target: int) -> list[tuple[str, float]]:
    """Gradient x input on the target option's logit, summed per token."""
    p = render(state, question, layout="pointer")
    logits, ids, embeds = option_logits(sc, p.text, p.n_options, grad=True)
    sc.model.zero_grad(set_to_none=True)
    logits[target].backward()
    contrib = (embeds.grad[0] * embeds[0]).sum(-1).detach().float().cpu()
    toks = sc.tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
    return list(zip(toks, contrib.tolist(), strict=True))


def probabilities(sc: LoRAScorer, state, question) -> list[float]:
    p = render(state, question, layout="pointer")
    with torch.inference_mode():
        logits, _, _ = option_logits(sc, p.text, p.n_options)
    return softmax(logits.float().cpu().tolist())


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts or [text]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("item", help='JSON file: {"state": ..., "question": {...}}')
    ap.add_argument("--option", default=None, help="which option to explain (default: the one the model picked)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    item = json.loads(Path(args.item).read_text())
    q = item["question"]
    kind = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}[q["type"]]
    question = kind.model_validate(q)
    state = item["state"]
    sc = LoRAScorer(run_dir=args.run_dir)

    p = render(state, question, layout="pointer")
    probs = probabilities(sc, state, question)
    labels = list(p.option_keys)
    target = labels.index(args.option) if args.option else max(range(len(probs)), key=probs.__getitem__)
    print(f"model: {sc.name}")
    print("answer: " + ", ".join(f"{lab} {pr:.3f}" for lab, pr in zip(labels, probs, strict=True)))
    print(f"explaining: {labels[target]!r} at p={probs[target]:.3f}\n")

    print("gradient x input, tokens that pushed this option up:")
    scored = attribute(sc, state, question, target)
    ranked = sorted((t for t in scored if t[0] not in ("<|endoftext|>",)), key=lambda t: -t[1])
    for tok, val in ranked[: args.top]:
        print(f"  {val:+.3f}  {tok.replace('Ġ', ' ')!r}")
    print("  ... and pushing it down:")
    for tok, val in ranked[-3:]:
        print(f"  {val:+.3f}  {tok.replace('Ġ', ' ')!r}")

    if isinstance(state, str):
        parts = sentences(state)
        if len(parts) > 1:
            print("\nsentence ablation, the probability with each sentence removed:")
            print(f"  {probs[target]:.3f}  (as given)")
            for i, sent in enumerate(parts):
                without = " ".join(parts[:i] + parts[i + 1 :])
                p_without = probabilities(sc, without, question)[target]
                delta = probs[target] - p_without
                print(f"  {p_without:.3f}  ({delta:+.3f} without) {sent[:70]!r}")


if __name__ == "__main__":
    main()
