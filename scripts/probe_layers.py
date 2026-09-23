"""How far down the stack is the answer already readable? (early exit)

Freezes a trained circuit (base + LoRA), runs it once over a sample of its own training
split, and records the decide-token and option-delimiter hidden states after layer k for
several k (each passed through the model's final norm, as the top layer is). A fresh
pointer head is trained per layer on those vectors and scored on the circuit's validation
split and on the unseen families. Layer = all layers is the control: a fresh head there
shows what the head-training budget alone reaches, and the circuit's own head is printed
beside it.

The adapter was trained with the head at the top, so lower layers were never pushed to
carry the answer: this understates how low the stack can be cut. A layer that comes close
here is a clear yes; one that doesn't needs a run that trains the adapter cut at that layer.

    uv run python scripts/probe_layers.py runs/circuit-1.7b-v1.2 --layers 8,12,16,20,24,28
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora
from train_lora import build_batch, ece15, soft_ce

from s1proto.media import PointerHead


def split(items: list[dict], seed: int, val_frac: float) -> tuple[list[dict], list[dict]]:
    """train_lora's family-stratified split, so validation here is the circuit's validation."""
    rng = random.Random(seed)
    by_fam: dict[str, list[dict]] = {}
    for it in items:
        by_fam.setdefault(it["family"], []).append(it)
    train, val = [], []
    for lst in by_fam.values():
        rng.shuffle(lst)
        k = max(1, int(len(lst) * val_frac))
        val.extend(lst[:k])
        train.extend(lst[k:])
    return train, val


@torch.no_grad()
def features(model, tok, items, layers, device, batch, max_length):
    """Per item: {layer: (h_decide [H], h_opts [n, H])} in float16 on the CPU, plus ref and kind."""
    body = model.get_base_model().model
    caught: dict[int, torch.Tensor] = {}
    hooks = [body.layers[k - 1].register_forward_hook(lambda m, i, o, k=k: caught.__setitem__(k, o[0] if isinstance(o, tuple) else o)) for k in layers]
    rng = random.Random(0)
    out = []
    t0 = time.time()
    for s in range(0, len(items), batch):
        chunk = items[s : s + batch]
        enc, ref, nopts, opt_pos, dec_pos, _src = build_batch(tok, chunk, rng, device, max_length, train=False, layout="pointer")
        body(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "position_ids")}, use_cache=False)
        rows = torch.arange(len(chunk), device=device)
        per = {}
        for k in layers:
            hs = body.norm(caught[k])
            per[k] = (hs[rows, dec_pos].half().cpu(), hs[rows.unsqueeze(1), opt_pos].half().cpu())
        for i, it in enumerate(chunk):
            n = int(nopts[i])
            out.append(
                {
                    "ref": ref[i, :n].cpu(),
                    "kind": it["kind"],
                    "h": {k: (per[k][0][i], per[k][1][i, :n]) for k in layers},
                }
            )
        if (s // batch) % 50 == 0:
            print(f"  {len(out)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()
    return out


def batches(feats, k, size, shuffle, rng):
    idx = list(range(len(feats)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), size):
        part = [feats[i] for i in idx[s : s + size]]
        maxn = max(len(f["ref"]) for f in part)
        hd = torch.stack([f["h"][k][0] for f in part]).float()
        ho = torch.zeros(len(part), maxn, hd.shape[-1])
        ref = torch.zeros(len(part), maxn)
        for i, f in enumerate(part):
            n = len(f["ref"])
            ho[i, :n] = f["h"][k][1].float()
            ref[i, :n] = f["ref"]
        yield hd, ho, ref, torch.tensor([len(f["ref"]) for f in part]), part


def score(head, feats, k):
    confs, correct = [], []
    with torch.no_grad():
        for hd, ho, ref, nopts, _ in batches(feats, k, 256, False, None):
            p = torch.softmax(head(hd, ho, nopts), -1)
            for i in range(len(nopts)):
                n = int(nopts[i])
                confs.append(float(p[i, :n].max()))
                correct.append(bool(p[i, :n].argmax() == ref[i, :n].argmax()))
    return sum(correct) / len(correct), ece15(confs, correct)


def train_head(feats, k, hidden, dim, epochs, seed):
    torch.manual_seed(seed)
    head = PointerHead(hidden, dim)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(seed)
    for _ in range(epochs):
        for hd, ho, ref, nopts, _ in batches(feats, k, 64, True, rng):
            loss = soft_ce(head(hd, ho, nopts), ref, nopts)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--layers", default="8,12,16,20,24,28")
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--unseen", default="data/unseen_eval.jsonl")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    if cfg.get("parallel_options"):
        train_lora.PARALLEL_OPTIONS = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    layers = [int(x) for x in args.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    tok.padding_side = tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(run / "adapter")).to(device).eval()
    n_layers = len(model.get_base_model().model.layers)
    assert max(layers) <= n_layers, f"model has {n_layers} layers"

    a = cfg["args"]
    items = [json.loads(line) for line in open(a["data"])]
    train, val = split(items, a["seed"], a["val_frac"])
    rng = random.Random(1)
    train, val = rng.sample(train, min(args.train_n, len(train))), rng.sample(val, min(args.val_n, len(val)))
    unseen = [json.loads(line) for line in open(args.unseen)]
    print(f"{n_layers} layers; probing {layers}; train {len(train)} val {len(val)} unseen {len(unseen)} on {device}", flush=True)

    feats = {}
    for name, its in (("train", train), ("val", val), ("unseen", unseen)):
        print(f"features: {name}", flush=True)
        feats[name] = features(model, tok, its, layers, device, args.batch, args.max_length)

    own = PointerHead(cfg["hidden"], cfg.get("head_dim", 256))
    own.load_state_dict(torch.load(run / "head.pt", map_location="cpu"))
    own.eval()
    results = {"run": str(run), "n_layers": n_layers, "train_n": len(train), "layers": {}}
    va, ve = score(own, feats["val"], n_layers) if n_layers in layers else (None, None)
    ua, ue = score(own, feats["unseen"], n_layers) if n_layers in layers else (None, None)
    if va is not None:
        results["own_head"] = {"val_acc": va, "val_ece": ve, "unseen_acc": ua, "unseen_ece": ue}
        print(f"own head (layer {n_layers}): val {va:.3f} ece {ve:.3f} | unseen {ua:.3f} ece {ue:.3f}", flush=True)
    for k in layers:
        head = train_head(feats["train"], k, cfg["hidden"], cfg.get("head_dim", 256), args.epochs, 0)
        va, ve = score(head, feats["val"], k)
        ua, ue = score(head, feats["unseen"], k)
        results["layers"][k] = {"val_acc": va, "val_ece": ve, "unseen_acc": ua, "unseen_ece": ue, "compute": round(k / n_layers, 3)}
        print(f"layer {k:2d} ({k / n_layers:4.0%} of the stack): val {va:.3f} ece {ve:.3f} | unseen {ua:.3f} ece {ue:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
