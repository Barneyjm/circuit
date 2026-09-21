"""Phase 3: LoRA fine-tune with a proper scoring rule against reference
distributions, plus a fresh 256-way answer head at the slot position.

What changes vs Phase 1:
  - The answer is read from a linear head on the slot's hidden state
    (masked to the question's N options), not from label-token logits.
    Removes letter bias and lifts the option cap to 255.
  - Loss = cross-entropy with soft targets (the averaged Jev+Gemini
    reference). Never argmax targets.
  - Option order is shuffled per example (Choice only) so the head
    learns position invariance.
  - Early stopping on validation ECE (ref argmax as label), not accuracy.

    uv run python scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/train.jsonl --out runs/1.7b-r16 --epochs 2
    uv run python scripts/eval_set.py lora:runs/1.7b-r16 data/eval.jsonl

Checkpoint layout: <out>/adapter (peft), <out>/head.pt, <out>/config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from s1proto import template as T
from s1proto.data.teachers import option_keys
from s1proto.media import (
    CAPTION,
    HEAD_DIM,
    HEAD_SIZE,
    PointerHead,
    SlotHead,
    chat_text,
    encode,
    head_logits,
    hidden_states,
    load_media,
)
from s1proto.parallel import is_choice, parallel_inputs
from s1proto.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1proto.template import render


def parse_question(q: dict):
    return {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}[q["type"]].model_validate(q)


def shuffled(item: dict, rng: random.Random) -> tuple[dict, list[float]]:
    """Return (question with shuffled options, ref in shown order)."""
    q = item["question"]
    keys = option_keys(q)
    if q["type"] == "choice":
        order = keys[:]
        rng.shuffle(order)
        q = {**q, "criteria": {k: q["criteria"][k] for k in order}}
        keys = order
    return q, [item["ref"][k] for k in keys]


def ece15(confs, correct, bins=15):
    n = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0.0)]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return e


PARALLEL_OPTIONS: torch.dtype | None = None  # set by --parallel-options to the model's dtype, which the mask must share


def build_batch(tok, items, rng, device, max_length: int, train: bool, layout: str = "letters", proc=None, image_root=None, modality: str | None = None):
    """Returns (enc, ref, nopts, opt_pos, dec_pos). `enc` holds the model
    inputs (input_ids, attention_mask, and for vision pixel_values and
    image_grid_thw). `opt_pos` [B, maxn] is each option's closing
    delimiter (pointer layout); `dec_pos` [B] is the decide token, which
    is the last token for text and found by id under a chat template."""
    modality = modality or ("vision" if proc is not None else "text")
    texts, refs, nopts, media = [], [], [], []
    for it in items:
        q, ref = shuffled(it, rng) if train else (it["question"], [it["ref"][k] for k in option_keys(it["question"])])
        if modality in ("vision", "audio"):  # the image or clip is the state; any text rides along as a caption
            key = "image" if modality == "vision" else "audio"
            p = render(it["state"].get("text") or CAPTION[modality], parse_question(q), layout=layout)
            texts.append(chat_text(proc, p.text, modality))
            media.append(load_media(modality, it["state"][key], root=Path(image_root)))
        else:
            texts.append(render(it["state"], parse_question(q), layout=layout).text)
        refs.append(ref)
        nopts.append(len(ref))
    if modality in ("vision", "audio"):
        enc = encode(proc, texts, media, modality)
    else:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    ids = enc["input_ids"]
    maxn = max(nopts)
    ref_t = torch.zeros((len(items), maxn))
    for i, r in enumerate(refs):
        ref_t[i, : len(r)] = torch.tensor(r)
    opt_pos = torch.zeros((len(items), maxn), dtype=torch.long)
    dec_pos = torch.full((len(items),), ids.shape[1] - 1, dtype=torch.long)
    if layout == "pointer":
        end_id, dec_id = tok.convert_tokens_to_ids(T.OPT_END), tok.convert_tokens_to_ids(T.DECIDE)
        for i in range(len(items)):
            pos = (ids[i] == end_id).nonzero(as_tuple=True)[0]
            if len(pos) != nopts[i]:
                raise ValueError(f"item {items[i].get('id')}: found {len(pos)} option delimiters for {nopts[i]} options (truncated? raise --max-length)")
            opt_pos[i, : nopts[i]] = pos
            dpos = (ids[i] == dec_id).nonzero(as_tuple=True)[0]
            if len(dpos) == 0:
                raise ValueError(f"item {items[i].get('id')}: no decide token")
            dec_pos[i] = dpos[-1]
    if PARALLEL_OPTIONS and layout == "pointer" and modality == "text":
        start_id = tok.convert_tokens_to_ids(T.OPT_START)
        mask4d, position_ids = parallel_inputs(ids, enc["attention_mask"], [is_choice(t) for t in texts], start_id, dec_id, PARALLEL_OPTIONS)
        enc = {"input_ids": ids, "attention_mask": mask4d, "position_ids": position_ids}
    enc = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
    return enc, ref_t.to(device), torch.tensor(nopts, device=device), opt_pos.to(device), dec_pos.to(device)


def soft_ce(logits: torch.Tensor, ref: torch.Tensor, nopts: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets over the first n options of each row."""
    logp = F.log_softmax(logits[:, : ref.shape[1]], dim=-1)
    logp = torch.nan_to_num(logp, neginf=0.0)  # masked slots have ref 0 anyway
    return -(ref * logp).sum(dim=-1).mean()


def pick_checkpoint(history: list[dict], acc_floor: float) -> dict:
    """The best-calibrated checkpoint among those that are also accurate.

    Lowest ECE alone is the right instinct for a model sold on calibration, and it has
    one failure: early in training a model can be well calibrated about knowing nothing
    (a router run kept a 26%-accurate step this way). So only checkpoints within
    `acc_floor` of the best validation accuracy seen are eligible. Past that floor more
    steps buy accuracy where the training data lives and spend calibration everywhere
    else, which is why this does not simply take the last one."""
    top = max(h["accuracy"] for h in history)
    return min((h for h in history if h["accuracy"] >= top - acc_floor), key=lambda h: h["ece"])


@torch.no_grad()
def evaluate(model, head, tok, items, device, batch, max_length, layout="letters", proc=None, image_root=None, modality="text"):
    model.eval()
    head.eval()
    confs, correct, kls = [], [], []
    rng = random.Random(0)
    for s in range(0, len(items), batch):
        chunk = items[s : s + batch]
        enc, ref, nopts, opt_pos, dec_pos = build_batch(
            tok, chunk, rng, device, max_length, train=False, layout=layout, proc=proc, image_root=image_root, modality=modality
        )
        logits = head_logits(head, hidden_states(model, enc, modality), nopts, opt_pos, dec_pos)[:, : ref.shape[1]]
        p = torch.softmax(logits, dim=-1)
        for i in range(len(chunk)):
            n = int(nopts[i])
            pi, ri = p[i, :n], ref[i, :n]
            confs.append(float(pi.max()))
            correct.append(bool(pi.argmax() == ri.argmax()))
            kls.append(float((ri * (torch.log(ri + 1e-9) - torch.log(pi + 1e-9))).sum()))
    model.train()
    head.train()
    return {"ece": ece15(confs, correct), "accuracy": sum(correct) / len(correct), "kl": sum(kls) / len(kls)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--acc-floor",
        type=float,
        default=0.01,
        help="keep the lowest-ECE checkpoint among those within this much of the best validation accuracy; 1.0 is lowest ECE outright",
    )
    ap.add_argument(
        "--parallel-options",
        action="store_true",
        help="encode a choice question's options side by side so the answer cannot depend on their order (s1proto/parallel.py)",
    )
    ap.add_argument(
        "--micro",
        type=int,
        default=0,
        help="rows per forward pass, accumulated to --batch (0 = whole batch). Use 1 for Qwen3.5: its backward pass returns NaN gradients on a left-padded batch, and a single row needs no padding.",
    )
    ap.add_argument("--grad-checkpoint", action="store_true", help="trade compute for activation memory (use for 8B+ on 64 GB)")
    ap.add_argument("--wandb", default=None, help="W&B project name; logs loss/lr every 10 steps and val metrics per checkpoint")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true", help="resume from <out>/latest if present")
    ap.add_argument(
        "--head",
        choices=["pointer", "slot"],
        default="pointer",
        help="pointer: query/key readout over option delimiters (no option cap); slot: fixed 256-way linear head",
    )
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: load the base in 4-bit NF4 (bitsandbytes, CUDA only); the scorer reloads it the same way")
    ap.add_argument(
        "--modality",
        choices=["text", "vision", "audio"],
        default="text",
        help="vision: a vision-language base (Qwen3-VL); items carry state.image relative to the data file's directory. audio: an audio-language base (Qwen2-Audio); items carry state.audio",
    )
    ap.add_argument("--exclude-family", default=None, help="comma-separated families to drop from training (leave-one-task-out)")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(project=args.wandb, name=args.run_name or Path(args.out).name, config=vars(args))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if args.parallel_options:
        global PARALLEL_OPTIONS
        PARALLEL_OPTIONS = getattr(torch, args.dtype)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    items = [json.loads(line) for line in open(args.data)]
    if args.exclude_family:
        drop = set(args.exclude_family.split(","))
        before = len(items)
        items = [it for it in items if it["family"] not in drop]
        print(f"excluded {before - len(items)} items from families {sorted(drop)}", flush=True)
    if args.limit:
        rng.shuffle(items)
        items = items[: args.limit]
    # family-stratified validation split
    by_fam: dict[str, list[dict]] = {}
    for it in items:
        by_fam.setdefault(it["family"], []).append(it)
    train, val = [], []
    for fam, lst in by_fam.items():
        rng.shuffle(lst)
        k = max(1, int(len(lst) * args.val_frac))
        val.extend(lst[:k])
        train.extend(lst[k:])
    print(f"train={len(train)} val={len(val)} families={len(by_fam)} device={device}")

    proc = image_root = None
    if args.modality in ("vision", "audio"):
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(args.model)
        tok = proc.tokenizer
        image_root = Path(args.data).parent
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
    pointer_tokens = T.pointer_tokens_for(tok) if args.head == "pointer" else None
    if pointer_tokens:
        T.use_pointer_tokens(*pointer_tokens)
    tok.padding_side = "left"
    # Truncate from the left: the answer slot is the last token and the head
    # reads it. Right truncation silently drops the slot on long option
    # lists (a 151-way CLINC prompt is ~840 tokens), and the head then
    # trains on a random mid-list position.
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=args.grad_checkpoint)
    elif args.modality == "vision":
        from transformers import AutoModelForImageTextToText

        base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        base.to(device)
        if args.grad_checkpoint:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
    elif args.modality == "audio":
        from transformers import Qwen2AudioForConditionalGeneration

        base = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        base.to(device)
        if args.grad_checkpoint:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
    else:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        base.to(device)
        if args.grad_checkpoint:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
    # LoRA on the language model only; the vision tower stays frozen
    targets = (
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        if args.modality in ("vision", "audio")
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=targets, bias="none")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    hidden = getattr(base.config, "hidden_size", None) or base.config.get_text_config().hidden_size  # VL configs nest the LM config
    head = (PointerHead(hidden) if args.head == "pointer" else SlotHead(hidden)).to(device)
    layout = "pointer" if args.head == "pointer" else "letters"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if args.resume and (out / "latest" / "state.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(out / "latest" / "adapter" / "adapter_model.safetensors"))
        head.load_state_dict(torch.load(out / "latest" / "head.pt", map_location="cpu"))
        start_step = json.load(open(out / "latest" / "state.json"))["step"]
        print(f"resumed from step {start_step}")

    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.05, 1 - s / max(1, total_steps)))

    # Length-bucketed batches: a batch is padded to its longest member, so
    # mixing one 840-token CLINC prompt into a batch of 100-token items
    # makes all of them cost 840. Sort within shuffled windows of 50
    # batches so batches are near-uniform in length, then shuffle the
    # batch order. Lengths are measured once on the rendered prompt.
    def length_of(it: dict) -> int:
        state = (it["state"].get("text") or "See the image.") if args.modality != "text" else it["state"]
        return (
            len(tok(render(state, parse_question(it["question"]), layout=layout).text)["input_ids"]) + {"text": 0, "vision": 300, "audio": 400}[args.modality]
        )

    lengths = {it["id"]: min(args.max_length, length_of(it)) for it in train}

    def bucketed_batches(rows: list[dict]) -> list[list[dict]]:
        rng.shuffle(rows)
        window = args.batch * 50
        batches: list[list[dict]] = []
        for w in range(0, len(rows), window):
            chunk = sorted(rows[w : w + window], key=lambda it: lengths[it["id"]])
            batches.extend(chunk[b : b + args.batch] for b in range(0, len(chunk), args.batch))
        rng.shuffle(batches)
        return batches

    best: dict = {}
    history: list[dict] = []
    step = 0
    t0 = time.time()
    skip = start_step
    model.train()
    head.train()
    for epoch in range(args.epochs):
        for chunk in bucketed_batches(train):
            if skip > 0:
                skip -= 1
                step += 1
                continue
            micro = args.micro or len(chunk)
            batch_loss = 0.0
            for m in range(0, len(chunk), micro):
                part = chunk[m : m + micro]
                enc, ref, nopts, opt_pos, dec_pos = build_batch(
                    tok, part, rng, device, args.max_length, train=True, layout=layout, proc=proc, image_root=image_root, modality=args.modality
                )
                if len(part) == 1 and args.modality == "text":
                    # Qwen3.5 compiles its linear-attention path once per sequence length, about
                    # 1.4 s each. Round the length up so there are 16 of them, not a thousand. The
                    # filler sits after the decide token and stays unmasked: a causal model never
                    # lets it reach the positions the head reads, and masking is what NaNs.
                    n = enc["input_ids"].shape[1]
                    fill = -n % 64
                    if fill:
                        enc["input_ids"] = F.pad(enc["input_ids"], (0, fill), value=tok.pad_token_id)
                        enc["attention_mask"] = F.pad(enc["attention_mask"], (0, fill), value=1)
                logits = head_logits(head, hidden_states(model, enc, args.modality), nopts, opt_pos, dec_pos)
                # weighted so the accumulated gradient is the mean over the whole batch
                loss = soft_ce(logits, ref, nopts) * len(part) / len(chunk)
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                # Bucketed batches vary in length, and the MPS allocator keeps a block per
                # size it has seen: left alone the pool grows until every step pages.
                if device == "mps":
                    torch.mps.empty_cache()
                done_here = step - start_step
                spd = (time.time() - t0) / max(1, done_here)
                eta_min = spd * (total_steps - step) / 60
                print(f"ep{epoch} step {step}/{total_steps} loss {batch_loss:.4f} ({spd:.2f}s/step, ETA {eta_min:.0f} min)", flush=True)
                if wb:
                    wb.log({"train/loss": batch_loss, "train/lr": sched.get_last_lr()[0], "train/s_per_step": spd, "train/eta_min": eta_min}, step=step)
            if step % args.eval_every == 0 or step == total_steps:
                m = evaluate(model, head, tok, val, device, args.batch, args.max_length, layout, proc, image_root, args.modality)
                print(f"  val: ece={m['ece']:.4f} acc={m['accuracy']:.4f} kl={m['kl']:.4f}", flush=True)
                if wb:
                    wb.log({"val/ece": m["ece"], "val/accuracy": m["accuracy"], "val/kl": m["kl"]}, step=step)
                # always keep the latest state for resume
                (out / "latest").mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out / "latest" / "adapter")
                torch.save(head.state_dict(), out / "latest" / "head.pt")
                json.dump({"step": step, "epoch": epoch}, open(out / "latest" / "state.json", "w"))
                # every evaluated step is kept, because which one is best is only known later
                here = out / "steps" / str(step)
                here.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(here / "adapter")
                torch.save(head.state_dict(), here / "head.pt")
                history.append({**m, "step": step})
                chosen = pick_checkpoint(history, args.acc_floor)
                if chosen["step"] != best.get("step"):
                    best = chosen
                    src = out / "steps" / str(best["step"])
                    shutil.rmtree(out / "adapter", ignore_errors=True)
                    shutil.copytree(src / "adapter", out / "adapter")
                    shutil.copy(src / "head.pt", out / "head.pt")
                    print(f"  kept step {best['step']} (ece {best['ece']:.4f}, acc {best['accuracy']:.4f})", flush=True)
                json.dump(
                    {
                        "base": args.model,
                        "hidden": hidden,
                        "head": args.head,
                        "head_size": HEAD_SIZE,
                        "head_dim": HEAD_DIM,
                        "layout": layout,
                        "load_4bit": args.load_4bit,
                        "modality": args.modality,
                        "pointer_tokens": pointer_tokens,
                        "parallel_options": bool(args.parallel_options),
                        "best": best,
                        "history": history,
                        "args": vars(args),
                    },
                    open(out / "config.json", "w"),
                    indent=1,
                )
    print(f"done in {(time.time() - t0) / 60:.1f} min; best {best}")
    if wb:
        wb.summary["best"] = best
        wb.finish()


if __name__ == "__main__":
    main()
