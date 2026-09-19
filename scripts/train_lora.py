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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from s1proto import template as T
from s1proto.data.teachers import option_keys
from s1proto.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1proto.template import render

HEAD_SIZE = 256  # slot head: fixed number of option slots
HEAD_DIM = 256  # pointer head: query/key width


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


class SlotHead(torch.nn.Module):
    def __init__(self, hidden: int, n_out: int = HEAD_SIZE):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, n_out)

    def forward(self, h: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h)  # [B, 256]
        mask = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0) >= n_options.unsqueeze(1)
        return logits.masked_fill(mask, float("-inf"))


class PointerHead(torch.nn.Module):
    """Kev-style readout: a query from the decide token's hidden state,
    a key from each option's closing-delimiter hidden state, scaled dot
    product, softmax over the options. Order-invariant, no option cap,
    and options interact only through the softmax."""

    def __init__(self, hidden: int, dim: int = HEAD_DIM):
        super().__init__()
        self.q = torch.nn.Linear(hidden, dim, bias=False)
        self.k = torch.nn.Linear(hidden, dim, bias=False)
        self.scale = dim**-0.5

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
        q = self.q(h_decide).unsqueeze(1)  # [B, 1, d]
        k = self.k(h_opts)  # [B, maxn, d]
        logits = (q * k).sum(-1) * self.scale  # [B, maxn]
        mask = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0) >= n_options.unsqueeze(1)
        return logits.masked_fill(mask, float("-inf"))


def ece15(confs, correct, bins=15):
    n = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0.0)]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return e


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
        if modality == "vision":  # the image is the state; any text rides along as a caption
            from PIL import Image

            p = render(it["state"].get("text") or "See the image.", parse_question(q), layout=layout)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p.text}]}]
            texts.append(proc.apply_chat_template(messages, add_generation_prompt=False, tokenize=False))
            media.append(Image.open(Path(image_root) / it["state"]["image"]).convert("RGB"))
        elif modality == "audio":  # the clip is the state
            p = render(it["state"].get("text") or "Listen to the audio.", parse_question(q), layout=layout)
            messages = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": p.text}]}]
            texts.append(proc.apply_chat_template(messages, add_generation_prompt=False, tokenize=False))
            media.append(read_audio(Path(image_root) / it["state"]["audio"]))
        else:
            texts.append(render(it["state"], parse_question(q), layout=layout).text)
        refs.append(ref)
        nopts.append(len(ref))
    if modality == "vision":
        enc = proc(text=texts, images=media, return_tensors="pt", padding=True)
    elif modality == "audio":
        enc = proc(text=texts, audio=media, sampling_rate=16000, return_tensors="pt", padding=True)
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
    enc = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
    return enc, ref_t.to(device), torch.tensor(nopts, device=device), opt_pos.to(device), dec_pos.to(device)


def read_audio(path: Path, sr: int = 16000):
    """Mono float32 at 16 kHz, which is what the audio processors expect."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != sr:
        import librosa

        data = librosa.resample(data, orig_sr=rate, target_sr=sr)
    return data


def hidden_states(model, enc, modality="text"):
    """Hidden states [B, L, H] without lm_head, which would otherwise
    materialize B x L x vocab logits (the single largest activation, and
    unused: the answer head reads the hidden state). For vision the body
    is the multimodal model (vision tower + language model)."""
    body = model.get_base_model().model  # Qwen3Model / Qwen3VLModel (LoRA layers are injected in place)
    if modality == "vision":
        return body(
            **{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}, use_cache=False
        ).last_hidden_state
    if modality == "audio":  # Qwen2AudioModel: audio tower + projector + language model
        return body(
            **{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "input_features", "feature_attention_mask")}, use_cache=False
        ).last_hidden_state
    return body(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).last_hidden_state


def head_logits(head, hs, nopts, opt_pos, dec_pos):
    """Apply either head at the decide position. Slot: linear over 256
    slots. Pointer: decide token against each option's closing delimiter."""
    h_dec = hs[torch.arange(hs.shape[0], device=hs.device), dec_pos].float()
    if isinstance(head, PointerHead):
        idx = opt_pos.unsqueeze(-1).expand(-1, -1, hs.shape[-1])
        h_opts = torch.gather(hs, 1, idx).float()
        return head(h_dec, h_opts, nopts)
    return head(h_dec, nopts)


def soft_ce(logits: torch.Tensor, ref: torch.Tensor, nopts: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets over the first n options of each row."""
    logp = F.log_softmax(logits[:, : ref.shape[1]], dim=-1)
    logp = torch.nan_to_num(logp, neginf=0.0)  # masked slots have ref 0 anyway
    return -(ref * logp).sum(dim=-1).mean()


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

    best = {"ece": float("inf")}
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
            enc, ref, nopts, opt_pos, dec_pos = build_batch(
                tok, chunk, rng, device, args.max_length, train=True, layout=layout, proc=proc, image_root=image_root, modality=args.modality
            )
            logits = head_logits(head, hidden_states(model, enc, args.modality), nopts, opt_pos, dec_pos)
            loss = soft_ce(logits, ref, nopts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                done_here = step - start_step
                spd = (time.time() - t0) / max(1, done_here)
                eta_min = spd * (total_steps - step) / 60
                print(f"ep{epoch} step {step}/{total_steps} loss {loss.item():.4f} ({spd:.2f}s/step, ETA {eta_min:.0f} min)", flush=True)
                if wb:
                    wb.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0], "train/s_per_step": spd, "train/eta_min": eta_min}, step=step)
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
                if m["ece"] < best["ece"]:
                    best = {**m, "step": step}
                    model.save_pretrained(out / "adapter")
                    torch.save(head.state_dict(), out / "head.pt")
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
                            "best": best,
                            "args": vars(args),
                        },
                        open(out / "config.json", "w"),
                        indent=1,
                    )
                    print(f"  saved (best ece {best['ece']:.4f} @ step {step})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; best {best}")
    if wb:
        wb.summary["best"] = best
        wb.finish()


if __name__ == "__main__":
    main()
