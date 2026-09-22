"""Score an audio-grid JSONL, either with a raw audio-language model by
letter logits (Phase 1) or with a trained run's pointer head.

    uv run python scripts/eval_audio.py data/audio/grid/eval.jsonl --out results/agrid_qwen2audio_raw.json
    uv run python scripts/eval_audio.py data/audio/grid/eval.jsonl --lora runs/circuit-audio-7b --out results/agrid_circuit-audio-7b.json

State is {"audio": <path relative to the JSONL's directory>, "text": <optional>}.
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

from audio_probe import LETTERS, letter_ids, load, read_audio
from eval_set import summarize
from eval_vision import option_texts

from s1proto.data.teachers import option_keys


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument(
        "--lora", default=None, help="a trained audio run dir (adapter/, head.pt, config.json); scores with its pointer head instead of letter logits"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    base_dir = Path(args.data).parent
    items = [json.loads(line) for line in open(args.data)]
    if args.limit:
        items = items[: args.limit]
    if args.lora:
        preds, lat, model_name = score_lora(args.lora, items, base_dir, args.device)
        finish(args, items, preds, lat, model_name)
        return
    proc, model = load(args.model, args.device)
    ids_by_letter = letter_ids(proc)
    preds, lat = [], []
    for i, it in enumerate(items):
        keys = option_keys(it["question"])
        audio = read_audio(base_dir / it["state"]["audio"])
        opts = option_texts(it["question"])
        lines = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        text = (
            it["state"].get("text", "") + "\n\n" if it["state"].get("text") else ""
        ) + f"{it['question']['instructions']}\n\nOptions:\n{lines}\n\nAnswer with the single letter of the best option."
        messages = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=prompt, audio=[audio], sampling_rate=16000, return_tensors="pt", padding=True).to(args.device)
        t0 = time.perf_counter()
        full = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
        lp = torch.stack([torch.logsumexp(full[ids], dim=0) for ids in ids_by_letter[: len(keys)]])
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(torch.softmax(lp, dim=-1).tolist())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}  {sum(lat) / len(lat):.0f} ms/item", flush=True)
    finish(args, items, preds, lat, args.model)


def finish(args, items, preds, lat, model_name: str) -> None:
    summary = summarize(items, preds)
    timing = {
        "device": args.device,
        "items": len(items),
        "ms_per_item": round(sum(lat) / len(lat), 1),
        "batch": 1,
        "note": "pointer head" if args.lora else "letter logits on a raw audio-language model",
    }
    result = {"model": model_name, "data": args.data, "timing": timing, "metrics": summary}
    print("timing:", json.dumps(timing))
    for k, v in summary.items():
        if k.startswith("family:"):
            print(f"  {k[7:].replace(' (heldout)', ''):<24} acc={v['accuracy']:.3f} ece={v['ece']:.3f} conf={v['mean_conf']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


@torch.inference_mode()
def score_lora(run_dir: str, items: list, base_dir: Path, device: str):
    """Score with a trained audio run: the trainer's own batching and head, one item at a time."""
    import random

    from peft import PeftModel
    from train_lora import PointerHead, SlotHead, build_batch, head_logits, hidden_states
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    from s1proto import template as T

    cfg = json.load(open(Path(run_dir) / "config.json"))
    if cfg.get("parallel_options"):
        # the run was trained with options side by side; evaluate it the same way
        import train_lora

        train_lora.PARALLEL_OPTIONS = torch.bfloat16
    if cfg.get("pointer_tokens"):
        T.use_pointer_tokens(*cfg["pointer_tokens"])
    proc = AutoProcessor.from_pretrained(cfg["base"])
    tok = proc.tokenizer
    tok.padding_side = "left"
    base = Qwen2AudioForConditionalGeneration.from_pretrained(cfg["base"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(Path(run_dir) / "adapter")).to(device).eval()
    head = PointerHead(cfg["hidden"], cfg.get("head_dim", 256)) if cfg.get("head") == "pointer" else SlotHead(cfg["hidden"])
    head.load_state_dict(torch.load(Path(run_dir) / "head.pt", map_location="cpu"))
    head.to(device).eval()
    preds, lat = [], []
    rng = random.Random(0)
    for i, it in enumerate(items):
        t0 = time.perf_counter()
        enc, _ref, nopts, opt_pos, dec_pos = build_batch(
            tok, [it], rng, device, 4096, train=False, layout=cfg.get("layout", "pointer"), proc=proc, image_root=base_dir, modality="audio"
        )
        logits = head_logits(head, hidden_states(model, enc, "audio"), nopts, opt_pos, dec_pos)[0, : int(nopts[0])]
        preds.append(torch.softmax(logits, dim=-1).tolist())
        lat.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}  {sum(lat) / len(lat):.0f} ms/item", flush=True)
    return preds, lat, f"lora:{Path(run_dir).name}"


if __name__ == "__main__":
    main()
