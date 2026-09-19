"""Phase-1 style probing of a raw audio-language model: audio + question +
lettered options, read the next-token distribution over the letters.

    uv run python scripts/audio_probe.py --set data/audio/probe_set.json --model Qwen/Qwen2-Audio-7B-Instruct

The set is [{audio, questions: [{q, options, expect}]}]; `expect` may be
a list of acceptable answers or null for a deliberately ambiguous one.
Audio files are resampled to 16 kHz mono.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load(model_id: str, device: str):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    return proc, model


def letter_ids(proc) -> list[list[int]]:
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


def read_audio(path: Path, sr: int = 16000) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != sr:
        import librosa

        data = librosa.resample(data, orig_sr=rate, target_sr=sr)
    return data


@torch.inference_mode()
def ask(proc, model, audio: np.ndarray, question: str, options: list[str], device: str) -> dict:
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    text = f"{question}\n\nOptions:\n{lines}\n\nAnswer with the single letter of the best option."
    messages = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=prompt, audio=[audio], sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    full = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
    lp = torch.stack([torch.logsumexp(full[ids], dim=0) for ids in letter_ids(proc)[: len(options)]])
    p = torch.softmax(lp, dim=-1).tolist()
    return {"probs": dict(zip(options, p, strict=True)), "letter_mass": math.exp(torch.logsumexp(lp, dim=0).item())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc, model = load(args.model, args.device)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.0f}s")
    items = json.loads(Path(args.set).read_text())
    hits = total = 0
    for it in items:
        audio = read_audio(Path(args.set).with_name(it["audio"]))
        print(f"\n== {it['audio']} ({len(audio) / 16000:.1f}s)")
        for q in it["questions"]:
            t0 = time.perf_counter()
            r = ask(proc, model, audio, q["q"], q["options"], args.device)
            top = max(r["probs"], key=r["probs"].get)
            expect = q.get("expect")
            ok = None if expect is None else top in (expect if isinstance(expect, list) else [expect])
            if ok is not None:
                total += 1
                hits += ok
            mark = "  ?" if ok is None else (" ok" if ok else " XX")
            dist = "  ".join(f"{o}={p:.2f}" for o, p in r["probs"].items())
            print(f" {mark} {q['q']:<50} -> {top:<24} [{dist}]  letters={r['letter_mass']:.2f}  {1000 * (time.perf_counter() - t0):.0f} ms")
    print(f"\n{hits}/{total} decidable questions right")


if __name__ == "__main__":
    main()
