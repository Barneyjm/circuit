"""The audio grid: synthesized clips with code-computed labels.

Same idea as the text and vision grids: span the structure of judgments
about sound rather than a catalogue of sounds, and know every label by
construction. Speech is synthesized from scripted templates with the
Kokoro-82M voices (Apache 2.0; the WAVs travel with the data), non-speech is generated with numpy. About 8% of items
are made undecidable (the key words drowned in noise, or the clip cut
before they arrive) with a soft label.

Formats (columns)                       Operations (rows)
  call     a scripted support call        classify     which kind of call is this?              choice
  list     a spoken enumeration           extract      what amount / number did they say?       choice
  numbers  a spoken readback              count        how many items / beeps?                  choice
  sounds   beeps, tones, noise            compare      which amount was larger / said first?    noul
  speech   a real LibriSpeech recording   consistency  does the claim match what was said?      noul
  digits   real people saying digits      negation     did they NOT mention X?                  noul
                                          mention      is the word X spoken?                    noul
                                          which        which sentence was read?                 choice
                                          order        is A said before B?                      noul

Output: data/audio/grid/<split>/<id>.wav plus a JSONL whose `state` is
{"audio": <relative path>, "text": <optional caption>}.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SR = 16000
# Kokoro-82M (Apache 2.0) voices, American and British, both sexes. The model runs on the laptop.
VOICES = ["af_heart", "af_bella", "af_sarah", "af_nicole", "am_adam", "am_michael", "am_fenrir", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
_PIPELINES: dict[str, Any] = {}

CATEGORIES = {
    "billing": [
        "my bill this month is way higher than usual",
        "I was charged twice on my last statement",
        "I need to set up a payment plan for my balance",
        "there is a late fee on my account that I don't understand",
    ],
    "metering": [
        "my meter reading looks wrong compared to last month",
        "nobody has read my meter in three months",
        "the new smart meter is showing zero usage",
        "I think my meter is running when nothing is on",
    ],
    "taste and odor": [
        "the water smells like rotten eggs",
        "there is a chlorine taste that started yesterday",
        "the tap water is cloudy and tastes metallic",
        "my water has had a musty smell all week",
    ],
    "outage": [
        "I have no water pressure at all this morning",
        "the water has been off since six a.m.",
        "we lost service on the whole street",
        "can you tell me when the water will be back on",
    ],
    "leak": [
        "there is water bubbling up in the street outside",
        "a pipe burst in front of my house",
        "my driveway is flooding from a broken main",
        "water is pouring out of the hydrant on the corner",
    ],
}
ITEMS = ["milk", "eggs", "bread", "coffee", "apples", "rice", "butter", "cheese", "pasta", "bananas", "juice", "yogurt"]
CITIES = ["Dayton", "Portland", "Trenton", "Salem", "Reno", "Camden", "Boulder", "Athens"]


@dataclass
class AItem:
    cell: str
    kind: str
    audio: np.ndarray
    caption: str | None
    question: dict[str, Any]
    ref: dict[str, float]
    ambiguous: bool = False


def onehot(keys: list[str], k: str) -> dict[str, float]:
    return {x: (1.0 if x == k else 0.0) for x in keys}


def uniform(keys: list[str]) -> dict[str, float]:
    return {x: 1.0 / len(keys) for x in keys}


def noul(p: float) -> dict[str, float]:
    return {"yes": p, "no": 1.0 - p}


# --- synthesis ---------------------------------------------------------------


TTS = os.environ.get("S1_TTS", "voxcpm")  # voxcpm: clone a random LibriSpeech speaker; kokoro: stock voices
VOXCPM_PY = os.environ.get("VOXCPM_PY", str(Path.home() / "Documents/code/VoxCPM/.venv/bin/python"))
_BRIDGE: Any = None


def _bridge():
    """The VoxCPM subprocess (its own venv), started on first use."""
    global _BRIDGE
    if _BRIDGE is None:
        import subprocess

        _BRIDGE = subprocess.Popen(
            [VOXCPM_PY, str(Path(__file__).resolve().parents[2] / "scripts" / "voxcpm_bridge.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        while True:
            line = _BRIDGE.stdout.readline()
            if not line:
                raise RuntimeError("VoxCPM bridge failed to start")
            if line.startswith("{") and json.loads(line).get("ready"):
                break
    return _BRIDGE


def speak_voxcpm(text: str, rng: random.Random) -> np.ndarray:
    """VoxCPM cloning the voice of a random LibriSpeech reader (a paired clip and transcript as the prompt)."""
    import tempfile

    import librosa

    ref = rng.choice(libri())
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    b = _bridge()
    b.stdin.write(
        json.dumps(
            {
                "text": text,
                "prompt_wav": str(SOURCES / "librispeech" / ref["file"]),
                "prompt_text": ref["text"],
                "out": out,
                "steps": 12,
                "seed": rng.randrange(1 << 30),
            }
        )
        + "\n"
    )
    b.stdin.flush()
    resp = json.loads(b.stdout.readline())
    if "error" in resp:
        raise RuntimeError(resp["error"])
    data, sr = sf.read(out, dtype="float32")
    Path(out).unlink()
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    peak = float(np.abs(data).max()) or 1.0
    return (data / peak * 0.8).astype(np.float32)


def speak(text: str, rng: random.Random, voice: str | None = None) -> np.ndarray:
    """Natural speech at 16 kHz, peak-normalized. VoxCPM by default (see TTS), Kokoro stock voices otherwise."""
    import librosa

    if TTS == "voxcpm" and voice is None:
        return speak_voxcpm(text, rng)
    voice = voice or rng.choice(VOICES)
    lang = "b" if voice.startswith("b") else "a"
    if lang not in _PIPELINES:
        from kokoro import KPipeline

        _PIPELINES[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    speed = rng.uniform(0.9, 1.12)
    parts = [a for _gs, _ps, a in _PIPELINES[lang](text, voice=voice, speed=speed)]
    data = np.concatenate([np.asarray(a, dtype=np.float32) for a in parts])
    data = librosa.resample(data, orig_sr=24000, target_sr=SR)
    peak = float(np.abs(data).max()) or 1.0
    return (data / peak * 0.8).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def tone(freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.01))
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def beeps(n: int, rng: random.Random) -> np.ndarray:
    f = rng.choice([440, 660, 880, 1000])
    gap = rng.uniform(0.25, 0.5)
    parts = [silence(0.3)]
    for _ in range(n):
        parts += [tone(f, rng.uniform(0.12, 0.25)), silence(gap)]
    return np.concatenate(parts)


def noise(seconds: float, rng: random.Random, amp: float = 0.3, samples: int | None = None) -> np.ndarray:
    g = np.random.default_rng(rng.getrandbits(32))
    return (amp * g.standard_normal(samples if samples is not None else int(seconds * SR))).astype(np.float32)


def drown(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Bury the speech under noise at about -8 dB SNR: audible as a voice, unintelligible."""
    n = noise(0, rng, amp=1.0, samples=len(audio))
    return np.clip(0.25 * audio + 0.6 * n, -1, 1).astype(np.float32)


def cut_before(audio: np.ndarray, frac: float) -> np.ndarray:
    return audio[: int(len(audio) * frac)]


def number_words(n: int) -> str:
    return " ".join(str(n))  # "four two seven one": read digit by digit like a call centre


# --- cells ---------------------------------------------------------------------


def cell_call_classify(rng: random.Random) -> AItem:
    cats = list(CATEGORIES)
    cat = rng.choice(cats)
    line = rng.choice(CATEGORIES[cat])
    text = rng.choice(["Hi, ", "Hello, yes, ", "Good morning. ", "Hi there, "]) + line + rng.choice([".", ", can someone help?", ", what can you do about it?"])
    audio = speak(text, rng)
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {"type": "choice", "instructions": "Which kind of water-utility call is this?", "criteria": {c: None for c in cats}}
    return AItem("classify/call", "choice", audio, None, q, uniform(cats) if amb else onehot(cats, cat), amb)


def cell_call_extract(rng: random.Random) -> AItem:
    amounts = rng.sample([18, 24, 37, 42, 56, 63, 78, 85, 91, 120, 145, 210], 4)
    said = amounts[0]
    text = f"Hi, I'm calling about my bill. It says I owe {said} dollars and that can't be right."
    audio = speak(text, rng)
    amb = rng.random() < 0.08
    if amb:
        audio = cut_before(audio, 0.45)  # cut before the amount is spoken
    keys = [f"${a}" for a in sorted(amounts)]
    q = {"type": "choice", "instructions": "How much does the caller say the bill is?", "criteria": {k: None for k in keys}}
    return AItem("extract/call", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, f"${said}"), amb)


def cell_call_consistency(rng: random.Random) -> AItem:
    cat = rng.choice(list(CATEGORIES))
    city = rng.choice(CITIES)
    line = rng.choice(CATEGORIES[cat])
    audio = speak(f"Hello, this is a customer in {city}. {line[0].upper() + line[1:]}.", rng)
    truth = rng.random() < 0.5
    claim_city = city if truth else rng.choice([c for c in CITIES if c != city])
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {
        "type": "noul",
        "instructions": f"Does the caller say they are in {claim_city}?",
        "criteria": {"true": "the caller names that city", "false": "the caller names a different city or none"},
    }
    return AItem("consistency/call", "noul", audio, None, q, noul(0.5) if amb else noul(1.0 if truth else 0.0), amb)


def cell_call_negation(rng: random.Random) -> AItem:
    cat = rng.choice(list(CATEGORIES))
    line = rng.choice(CATEGORIES[cat])
    mentions_refund = rng.random() < 0.5
    tail = " I want a refund for this." if mentions_refund else " Please look into it."
    audio = speak(f"Hi. {line[0].upper() + line[1:]}.{tail}", rng)
    amb = rng.random() < 0.08
    if amb:
        audio = cut_before(audio, 0.6)  # ends before the caller gets to the request
    q = {
        "type": "noul",
        "instructions": "Does the caller NOT ask for a refund?",
        "criteria": {"true": "no refund is requested", "false": "the caller asks for a refund"},
    }
    return AItem("negation/call", "noul", audio, None, q, noul(0.5) if amb else noul(0.0 if mentions_refund else 1.0), amb)


def cell_list_count(rng: random.Random) -> AItem:
    n = rng.randint(1, 5)
    items = rng.sample(ITEMS, n)
    spoken = items[0] if n == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    audio = speak(rng.choice(["I need ", "Can you pick up ", "We're out of ", "Add to the order: "]) + spoken + ".", rng)
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    keys = ["1", "2", "3", "4", "5"]
    q = {"type": "choice", "instructions": "How many different items does the speaker list?", "criteria": {k: None for k in keys}}
    return AItem("count/list", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, str(n)), amb)


def cell_list_negation(rng: random.Random) -> AItem:
    items = rng.sample(ITEMS, rng.randint(2, 4))
    probe = rng.choice(ITEMS)
    absent = probe not in items
    audio = speak("Please get " + ", ".join(items[:-1]) + " and " + items[-1] + ".", rng)
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {
        "type": "noul",
        "instructions": f"Is {probe} absent from the list?",
        "criteria": {"true": f"{probe} is not mentioned", "false": f"{probe} is mentioned"},
    }
    return AItem("negation/list", "noul", audio, None, q, noul(0.5) if amb else noul(1.0 if absent else 0.0), amb)


def cell_numbers_extract(rng: random.Random) -> AItem:
    base = rng.randint(1000, 9999)
    said = base
    digits = list(str(base))
    others = set()
    while len(others) < 3:
        d = digits[:]
        i = rng.randrange(4)
        d[i] = str((int(d[i]) + rng.randint(1, 9)) % 10)
        cand = int("".join(d))
        if cand != said:
            others.add(cand)
    audio = speak(f"My account number is {number_words(said)}.", rng)
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    keys = [str(k) for k in sorted([said, *others])]
    q = {"type": "choice", "instructions": "Which account number does the speaker read out?", "criteria": {k: None for k in keys}}
    return AItem("extract/numbers", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, str(said)), amb)


def cell_numbers_compare(rng: random.Random) -> AItem:
    a, b = rng.sample(range(10, 200), 2)
    audio = speak(f"Last month I paid {a} dollars, and this month the bill is {b} dollars.", rng)
    amb = rng.random() < 0.08
    if amb:
        b2 = a  # equal amounts: the question has no answer
        audio = speak(f"Last month I paid {a} dollars, and this month the bill is {b2} dollars.", rng)
    q = {
        "type": "noul",
        "instructions": "Is this month's bill higher than last month's?",
        "criteria": {"true": "this month is the larger amount", "false": "this month is smaller"},
    }
    return AItem("compare/numbers", "noul", audio, None, q, noul(0.5) if amb else noul(1.0 if b > a else 0.0), amb)


def cell_sounds_count(rng: random.Random) -> AItem:
    n = rng.randint(1, 5)
    audio = beeps(n, rng)
    amb = rng.random() < 0.08
    if amb:
        audio = np.clip(audio + noise(0, rng, amp=0.9, samples=len(audio)), -1, 1).astype(np.float32)
    keys = ["1", "2", "3", "4", "5"]
    q = {"type": "choice", "instructions": "How many beeps are in the clip?", "criteria": {k: None for k in keys}}
    return AItem("count/sounds", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, str(n)), amb)


def cell_sounds_classify(rng: random.Random) -> AItem:
    kinds = ["speech", "beeps", "noise", "silence"]
    kind = rng.choice(kinds)
    if kind == "speech":
        audio = load_clip(SOURCES / "librispeech" / rng.choice(libri())["file"])[: int(4 * SR)]
    elif kind == "beeps":
        audio = beeps(rng.randint(2, 5), rng)
    elif kind == "noise":
        audio = noise(rng.uniform(2, 4), rng)
    else:
        audio = silence(rng.uniform(2, 4))
    amb = kind == "speech" and rng.random() < 0.2
    if amb:  # speech buried so deep it could pass for noise
        audio = np.clip(0.08 * audio + 0.7 * noise(0, rng, amp=1.0, samples=len(audio)), -1, 1).astype(np.float32)
    q = {
        "type": "choice",
        "instructions": "What is in the clip?",
        "criteria": {"speech": "a person talking", "beeps": "tones or beeps", "noise": "static or hiss", "silence": "nothing audible"},
    }
    ref = {"speech": 0.5, "beeps": 0.0, "noise": 0.5, "silence": 0.0} if amb else onehot(kinds, kind)
    return AItem("classify/sounds", "choice", audio, None, q, ref, amb)


# --- real recordings ---------------------------------------------------------------
# LibriSpeech (CC BY 4.0): read sentences with transcripts, so questions about what was
# said are labeled from the transcript. Free Spoken Digit Dataset (CC BY-SA 4.0): real
# people saying digits, stitched into account-number readbacks.

SOURCES = Path(__file__).resolve().parents[2] / "data" / "audio" / "sources"
STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "is",
        "was",
        "were",
        "be",
        "been",
        "are",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "his",
        "her",
        "their",
        "our",
        "your",
        "not",
        "no",
        "as",
        "but",
        "if",
        "so",
        "than",
        "then",
        "there",
        "here",
        "which",
        "what",
        "who",
        "whom",
        "when",
        "where",
        "how",
        "all",
        "any",
        "some",
        "such",
        "very",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "more",
        "most",
    ]
)
_LIBRI: list[dict[str, Any]] | None = None
_FSDD: dict[str, list[Path]] | None = None


def libri() -> list[dict[str, Any]]:
    global _LIBRI
    if _LIBRI is None:
        _LIBRI = json.loads((SOURCES / "librispeech" / "index.json").read_text())
    return _LIBRI


def fsdd() -> dict[str, list[Path]]:
    """speaker -> digit clips (file names are <digit>_<speaker>_<take>.wav)."""
    global _FSDD
    if _FSDD is None:
        _FSDD = {}
        for p in sorted((SOURCES / "fsdd" / "recordings").glob("*.wav")):
            _FSDD.setdefault(p.stem.split("_")[1], []).append(p)
    return _FSDD


def load_clip(path: Path) -> np.ndarray:
    import librosa

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    peak = float(np.abs(data).max()) or 1.0
    return (data / peak * 0.8).astype(np.float32)


def content_words(text: str) -> list[str]:
    seen, out = set(), []
    for w in text.replace("'", "").split():
        w = "".join(ch for ch in w if ch.isalpha())
        if len(w) >= 5 and w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def cell_speech_mention(rng: random.Random) -> AItem:
    rows = libri()
    row = rng.choice(rows)
    words = content_words(row["text"])
    if len(words) < 2:
        return cell_speech_mention(rng)
    present = rng.random() < 0.5
    if present:
        word = rng.choice(words)
    else:
        other = content_words(rng.choice(rows)["text"])
        cands = [w for w in other if w not in row["text"].split()]
        if not cands:
            return cell_speech_mention(rng)
        word = rng.choice(cands)
    audio = load_clip(SOURCES / "librispeech" / row["file"])
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {
        "type": "noul",
        "instructions": f'Does the speaker say the word "{word}"?',
        "criteria": {"true": "the word is spoken in the clip", "false": "it is not"},
    }
    return AItem("mention/speech", "noul", audio, None, q, noul(0.5) if amb else noul(1.0 if present else 0.0), amb)


def cell_speech_which(rng: random.Random) -> AItem:
    rows = libri()
    row = rng.choice(rows)
    others = rng.sample([r for r in rows if r is not row], 3)
    snippet = lambda r: " ".join(r["text"].split()[:9]) + ("…" if len(r["text"].split()) > 9 else "")
    keys = [snippet(r) for r in [row, *others]]
    rng.shuffle(keys)
    audio = load_clip(SOURCES / "librispeech" / row["file"])
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {"type": "choice", "instructions": "Which of these did the speaker read?", "criteria": {k: None for k in keys}}
    return AItem("which/speech", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, snippet(row)), amb)


def cell_speech_order(rng: random.Random) -> AItem:
    rows = libri()
    row = rng.choice(rows)
    words = content_words(row["text"])
    if len(words) < 3:
        return cell_speech_order(rng)
    a, b = rng.sample(words, 2)
    toks = row["text"].split()
    first = min(i for i, t in enumerate(toks) if a in t) < min(i for i, t in enumerate(toks) if b in t)
    audio = load_clip(SOURCES / "librispeech" / row["file"])
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    q = {"type": "noul", "instructions": f'Does the speaker say "{a}" before "{b}"?', "criteria": {"true": f"{a} comes first", "false": f"{b} comes first"}}
    return AItem("order/speech", "noul", audio, None, q, noul(0.5) if amb else noul(1.0 if first else 0.0), amb)


def cell_digits_extract(rng: random.Random) -> AItem:
    """A real person reading four digits, one clip per digit, one speaker."""
    speaker = rng.choice(list(fsdd()))
    clips = fsdd()[speaker]
    by_digit: dict[str, list[Path]] = {}
    for p in clips:
        by_digit.setdefault(p.stem.split("_")[0], []).append(p)
    digits = [rng.choice("0123456789") for _ in range(4)]
    said = "".join(digits)
    parts = [silence(0.25)]
    for d in digits:
        parts += [load_clip(rng.choice(by_digit[d])), silence(rng.uniform(0.15, 0.35))]
    audio = np.concatenate(parts)
    others: set[str] = set()
    while len(others) < 3:
        d = digits[:]
        i = rng.randrange(4)
        d[i] = str((int(d[i]) + rng.randint(1, 9)) % 10)
        if "".join(d) != said:
            others.add("".join(d))
    amb = rng.random() < 0.08
    if amb:
        audio = drown(audio, rng)
    keys = sorted([said, *others])
    q = {"type": "choice", "instructions": "Which four-digit number does the speaker read out?", "criteria": {k: None for k in keys}}
    return AItem("extract/digits", "choice", audio, None, q, uniform(keys) if amb else onehot(keys, said), amb)


CELLS = {
    "classify/call": cell_call_classify,
    "extract/call": cell_call_extract,
    "consistency/call": cell_call_consistency,
    "negation/call": cell_call_negation,
    "count/list": cell_list_count,
    "negation/list": cell_list_negation,
    "extract/numbers": cell_numbers_extract,
    "compare/numbers": cell_numbers_compare,
    "count/sounds": cell_sounds_count,
    "classify/sounds": cell_sounds_classify,
    "mention/speech": cell_speech_mention,
    "which/speech": cell_speech_which,
    "order/speech": cell_speech_order,
    "extract/digits": cell_digits_extract,
}


def generate(per_cell: int, seed: int, split: str, out_dir: Path) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}-{split}-audio")
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cell, fn in CELLS.items():
        for _ in range(per_cell):
            it = fn(rng)
            digest = hashlib.sha1(it.audio.tobytes()).hexdigest()[:12]
            rel = f"{split}/{it.cell.replace('/', '-')}-{digest}.wav"
            sf.write(out_dir / rel, it.audio, SR, subtype="PCM_16")
            keys = list(it.ref)
            rows.append(
                {
                    "id": f"agrid-{it.cell.replace('/', '-')}-{digest}",
                    "family": (f"ambiguous/{it.cell.split('/')[0]}" if it.ambiguous else it.cell),
                    "operation": it.cell.split("/")[0],
                    "format": it.cell.split("/")[1],
                    "kind": it.kind,
                    "heldout": split == "eval",
                    "ambiguous": it.ambiguous,
                    "state": {"audio": rel, **({"text": it.caption} if it.caption else {})},
                    "question": it.question,
                    "refs": {"code": {k: it.ref[k] for k in keys}},
                    "ref": {k: it.ref[k] for k in keys},
                    "source": "audio_grid",
                }
            )
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--eval-per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/audio/grid")
    args = ap.parse_args()
    out = Path(args.out)
    for split, n in (("train", args.per_cell), ("eval", args.eval_per_cell)):
        rows = generate(n, args.seed, split, out)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        secs = sum(sf.info(out / r["state"]["audio"]).duration for r in rows)
        print(
            f"{split}: {len(rows)} items, cells {len(Counter(r['family'] for r in rows if not r['ambiguous']))}, ambiguous {sum(r['ambiguous'] for r in rows)}, {secs / 60:.1f} min of audio -> {out / f'{split}.jsonl'}"
        )
