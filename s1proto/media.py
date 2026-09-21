"""Multimodal pieces shared by the trainer, the evaluators, and the service:
the readout heads, media loading (path, data URI, or URL), the chat-template
wrapping that puts a clip or an image in front of a rendered prompt, and the
body forward that returns hidden states for any of the three modalities.

A media state on the wire is {"image": <spec>, "text": <optional>} or
{"audio": <spec>, "text": <optional>}; <spec> is a data URI, an https URL,
or (only where the server allows it) a local path.
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Any

import torch

HEAD_SIZE = 256  # slot head: fixed number of option slots
HEAD_DIM = 256  # pointer head: query/key width
MODALITIES = ("text", "vision", "audio")
MEDIA_KEYS = {"image": "vision", "audio": "audio"}
CAPTION = {"vision": "See the image.", "audio": "Listen to the audio."}
MAX_MEDIA_BYTES = 25 * 1024 * 1024
USER_AGENT = "s1proto-media/1.0 (+https://decisioncircuits.com)"


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


# --- media ---------------------------------------------------------------------


MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


def is_media_spec(spec: Any) -> bool:
    """The forms a media state actually carries on the wire: raw bytes, a data URI,
    or an http(s) URL. Unambiguous — no ordinary data record holds one of these."""
    if isinstance(spec, bytes):
        return True
    if not isinstance(spec, str):
        return False
    return bool(_DATA_URI.match(spec)) or spec.startswith(("http://", "https://"))


def looks_like_media(spec: Any) -> bool:
    """A spec, or a bare filename. A filename is meant as media even though the
    server will refuse to read a local path, so the caller gets that error rather
    than silence — but only when nothing else in the state contradicts it."""
    return is_media_spec(spec) or (isinstance(spec, str) and spec.lower().endswith(MEDIA_SUFFIXES))


def split_media_state(state: Any) -> tuple[Any, str, Any] | None:
    """(text_state, modality, media_spec) when `state` is a media state, else None.

    A media state is {"image"|"audio": <spec>} with an optional "text". Ordinary
    data is not: {"image": "cover.jpg", "id": 7} is a row about an image, not an
    image, and is scored as the JSON it is. The two are told apart by the other
    keys, and where that is ambiguous, by whether the value is a spec at all."""
    if not isinstance(state, dict):
        return None
    keys = [k for k in MEDIA_KEYS if k in state]
    if not keys:
        return None
    if len(keys) > 1:
        if all(looks_like_media(state[k]) for k in keys):
            raise ValueError("a state carries one of image or audio, not both")
        return None
    k = keys[0]
    if not looks_like_media(state[k]):
        return None
    extra = sorted(x for x in state if x not in (k, "text"))
    if extra:
        # A data URI or URL beside other fields is a malformed media state, worth
        # saying so. A filename beside other fields is a row in a catalogue.
        if is_media_spec(state[k]):
            raise ValueError(f"a media state holds {k!r} and an optional 'text'; unexpected keys {extra}")
        return None
    return state.get("text") or CAPTION[MEDIA_KEYS[k]], MEDIA_KEYS[k], state[k]


_DATA_URI = re.compile(r"^data:([\w.+-]+/[\w.+-]+)?(;base64)?,(.*)$", re.DOTALL)


def media_bytes(spec: Any, root: Path | None = None, allow_paths: bool = False) -> bytes:
    """Bytes of an image or clip from a data URI, an http(s) URL, or a local path."""
    if isinstance(spec, bytes):
        return spec
    if not isinstance(spec, str):
        raise TypeError("media must be a data URI, a URL, or a path string")
    m = _DATA_URI.match(spec)
    if m:
        payload = m.group(3)
        data = base64.b64decode(payload) if m.group(2) else payload.encode()
        if len(data) > MAX_MEDIA_BYTES:
            raise ValueError("media larger than 25 MB")
        return data
    if spec.startswith(("http://", "https://")):
        import urllib.request

        # A named User-Agent: bare urllib gets a 403 from anything behind a bot check,
        # our own CDN included, which made a hosted clip unfetchable by its own URL.
        req = urllib.request.Request(spec, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(MAX_MEDIA_BYTES + 1)
        if len(data) > MAX_MEDIA_BYTES:
            raise ValueError("media larger than 25 MB")
        return data
    if allow_paths or root is not None:
        p = Path(spec) if root is None or os.path.isabs(spec) else root / spec
        return p.read_bytes()
    raise ValueError("media must be a data URI or an http(s) URL")


def load_image(spec: Any, root: Path | None = None, allow_paths: bool = False):
    from PIL import Image

    if not isinstance(spec, str | bytes):  # an already-open image
        return spec.convert("RGB")
    return Image.open(io.BytesIO(media_bytes(spec, root, allow_paths))).convert("RGB")


def read_audio(path: Path | str, sr: int = 16000):
    """Mono float32 at 16 kHz from a file on disk."""
    return decode_audio(Path(path).read_bytes(), sr)


def decode_audio(data: bytes, sr: int = 16000):
    import soundfile as sf

    arr, rate = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if rate != sr:
        import librosa

        arr = librosa.resample(arr, orig_sr=rate, target_sr=sr)
    return arr


def load_audio(spec: Any, root: Path | None = None, allow_paths: bool = False):
    if not isinstance(spec, str | bytes):  # an array already
        return spec
    return decode_audio(media_bytes(spec, root, allow_paths))


def load_media(modality: str, spec: Any, root: Path | None = None, allow_paths: bool = False):
    return load_image(spec, root, allow_paths) if modality == "vision" else load_audio(spec, root, allow_paths)


# --- model plumbing ------------------------------------------------------------


def chat_text(proc, text: str, modality: str) -> str:
    """The base's chat template with the media slot in front of the rendered prompt."""
    part = {"type": "image"} if modality == "vision" else {"type": "audio", "audio_url": "clip.wav"}
    messages = [{"role": "user", "content": [part, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)


def encode(proc, texts: list[str], media: list[Any], modality: str):
    """Processor call for a batch of already chat-wrapped texts and loaded media."""
    if modality == "vision":
        return proc(text=texts, images=media, return_tensors="pt", padding=True)
    if modality == "audio":
        return proc(text=texts, audio=media, sampling_rate=16000, return_tensors="pt", padding=True)
    raise ValueError(modality)


def load_base(model_id: str, modality: str, dtype=torch.bfloat16):
    """The right AutoModel for a modality; the processor for non-text ones."""
    if modality == "vision":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype), AutoProcessor.from_pretrained(model_id)
    if modality == "audio":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        return Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=dtype), AutoProcessor.from_pretrained(model_id)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype), None


def hidden_states(model, enc, modality: str = "text"):
    """Hidden states [B, L, H] without lm_head, which would otherwise
    materialize B x L x vocab logits (the single largest activation, and
    unused: the answer head reads the hidden state). For vision and audio
    the body is the multimodal model (encoder + projector + language model)."""
    body = model.get_base_model().model  # LoRA layers are injected in place
    if modality == "vision":
        keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    elif modality == "audio":
        keep = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")
    else:
        keep = ("input_ids", "attention_mask", "position_ids")  # position_ids only when options are encoded side by side
    return body(**{k: v for k, v in enc.items() if k in keep}, use_cache=False).last_hidden_state


def head_logits(head, hs, nopts, opt_pos, dec_pos):
    """Apply either head at the decide position. Slot: linear over 256
    slots. Pointer: decide token against each option's closing delimiter."""
    h_dec = hs[torch.arange(hs.shape[0], device=hs.device), dec_pos].float()
    if isinstance(head, PointerHead):
        idx = opt_pos.unsqueeze(-1).expand(-1, -1, hs.shape[-1])
        h_opts = torch.gather(hs, 1, idx).float()
        return head(h_dec, h_opts, nopts)
    return head(h_dec, nopts)
