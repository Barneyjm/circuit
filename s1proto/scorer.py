"""Prefill-only logit scoring on a Hugging Face causal LM.

`Scorer.score(prompts)` runs one forward pass over a batch of rendered
prompts, reads the next-token logits at the last real position of each
sequence, keeps only the label tokens (" A", " B", ...), and returns a
softmax per prompt. No generation, no sampling.

Backends:
- `HFScorer`: torch + transformers. Runs on MPS (Apple), CUDA, or CPU.
  This is the Phase 1 development backend on a Mac; vLLM's prefill-only
  path is the intended serving backend on a GPU box and shares this
  module's contract (`ScorerProtocol`).
- `FakeScorer`: deterministic, model-free; for schema/fuzz tests and
  the service test-suite.

Temperature is a per-question-type scalar applied to the label logits
before softmax. Phase 1 uses 1.0 everywhere; Phase 2 fits it.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from s1proto import template as T
from s1proto.template import LETTER_LABELS, Prompt


@dataclass
class ScoreResult:
    probabilities: list[float]  # over the prompt's options, sums to 1
    logits: list[float]  # raw label logits (pre-temperature), for calibration
    input_tokens: int


class ScorerProtocol(Protocol):
    name: str
    max_options: int

    def score(self, prompts: list[Prompt], temperatures: list[float] | None = None) -> list[ScoreResult]: ...


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


@dataclass
class FakeScorer:
    """Hash-seeded, deterministic per prompt text. Produces a peaked but
    not one-hot distribution so confidence/score math is exercised."""

    name: str = "fake"
    max_options: int = 255

    def score(self, prompts: list[Prompt], temperatures: list[float] | None = None) -> list[ScoreResult]:
        out = []
        for i, p in enumerate(prompts):
            h = hashlib.sha256(p.text.encode("utf-8")).digest()
            logits = [((h[j % 32] / 255.0) * 4.0 - 2.0) for j in range(p.n_options)]
            t = (temperatures or [1.0] * len(prompts))[i]
            probs = softmax([x / t for x in logits])
            out.append(ScoreResult(probabilities=probs, logits=logits, input_tokens=max(1, len(p.text) // 4)))
        return out


@dataclass
class HFScorer:
    model_id: str
    device: str | None = None
    dtype: str = "bfloat16"
    max_length: int = 4096
    # Prefill the shared state once per request and run each question's
    # tail against the cached KV. Off = one full sequence per question.
    prefix_cache: bool = True
    name: str = field(init=False)
    max_options: int = field(init=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = os.path.basename(self.model_id.rstrip("/"))
        self.max_options = len(LETTER_LABELS)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = getattr(torch, self.dtype)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Left padding so "the last position" is the same index for
        # every row in the batch.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"  # the label slot is the last token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0

        # Label token ids: the token for " A", " B", ... as it would
        # follow "Answer:". Each must be a single token or the slot
        # read is wrong; assert rather than silently mis-score.
        self.label_ids: list[int] = []
        for label in LETTER_LABELS:
            ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"label {label!r} is not a single token for {self.model_id}: {ids}")
            self.label_ids.append(ids[0])
        self._label_ids_t = torch.tensor(self.label_ids, device=self.device)

    def score(self, prompts: list[Prompt], temperatures: list[float] | None = None) -> list[ScoreResult]:
        if not prompts:
            return []
        too_many = [p.n_options for p in prompts if p.n_options > self.max_options]
        if too_many:
            raise ValueError(f"label-token scorer supports at most {self.max_options} options, got {max(too_many)}; use a head-based (lora:) scorer")
        if self.prefix_cache and len(prompts) > 1 and all(p.prefix == prompts[0].prefix for p in prompts) and prompts[0].prefix:
            return self._score_shared_prefix(prompts, temperatures)
        return self._score_independent(prompts, temperatures)

    def _score_shared_prefix(self, prompts: list[Prompt], temperatures: list[float] | None) -> list[ScoreResult]:
        """Prefill the shared state once, then run only each question's
        tail against the cached KV. Same numbers as the independent path
        up to bf16 noise; cost is O(state + sum(tails)) instead of
        O(n_questions * state)."""
        import torch

        temps = temperatures or [1.0] * len(prompts)
        n = len(prompts)
        prefix_ids = self.tokenizer.encode(prompts[0].prefix, add_special_tokens=False)
        tails = [self.tokenizer.encode(p.tail, add_special_tokens=False) for p in prompts]
        max_tail = max(len(t) for t in tails)
        pad_id = self.tokenizer.pad_token_id

        with torch.inference_mode():
            pre = torch.tensor([prefix_ids], device=self.device)
            out = self.model(input_ids=pre, use_cache=True)
            past = out.past_key_values
            # Expand the single-row cache to n rows.
            past.batch_repeat_interleave(n)

            # Right-pad tails; the answer slot for row i is at position
            # len(tails[i]) - 1, read via gather below. Attention mask
            # covers prefix (all ones) + tail (ones then zeros).
            tail_ids = torch.full((n, max_tail), pad_id, device=self.device)
            tail_mask = torch.zeros((n, max_tail), dtype=torch.long, device=self.device)
            for i, t in enumerate(tails):
                tail_ids[i, : len(t)] = torch.tensor(t, device=self.device)
                tail_mask[i, : len(t)] = 1
            attn = torch.cat([torch.ones((n, len(prefix_ids)), dtype=torch.long, device=self.device), tail_mask], dim=1)
            pos = torch.arange(len(prefix_ids), len(prefix_ids) + max_tail, device=self.device).unsqueeze(0).expand(n, -1)
            out = self.model(input_ids=tail_ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            last_idx = torch.tensor([len(t) - 1 for t in tails], device=self.device)
            last = out.logits[torch.arange(n, device=self.device), last_idx, :].float()
            label_logits = last.index_select(1, self._label_ids_t).cpu()

        results = []
        for i, p in enumerate(prompts):
            raw = label_logits[i, : p.n_options].tolist()
            probs = softmax([x / temps[i] for x in raw])
            results.append(ScoreResult(probabilities=probs, logits=raw, input_tokens=len(prefix_ids) + len(tails[i])))
        return results

    def _score_independent(self, prompts: list[Prompt], temperatures: list[float] | None) -> list[ScoreResult]:
        import torch

        temps = temperatures or [1.0] * len(prompts)
        enc = self.tokenizer(
            [p.text for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            # Last position is the final token for every row thanks to
            # left padding.
            last = out.logits[:, -1, :].float()  # [B, V]
            label_logits = last.index_select(1, self._label_ids_t)  # [B, 26]
        label_logits = label_logits.cpu()
        lengths = attention_mask.sum(dim=1).tolist()

        results = []
        for i, p in enumerate(prompts):
            raw = label_logits[i, : p.n_options].tolist()
            probs = softmax([x / temps[i] for x in raw])
            results.append(ScoreResult(probabilities=probs, logits=raw, input_tokens=int(lengths[i])))
        return results


@dataclass
class LoRAScorer:
    """Phase 3 checkpoint: base model + LoRA adapter + a readout head
    (see scripts/train_lora.py). `pointer` heads score each option's
    closing delimiter against the decide token (no option cap, order
    invariant); `slot` heads read a fixed 256-way linear layer at the
    last position. Callers render prompts with `layout=scorer.layout`."""

    run_dir: str
    device: str | None = None
    max_length: int = 4096
    name: str = field(init=False)
    max_options: int = field(init=False)
    layout: str = field(init=False)

    def __post_init__(self) -> None:
        import json

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = json.load(open(os.path.join(self.run_dir, "config.json")))
        self.name = f"lora:{os.path.basename(self.run_dir.rstrip('/'))}"
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["base"])
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"  # keep the answer slot; the head reads the last position
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if cfg.get("load_4bit"):
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            base = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
            self.model = PeftModel.from_pretrained(base, os.path.join(self.run_dir, "adapter")).eval()
        else:
            base = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16)
            self.model = PeftModel.from_pretrained(base, os.path.join(self.run_dir, "adapter"))
            self.model.to(self.device).eval()
        self.head_kind = cfg.get("head", "slot")
        self.layout = cfg.get("layout", "letters")
        state = torch.load(os.path.join(self.run_dir, "head.pt"), map_location="cpu")
        if self.head_kind == "pointer":
            dim = cfg.get("head_dim", 256)
            self.q = torch.nn.Linear(cfg["hidden"], dim, bias=False)
            self.k = torch.nn.Linear(cfg["hidden"], dim, bias=False)
            self.q.load_state_dict({"weight": state["q.weight"]})
            self.k.load_state_dict({"weight": state["k.weight"]})
            self.q.to(self.device).eval()
            self.k.to(self.device).eval()
            self.scale = dim**-0.5
            self.max_options = 255  # the API cap; the head itself has none
            if cfg.get("pointer_tokens"):
                T.use_pointer_tokens(*cfg["pointer_tokens"])
            self.opt_end_id = self.tokenizer.convert_tokens_to_ids(T.OPT_END)
        else:
            self.head = torch.nn.Linear(cfg["hidden"], cfg["head_size"])
            self.head.load_state_dict({k.replace("proj.", ""): v for k, v in state.items()})
            self.head.to(self.device).eval()
            self.max_options = cfg["head_size"]
        self.prefix_cache = False
        self.load_seconds = time.perf_counter() - t0

    def _head_logits(self, hs, h_last, ids, prompts, offset: int = 0):
        """Pointer: the decide token against each option's closing delimiter.
        Slot: a linear map at the decide position. `offset` shifts delimiter
        positions when `ids` holds only the tail of the sequence."""
        if self.head_kind != "pointer":
            return self.head(h_last).cpu()
        q = self.q(h_last)
        rows = []
        for i, p in enumerate(prompts):
            pos = (ids[i] == self.opt_end_id).nonzero(as_tuple=True)[0]
            if len(pos) != p.n_options:
                raise ValueError(f"found {len(pos)} option delimiters for {p.n_options} options; render with layout='pointer' and check truncation")
            k = self.k(hs[i, pos + offset, :].float())
            rows.append(((q[i] * k).sum(-1) * self.scale).cpu())
        return rows

    def _score_shared_prefix(self, prompts: list[Prompt], temperatures: list[float] | None) -> list[ScoreResult]:
        """One pass over the state, then one short pass per question against the
        cached state. Same numbers as scoring each question's full text, up to
        bf16 noise; cost is O(state + sum of tails) rather than O(questions x state)."""
        import torch

        temps = temperatures or [1.0] * len(prompts)
        n = len(prompts)
        prefix_ids = self.tokenizer.encode(prompts[0].prefix, add_special_tokens=False)
        tails = [self.tokenizer.encode(p.tail, add_special_tokens=False) for p in prompts]
        max_tail = max(len(t) for t in tails)
        pad_id = self.tokenizer.pad_token_id
        if len(prefix_ids) + max_tail > self.max_length:  # truncation would land mid-state; take the plain path
            return self._score_independent(prompts, temperatures)

        with torch.inference_mode():
            decoder = self.model.get_base_model().model  # no lm_head: the answer head reads hidden states
            pre = torch.tensor([prefix_ids], device=self.device)
            past = decoder(input_ids=pre, use_cache=True).past_key_values
            past.batch_repeat_interleave(n)  # one row per question, same cached state

            # Tails are right-padded; each row's decide token sits at len(tail) - 1.
            tail_ids = torch.full((n, max_tail), pad_id, device=self.device)
            tail_mask = torch.zeros((n, max_tail), dtype=torch.long, device=self.device)
            for i, t in enumerate(tails):
                tail_ids[i, : len(t)] = torch.tensor(t, device=self.device)
                tail_mask[i, : len(t)] = 1
            attn = torch.cat([torch.ones((n, len(prefix_ids)), dtype=torch.long, device=self.device), tail_mask], dim=1)
            pos = torch.arange(len(prefix_ids), len(prefix_ids) + max_tail, device=self.device).unsqueeze(0).expand(n, -1)
            hs = decoder(input_ids=tail_ids, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True).last_hidden_state
            last_idx = torch.tensor([len(t) - 1 for t in tails], device=self.device)
            h_last = hs[torch.arange(n, device=self.device), last_idx, :].float()
            logits = self._head_logits(hs, h_last, tail_ids, prompts)

        results = []
        for i, p in enumerate(prompts):
            raw = logits[i][: p.n_options].tolist()
            probs = softmax([x / temps[i] for x in raw])
            results.append(ScoreResult(probabilities=probs, logits=raw, input_tokens=len(prefix_ids) + len(tails[i])))
        return results

    def score(self, prompts: list[Prompt], temperatures: list[float] | None = None, media: list[Any] | None = None) -> list[ScoreResult]:
        if not prompts:
            return []
        # Several questions about one state is the common request, and the state is
        # usually the long part. Encode it once and run the question tails against
        # the cached keys and values instead of re-reading it per question.
        if self.prefix_cache and len(prompts) > 1 and prompts[0].prefix and all(p.prefix == prompts[0].prefix for p in prompts):
            return self._score_shared_prefix(prompts, temperatures)
        return self._score_independent(prompts, temperatures)

    def _score_independent(self, prompts: list[Prompt], temperatures: list[float] | None) -> list[ScoreResult]:
        import torch

        temps = temperatures or [1.0] * len(prompts)
        enc = self.tokenizer([p.text for p in prompts], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            decoder = self.model.get_base_model().model  # skip lm_head; the answer head reads the hidden state
            hs = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            h_last = hs[:, -1, :].float()
            if self.head_kind == "pointer":
                q = self.q(h_last)  # [B, d]
                rows = []
                for i, p in enumerate(prompts):
                    pos = (ids[i] == self.opt_end_id).nonzero(as_tuple=True)[0]
                    if len(pos) != p.n_options:
                        raise ValueError(f"found {len(pos)} option delimiters for {p.n_options} options; render with layout='pointer' and check truncation")
                    k = self.k(hs[i, pos, :].float())  # [n, d]
                    rows.append(((q[i] * k).sum(-1) * self.scale).cpu())
                logits = rows
            else:
                logits = self.head(h_last).cpu()
        lengths = mask.sum(dim=1).tolist()
        results = []
        for i, p in enumerate(prompts):
            raw = logits[i][: p.n_options].tolist()
            probs = softmax([x / temps[i] for x in raw])
            results.append(ScoreResult(probabilities=probs, logits=raw, input_tokens=int(lengths[i])))
        return results


@dataclass
class MultimodalScorer:
    """A vision or audio run (config.json `modality`): the base's encoder
    plus LoRA language model plus the pointer head. `score` takes the
    loaded media for each prompt (the same image or clip for every
    question in a request)."""

    run_dir: str
    device: str | None = None
    max_length: int = 4096
    name: str = field(init=False)
    max_options: int = field(init=False)
    layout: str = field(init=False)
    modality: str = field(init=False)

    def __post_init__(self) -> None:
        import json

        import torch
        from peft import PeftModel

        from s1proto.media import HEAD_SIZE, PointerHead, SlotHead, load_base

        cfg = json.load(open(os.path.join(self.run_dir, "config.json")))
        self.modality = cfg["modality"]
        if self.modality not in ("vision", "audio"):
            raise ValueError(f"{self.run_dir} is a {self.modality} run; use LoRAScorer")
        if cfg.get("pointer_tokens"):
            T.use_pointer_tokens(*cfg["pointer_tokens"])
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        t0 = time.perf_counter()
        base, self.proc = load_base(cfg["base"], self.modality)
        self.tokenizer = self.proc.tokenizer
        self.tokenizer.padding_side = "left"
        self.model = PeftModel.from_pretrained(base, os.path.join(self.run_dir, "adapter")).to(self.device).eval()
        self.head_kind = cfg.get("head", "pointer")
        self.layout = cfg.get("layout", "pointer")
        self.head = PointerHead(cfg["hidden"], cfg.get("head_dim", 256)) if self.head_kind == "pointer" else SlotHead(cfg["hidden"])
        self.head.load_state_dict(torch.load(os.path.join(self.run_dir, "head.pt"), map_location="cpu"))
        self.head.to(self.device).eval()
        self.max_options = 255 if self.head_kind == "pointer" else HEAD_SIZE
        self.opt_end_id = self.tokenizer.convert_tokens_to_ids(T.OPT_END)
        self.dec_id = self.tokenizer.convert_tokens_to_ids(T.DECIDE)
        self.name = f"lora:{os.path.basename(self.run_dir.rstrip('/'))}"
        self.load_seconds = time.perf_counter() - t0

    def score(self, prompts: list[Prompt], temperatures: list[float] | None = None, media: list[Any] | None = None) -> list[ScoreResult]:
        import torch

        from s1proto.media import chat_text, encode, head_logits, hidden_states

        if not prompts:
            return []
        if media is None or len(media) != len(prompts):
            raise ValueError(f"{self.name} needs one loaded {self.modality} item per prompt")
        temps = temperatures or [1.0] * len(prompts)
        texts = [chat_text(self.proc, p.text, self.modality) for p in prompts]
        enc = encode(self.proc, texts, media, self.modality)
        enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}
        ids = enc["input_ids"]
        maxn = max(p.n_options for p in prompts)
        opt_pos = torch.zeros((len(prompts), maxn), dtype=torch.long)
        dec_pos = torch.full((len(prompts),), ids.shape[1] - 1, dtype=torch.long)
        for i, p in enumerate(prompts):
            if self.head_kind == "pointer":
                pos = (ids[i] == self.opt_end_id).nonzero(as_tuple=True)[0]
                if len(pos) != p.n_options:
                    raise ValueError(f"found {len(pos)} option delimiters for {p.n_options} options; render with layout='pointer'")
                opt_pos[i, : p.n_options] = pos
                dpos = (ids[i] == self.dec_id).nonzero(as_tuple=True)[0]
                dec_pos[i] = dpos[-1]
        nopts = torch.tensor([p.n_options for p in prompts], device=self.device)
        with torch.inference_mode():
            logits = head_logits(self.head, hidden_states(self.model, enc, self.modality), nopts, opt_pos.to(self.device), dec_pos.to(self.device)).cpu()
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        results = []
        for i, p in enumerate(prompts):
            raw = logits[i][: p.n_options].tolist()
            probs = softmax([x / temps[i] for x in raw])
            results.append(ScoreResult(probabilities=probs, logits=raw, input_tokens=int(lengths[i])))
        return results


def load_scorer(spec: str | None) -> ScorerProtocol:
    """`fake` -> FakeScorer; `lora:<run_dir>` -> LoRAScorer; anything
    else -> HF model id / path."""
    spec = spec or os.environ.get("S1_MODEL", "fake")
    if spec == "fake":
        return FakeScorer()
    if spec.startswith("lora:"):
        import json

        run_dir = spec[len("lora:") :]
        cfg_path = os.path.join(run_dir, "config.json")
        modality = json.load(open(cfg_path)).get("modality", "text") if os.path.exists(cfg_path) else "text"
        if modality in ("vision", "audio"):
            return MultimodalScorer(run_dir=run_dir)
        return LoRAScorer(run_dir=run_dir)
    return HFScorer(model_id=spec)
