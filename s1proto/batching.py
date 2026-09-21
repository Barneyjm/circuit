"""Coalesce concurrent requests into one forward pass.

A GPU answering one request at a time is idle between them, and the routing
workload makes that obvious: every request is a few hundred tokens, arrives
alongside others, and takes longer to schedule than to compute. Batching eight
of them into a single pass costs barely more than one and removes seven.

The scorer already batches the prompts *within* a request. This does the same
across requests: a caller hands its prompts to the queue and waits, a single
worker thread drains whatever has arrived, scores it in one call, and hands
each caller back its own slice. Work is ordered by arrival, so nobody is
starved, and a batch leaves as soon as either it is full or the first prompt
in it has waited long enough.

    S1_BATCH_MAX     prompts in one pass (1, the default, disables it entirely)
    S1_BATCH_WAIT_MS how long the first prompt in a batch waits for company
    S1_BATCH_UNIFORM only batch prompts of the same token length (1, the default)

**Batching normally costs you a reproducible answer, and it does not have to.**
A bf16 matmul reduces in a different order at a different batch size, so the
same prompt scored in a batch answers about 0.01 away from the same prompt
scored alone. Two conditions together remove that, both measured on an L40S:

  batch-invariant kernels   thinking-machines-lab/batch_invariant_ops swaps mm,
                            addmm, _log_softmax and mean.dim through
                            torch.Library, for about 5 ms a pass. Enabled here
                            whenever it is installed and CUDA is present.

  uniform token length      attention is not one of the ops they swap, so a pass
                            holding prompts of different lengths still drifts.
                            Batching only prompts of the same length closes it.

With both, a prompt batched with four others of the same length and entirely
different text returns bit-identical numbers to the same prompt scored alone.
With neither, that case moves by 1.56e-02, which is enough to cross a threshold.

**What it costs is batching itself, on mixed traffic.** Eight requests of eight
different lengths bucket into eight passes — determinism bought by never
batching. Measured on an L40S: uniform on, 9 passes for 8 requests and a
bit-identical answer; uniform off, 2 passes, 2.2x faster, and 6.47e-02 of
drift. The win is real where lengths cluster naturally, which is the screening
shape — many similar items, one template — and absent where they do not, which
is a public endpoint taking whatever arrives. S1_BATCH_UNIFORM=0 takes the
throughput and gives up the guarantee.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _no_invariance():
    """A context manager that does nothing, for CPU, MPS, or a missing library."""
    from contextlib import nullcontext

    return nullcontext()


def _batch_invariant_mode():
    """Kernels whose result does not depend on batch size, when they are available.

    CUDA only: the implementations are Triton. Without them a batch still answers
    within about 0.01 of a single prompt, which is fine for throughput work and not
    fine for an answer someone has to reproduce."""
    try:
        import torch
        from batch_invariant_ops import set_batch_invariant_mode
    except ImportError:
        return _no_invariance
    if not torch.cuda.is_available():
        return _no_invariance
    return set_batch_invariant_mode


@dataclass
class _Job:
    prompts: list[Any]
    temperatures: list[float]
    media: list[Any] | None
    done: threading.Event = field(default_factory=threading.Event)
    results: list[Any] | None = None
    error: BaseException | None = None


class Batcher:
    """Runs one scorer on a queue, so concurrent callers share a forward pass."""

    def __init__(self, scorer: Any, max_batch: int | None = None, wait_ms: float | None = None, uniform: bool | None = None) -> None:
        self.scorer = scorer
        self.uniform = (os.environ.get("S1_BATCH_UNIFORM", "1") == "1") if uniform is None else uniform
        self._lengths: dict[str, int] = {}
        self.invariant = _batch_invariant_mode()
        self.max_batch = int(os.environ.get("S1_BATCH_MAX", "1") if max_batch is None else max_batch)
        self.wait_s = float(os.environ.get("S1_BATCH_WAIT_MS", "6") if wait_ms is None else wait_ms) / 1000
        self._queue: list[_Job] = []
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        self.batches = 0
        self.prompts_scored = 0
        if self.max_batch > 1:
            threading.Thread(target=self._run, name="s1-batcher", daemon=True).start()

    # The scorer's own interface, so callers cannot tell the difference.
    def score(self, prompts: list[Any], temperatures: list[float] | None = None, media: list[Any] | None = None) -> list[Any]:
        if self.max_batch <= 1 or not prompts:
            return self._call(prompts, temperatures, media)
        job = _Job(prompts, temperatures or [1.0] * len(prompts), media)
        with self._arrived:
            self._queue.append(job)
            self._arrived.notify()
        job.done.wait()
        if job.error:
            raise job.error
        return job.results or []

    def _call(self, prompts: list[Any], temperatures: list[float] | None, media: list[Any] | None) -> list[Any]:
        with self.invariant():
            if media is not None:
                return self.scorer.score(prompts, temperatures, media=media)
            return self.scorer.score(prompts, temperatures)

    def _length(self, prompt: Any) -> int:
        """Token count when a tokenizer is reachable, character count otherwise.
        Either way it is only a bucket key: prompts that disagree do not share a pass."""
        n = self._lengths.get(prompt.text)
        if n is None:
            tok = getattr(self.scorer, "tokenizer", None)
            n = len(tok.encode(prompt.text)) if tok is not None else len(prompt.text)
            if len(self._lengths) < 4096:  # a bounded cache; prompts rarely repeat
                self._lengths[prompt.text] = n
        return n

    def _key(self, job: _Job) -> Any:
        """Jobs share a pass only when they agree on everything the kernels see."""
        try:
            if job.media is not None:
                return ("media", id(job))
            if not self.uniform:
                return "any"
            return ("len", tuple(sorted({self._length(p) for p in job.prompts})))
        except Exception:  # never let a key failure strand the caller waiting
            return ("alone", id(job))

    def _take(self) -> list[_Job]:
        """The next batch: everything queued, up to the prompt cap, oldest first.
        A job is never split, so one large request goes alone rather than half."""
        with self._arrived:
            while not self._queue:
                self._arrived.wait()
            first = self._queue.pop(0)
            key = self._key(first)
            batch, size = [first], len(first.prompts)
            deadline = time.perf_counter() + self.wait_s
            while size < self.max_batch:
                # Only a job whose prompts are the same length may join: mixing lengths
                # pads, padding changes what attention reduces over, and the answer moves.
                fits = next((j for j in self._queue if self._key(j) == key and len(j.prompts) + size <= self.max_batch), None)
                if fits is not None:
                    self._queue.remove(fits)
                    batch.append(fits)
                    size += len(fits.prompts)
                    continue
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._arrived.wait(remaining):
                    break
            return batch

    def _run(self) -> None:
        while True:
            batch = self._take()
            # Media states are batched only with their own kind: the processor takes one
            # list of images or clips, and mixing a text job in would misalign them.
            text_only = [j for j in batch if j.media is None]
            media_jobs = [j for j in batch if j.media is not None]
            for group in ([text_only] if text_only else []) + [[j] for j in media_jobs]:
                if not group:
                    continue
                prompts = [p for j in group for p in j.prompts]
                temps = [t for j in group for t in j.temperatures]
                media = [m for j in group for m in (j.media or [])] or None
                try:
                    results = self._call(prompts, temps, media)
                    self.batches += 1
                    self.prompts_scored += len(prompts)
                except BaseException as e:  # a bad batch must not take the thread down
                    for j in group:
                        j.error = e
                        j.done.set()
                    continue
                at = 0
                for j in group:
                    j.results = results[at : at + len(j.prompts)]
                    at += len(j.prompts)
                    j.done.set()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "max_batch": self.max_batch,
            "uniform_length": self.uniform,
            "batch_invariant": self.invariant is not _no_invariance,
            "wait_ms": round(self.wait_s * 1000, 1),
            "batches": self.batches,
            "prompts_scored": self.prompts_scored,
            "prompts_per_batch": round(self.prompts_scored / self.batches, 2) if self.batches else 0,
        }

    # Everything else about the scorer (name, layout, max_options, modality) passes through.
    def __getattr__(self, item: str) -> Any:
        return getattr(self.scorer, item)
