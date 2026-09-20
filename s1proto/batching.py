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

**Off by default, and the reason is not performance.** A batched matmul in bf16
reduces in a different order than a single-row one, so an answer moves by about
0.01 depending on the batch it was computed in — measured at 0.5050 alone
against 0.4954 batched with four *identical* prompts, no padding involved. With
batching on, that means an answer depends on what other callers happened to send
at the same moment, and a probability sitting on a threshold can land either
side of it. For a shared API that advertises a reproducible number, that trade
is wrong. For a screening run over thousands of items on your own machine, where
throughput is the whole point and no single answer is quoted later, it is right:
set S1_BATCH_MAX=8 and take roughly 3x.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


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

    def __init__(self, scorer: Any, max_batch: int | None = None, wait_ms: float | None = None) -> None:
        self.scorer = scorer
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
        if media is not None:
            return self.scorer.score(prompts, temperatures, media=media)
        return self.scorer.score(prompts, temperatures)

    def _take(self) -> list[_Job]:
        """The next batch: everything queued, up to the prompt cap, oldest first.
        A job is never split, so one large request goes alone rather than half."""
        with self._arrived:
            while not self._queue:
                self._arrived.wait()
            first = self._queue.pop(0)
            batch, size = [first], len(first.prompts)
            deadline = time.perf_counter() + self.wait_s
            while size < self.max_batch:
                if not self._queue:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0 or not self._arrived.wait(remaining):
                        break
                    continue
                if len(self._queue[0].prompts) + size > self.max_batch:
                    break
                nxt = self._queue.pop(0)
                batch.append(nxt)
                size += len(nxt.prompts)
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
            "wait_ms": round(self.wait_s * 1000, 1),
            "batches": self.batches,
            "prompts_scored": self.prompts_scored,
            "prompts_per_batch": round(self.prompts_scored / self.batches, 2) if self.batches else 0,
        }

    # Everything else about the scorer (name, layout, max_options, modality) passes through.
    def __getattr__(self, item: str) -> Any:
        return getattr(self.scorer, item)
