"""Coalescing concurrent callers into one forward pass."""

import threading
import time

from s1proto.batching import Batcher
from s1proto.schema import NoulQuestion
from s1proto.scorer import FakeScorer
from s1proto.template import render


class CountingScorer(FakeScorer):
    """A FakeScorer that records how the prompts arrived."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def score(self, prompts, temperatures=None, media=None):
        self.calls.append(len(prompts))
        time.sleep(0.02)  # a forward pass takes time; without it nothing ever overlaps
        return super().score(prompts, temperatures)


def one_prompt(text="hello"):
    return render(text, NoulQuestion(type="noul", instructions="Well?"), layout="letters")


def test_concurrent_callers_share_a_pass():
    inner = CountingScorer()
    b = Batcher(inner, max_batch=8, wait_ms=25)
    out = {}

    def call(i):
        out[i] = b.score([one_prompt(f"message {i}")])

    threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(out) == 8
    assert all(len(v) == 1 for v in out.values())
    assert max(inner.calls) > 1, f"nothing batched: {inner.calls}"
    assert sum(inner.calls) == 8


def test_each_caller_gets_its_own_answers_back():
    """The slice a caller receives must be its own prompts, in order."""
    b = Batcher(CountingScorer(), max_batch=8, wait_ms=20)
    results = {}

    def call(i, n):
        results[i] = b.score([one_prompt(f"{i}-{j}") for j in range(n)])

    threads = [threading.Thread(target=call, args=(i, n)) for i, n in enumerate([1, 3, 2])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [len(results[i]) for i in range(3)] == [1, 3, 2]
    alone = Batcher(CountingScorer(), max_batch=1).score([one_prompt("0-0")])
    assert results[0][0].probabilities == alone[0].probabilities  # batching changes nothing about the answer


def test_a_failing_batch_does_not_wedge_the_queue():
    class Breaks(CountingScorer):
        def score(self, prompts, temperatures=None, media=None):
            if any("boom" in p.text for p in prompts):
                raise RuntimeError("upstream blew up")
            return super().score(prompts, temperatures)

    b = Batcher(Breaks(), max_batch=4, wait_ms=10)
    try:
        b.score([one_prompt("boom")])
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert len(b.score([one_prompt("fine")])) == 1  # thread still alive


def test_disabled_batcher_calls_straight_through():
    inner = CountingScorer()
    b = Batcher(inner, max_batch=1)
    b.score([one_prompt("a"), one_prompt("b")])
    assert inner.calls == [2]
    assert b.max_options == inner.max_options  # attributes pass through
