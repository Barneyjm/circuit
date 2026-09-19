"""Latency benchmark for the Phase 1 acceptance criteria.

Shape: one request = a ~512-token state + 4 questions (2 choice, 1
score, 1 noul), scored as one batch. Reports p50/p99 wall time of the
scoring call (tokenize + prefill + softmax) and of the full HTTP path
through the FastAPI app in-process.

    uv run python scripts/bench.py Qwen/Qwen3-0.6B-Base --n 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from fastapi.testclient import TestClient

from s1proto.schema import SystemOneRequest
from s1proto.scorer import HFScorer
from s1proto.service import DEFAULT_TEMPERATURES, build_answers, create_app

FILLER = (
    "Ticket #48213. Customer reports that scheduled payouts to their bank account have not arrived for three "
    "consecutive days. They have verified the account details, confirmed there are no holds on the account, and "
    "state that support chat gave conflicting answers. They run a small business and rely on these payouts for "
    "payroll. Prior tickets: one in March about a declined card, resolved. Account tier: Pro. Region: US-East. "
)


def make_request(scorer: HFScorer, target_tokens: int = 512) -> dict:
    state = FILLER
    while len(scorer.tokenizer.encode(state)) < target_tokens:
        state += FILLER
    ids = scorer.tokenizer.encode(state)[:target_tokens]
    state = scorer.tokenizer.decode(ids)
    return {
        "state": state,
        "model": "bench",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                    "trust": "Fraud, holds, compliance",
                },
            },
            "action": {
                "type": "choice",
                "instructions": "What should the assistant do next?",
                "criteria": {"say": None, "refund": None, "escalate": None, "close": None, "flag": None},
            },
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        },
    }


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    scorer = HFScorer(args.model, prefix_cache=not args.no_prefix_cache, dtype=args.dtype)
    raw = make_request(scorer, args.tokens)
    req = SystemOneRequest.model_validate(raw)

    # warm-up (MPS compiles kernels on first shapes)
    for _ in range(3):
        build_answers(req, scorer, DEFAULT_TEMPERATURES)

    score_ms = []
    for _ in range(args.n):
        t = time.perf_counter()
        _, toks = build_answers(req, scorer, DEFAULT_TEMPERATURES)
        score_ms.append((time.perf_counter() - t) * 1000)

    http_ms = []
    with TestClient(create_app(scorer=scorer)) as c:
        for _ in range(args.n):
            t = time.perf_counter()
            r = c.post("/v1/systemone", json=raw, headers={"Authorization": "Bearer x"})
            http_ms.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200

    out = {
        "model": scorer.name,
        "device": scorer.device,
        "load_seconds": round(scorer.load_seconds, 1),
        "prefix_cache": scorer.prefix_cache,
        "dtype": scorer.dtype,
        "state_tokens_per_question": args.tokens,
        "questions": 4,
        "batch_input_tokens": toks,
        "n": args.n,
        "score_ms": {"p50": round(statistics.median(score_ms), 1), "p99": round(pct(score_ms, 99), 1), "mean": round(statistics.mean(score_ms), 1)},
        "http_ms": {"p50": round(statistics.median(http_ms), 1), "p99": round(pct(http_ms, 99), 1)},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
