"""FastAPI service exposing POST /v1/systemone in TypeSafe's format."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from s1proto import __version__
from s1proto.circuits import Gate, evaluate_gates
from s1proto.media import load_media, split_media_state
from s1proto.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    Explanation,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Segment,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    confidence_from_probabilities,
)
from s1proto.scorer import ScorerProtocol, load_scorer
from s1proto.template import Prompt, render

# Per-question-type temperature. 1.0 = raw logits (Phase 1). Phase 2
# fits these on a validation split and writes them to S1_TEMPERATURES
# as "noul=1.3,choice=1.1,score=0.9".
DEFAULT_TEMPERATURES = {"noul": 1.0, "choice": 1.0, "score": 1.0}


def parse_temperatures(spec: str | None) -> dict[str, float]:
    temps = dict(DEFAULT_TEMPERATURES)
    if not spec:
        return temps
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k in temps:
            temps[k] = float(v)
    return temps


def text_state_of(req: SystemOneRequest) -> Any:
    """The text part of a state, whether or not it also carries media."""
    split = split_media_state(req.state)
    return split[0] if split else req.state


def build_answers(req: SystemOneRequest, scorer: ScorerProtocol, temps: dict[str, float]) -> tuple[dict[str, Any], int]:
    ids = list(req.questions.keys())
    split = split_media_state(req.state)  # {"image"|"audio": spec, "text"?: ...} or a plain text/JSON state
    text_state = split[0] if split else req.state
    prompts = [render(text_state, req.questions[i], layout=getattr(scorer, "layout", "letters")) for i in ids]
    temperatures = [temps[req.questions[i].type] for i in ids]
    if split:
        _, modality, spec = split
        if getattr(scorer, "modality", "text") != modality:
            raise HTTPException(status_code=422, detail=f"model {scorer.name} takes {getattr(scorer, 'modality', 'text')} states, not {modality}")
        try:
            item = load_media(modality, spec)  # data URI or URL; the server never reads local paths
        except (ValueError, TypeError, OSError) as e:
            raise HTTPException(status_code=422, detail=f"state.{'image' if modality == 'vision' else 'audio'}: {e}") from e
        results = scorer.score(prompts, temperatures, media=[item] * len(prompts))
    elif getattr(scorer, "modality", "text") != "text":
        raise HTTPException(status_code=422, detail=f"model {scorer.name} needs a state with an {'image' if scorer.modality == 'vision' else 'audio'} field")
    else:
        results = scorer.score(prompts, temperatures)

    answers: dict[str, Any] = {}
    total_tokens = 0
    for qid, q, p, r in zip(ids, [req.questions[i] for i in ids], prompts, results, strict=True):
        total_tokens += r.input_tokens
        probs = r.probabilities
        if isinstance(q, NoulQuestion):
            answers[qid] = NoulAnswer(noul=probs[0])  # option 0 is "yes"
        elif isinstance(q, ChoiceQuestion):
            dist = dict(zip(p.option_keys, probs, strict=True))
            best = max(dist, key=dist.__getitem__)
            answers[qid] = ChoiceAnswer(choice=best, probabilities=dist, confidence=confidence_from_probabilities(probs))
        elif isinstance(q, ScoreQuestion):
            dist = {str(i): pr for i, pr in enumerate(probs)}
            expected = sum(i * pr for i, pr in enumerate(probs))
            legend = {str(i): level for i, level in enumerate(q.criteria)}
            answers[qid] = ScoreAnswer(score=expected, legend=legend, probabilities=dist, confidence=confidence_from_probabilities(probs))
    return answers, total_tokens


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def segment_state(text: str, unit: str, limit: int) -> list[str]:
    parts = text.splitlines() if unit == "line" else _SENTENCE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:limit]


def build_explanations(
    req: SystemOneRequest, scorer: ScorerProtocol, temps: dict[str, float], answers: dict[str, Any], text_state: Any
) -> tuple[dict[str, Any], int]:
    """Ablation attribution: the same question with one segment of the state
    removed at a time. Every variant of every question goes through the scorer
    in one batch, so the cost is forward passes, not round trips."""
    spec = req.explain
    if not isinstance(text_state, str):
        raise HTTPException(status_code=422, detail="explain: only a text state can be segmented")
    qids = spec.questions or list(req.questions.keys())
    unknown = [q for q in qids if q not in req.questions]
    if unknown:
        raise HTTPException(status_code=422, detail=f"explain: no such question {unknown[0]!r}")
    segments = segment_state(text_state, spec.unit, spec.max_segments)
    if len(segments) < 2:
        return {}, 0

    layout = getattr(scorer, "layout", "letters")
    jobs, prompts, temperatures = [], [], []
    for qid in qids:
        q = req.questions[qid]
        for i in range(len(segments)):
            without = " ".join(segments[:i] + segments[i + 1 :])
            jobs.append((qid, i))
            prompts.append(render(without, q, layout=layout))
            temperatures.append(temps[q.type])
    results = scorer.score(prompts, temperatures)

    tokens = sum(r.input_tokens for r in results)
    out: dict[str, Any] = {}
    for qid in qids:
        q = req.questions[qid]
        full = render(text_state, q, layout=layout)
        option, p_full = explained_option(q, full, answers[qid])
        idx = list(full.option_keys).index(option)
        rows = []
        for (job_qid, i), r in zip(jobs, results, strict=True):
            if job_qid != qid:
                continue
            p_without = r.probabilities[idx]
            rows.append(Segment(text=segments[i], p_without=p_without, delta=p_full - p_without))
        out[qid] = Explanation(unit=spec.unit, option=option, p=p_full, segments=rows)
    return {k: v.model_dump() for k, v in out.items()}, tokens


def explained_option(q: Question, prompt: Prompt, answer: Any) -> tuple[str, float]:
    """Which option the deltas are measured against: the one the model picked."""
    if isinstance(q, NoulQuestion):
        p = answer.noul
        return ("yes", p) if p >= 0.5 else ("no", 1.0 - p)
    if isinstance(q, ChoiceQuestion):
        return answer.choice, answer.probabilities[answer.choice]
    best = max(answer.probabilities, key=answer.probabilities.__getitem__)
    return best, answer.probabilities[best]


def create_app(scorer: ScorerProtocol | None = None, temperatures: dict[str, float] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.scorer is None:
            app.state.scorer = load_scorer(os.environ.get("S1_MODEL"))
        yield

    app = FastAPI(title="s1proto", version=__version__, lifespan=lifespan)
    app.state.scorer = scorer
    app.state.temperatures = temperatures or parse_temperatures(os.environ.get("S1_TEMPERATURES"))
    # One device, one forward pass at a time. Two concurrent passes on Apple GPUs
    # do not run twice as fast; they hang, holding both requests forever. So
    # scoring is serialized and the queue is short and explicit:
    #
    #   S1_CONCURRENCY    passes running at once (1, unless the device can take more)
    #   S1_MAX_INFLIGHT   running + waiting before this box declines with a 503
    #   S1_QUEUE_WAIT_S   how long a request waits for its turn before it declines too
    #
    # A 503 with x-s1-busy is not an error, it is this box saying "send it somewhere
    # with more room" — which is exactly what the API gateway in front of it does.
    app.state.max_inflight = int(os.environ.get("S1_MAX_INFLIGHT", "0"))
    app.state.queue_wait_s = float(os.environ.get("S1_QUEUE_WAIT_S", "10"))
    app.state.device_slots = threading.Semaphore(int(os.environ.get("S1_CONCURRENCY", "1")))
    app.state.inflight = 0
    app.state.inflight_lock = threading.Lock()

    def take_slot() -> bool:
        with app.state.inflight_lock:
            if app.state.max_inflight and app.state.inflight >= app.state.max_inflight:
                return False
            app.state.inflight += 1
            return True

    def release_slot() -> None:
        with app.state.inflight_lock:
            app.state.inflight -= 1

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        s = app.state.scorer
        return {
            "ok": s is not None,
            "model": getattr(s, "name", None),
            "max_options": getattr(s, "max_options", None),
            "inflight": app.state.inflight,
            "max_inflight": app.state.max_inflight,
            "concurrency": app.state.device_slots._value,
        }

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest, request: Request, authorization: str | None = Header(default=None)) -> Any:
        # TypeSafe requires a bearer key. With S1_API_KEY set the token must
        # match it; without it any non-empty token is accepted (local runs).
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        expected = os.environ.get("S1_API_KEY")
        if expected and authorization[7:].strip() != expected:
            raise HTTPException(status_code=401, detail="invalid api key")
        if not take_slot():
            raise HTTPException(
                status_code=503,
                detail=f"at capacity: {app.state.max_inflight} questions already in flight",
                headers={"x-s1-busy": "1", "retry-after": "1"},
            )
        try:
            if not app.state.device_slots.acquire(timeout=app.state.queue_wait_s):
                raise HTTPException(
                    status_code=503,
                    detail=f"waited {app.state.queue_wait_s:g}s for the device and it is still busy",
                    headers={"x-s1-busy": "1", "retry-after": "1"},
                )
            try:
                return answer(req)
            finally:
                app.state.device_slots.release()
        finally:
            release_slot()

    def answer(req: SystemOneRequest) -> Any:
        cap = getattr(app.state.scorer, "max_options", 255)
        for qid, q in req.questions.items():
            n = 2 if isinstance(q, NoulQuestion) else len(q.criteria)
            if n > cap:
                raise HTTPException(status_code=422, detail=f"question {qid!r}: {n} options exceeds this model's cap of {cap}")
        t0 = time.perf_counter()
        answers, tokens = build_answers(req, app.state.scorer, app.state.temperatures)
        resp = SystemOneResponse(model=app.state.scorer.name, answers=answers, usage=Usage(input_tokens=tokens, output_tokens=0))
        body = resp.model_dump()
        if req.gates:
            # Gates are evaluated by code from the calibrated answers; the
            # model never sees them. Only present in the response when asked
            # for, so the TypeSafe SDK's response parser is unaffected.
            try:
                gate_results = evaluate_gates({k: Gate.model_validate(v) for k, v in req.gates.items()}, body["answers"])
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"gates: {e}") from e
            body["gates"] = {k: v.model_dump() for k, v in gate_results.items()}
        if req.explain:
            explanations, extra = build_explanations(req, app.state.scorer, app.state.temperatures, answers, text_state_of(req))
            if explanations:
                body["explanations"] = explanations
                body["usage"]["input_tokens"] += extra
        headers = {"x-s1-latency-ms": f"{(time.perf_counter() - t0) * 1000:.1f}", "x-request-id": body["request_id"]}
        return JSONResponse(content=body, headers=headers)

    return app


app = create_app()
