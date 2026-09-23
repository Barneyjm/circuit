"""FastAPI service exposing POST /v1/systemone in TypeSafe's format."""

from __future__ import annotations

import itertools
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from s1proto import __version__, telemetry
from s1proto.batching import Batcher
from s1proto.circuits import Gate, evaluate_gates
from s1proto.media import load_media, split_media_state
from s1proto.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    Explanation,
    LocateAnswer,
    Located,
    LocateQuestion,
    MatchAnswer,
    Matched,
    MatchQuestion,
    MultiAnswer,
    MultiQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    RankAnswer,
    RankQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Segment,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    confidence_from_probabilities,
)
from s1proto.scorer import ScorerProtocol, load_scorer
from s1proto.template import V2_KINDS, Prompt, locate_candidates, render

# Per-question-type temperature. 1.0 = raw logits (Phase 1). Phase 2
# fits these on a validation split and writes them to S1_TEMPERATURES
# as "noul=1.3,choice=1.1,score=0.9".
DEFAULT_TEMPERATURES = {"noul": 1.0, "choice": 1.0, "score": 1.0, "multi": 1.0, "locate": 1.0, "rank": 1.0, "match": 1.0}


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
    try:
        prompts = [render(text_state, req.questions[i], layout=getattr(scorer, "layout", "letters")) for i in ids]
    except ValueError as e:  # a locate state with no text, or too many candidates
        raise HTTPException(status_code=422, detail=str(e)) from e
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
        elif isinstance(q, MultiQuestion):
            dist = dict(zip(p.option_keys, probs, strict=True))
            selected = sorted((k for k, v in dist.items() if v >= 0.5), key=lambda k: -dist[k])
            answers[qid] = MultiAnswer(selected=selected, probabilities=dist)
        elif isinstance(q, LocateQuestion):
            texts = dict(locate_candidates(text_state))
            ranked = sorted(zip(p.option_keys[:-1], probs[:-1], strict=True), key=lambda kv: -kv[1])[:3]
            located = [Located(path=k, text=texts[k], probability=v) for k, v in ranked]
            answers[qid] = LocateAnswer(located=located, none=probs[-1], confidence=confidence_from_probabilities(probs))
        elif isinstance(q, RankQuestion):
            order = sorted(range(len(probs)), key=lambda j: -probs[j])
            # Plackett-Luce: of two options, a comes first with probability p_a / (p_a + p_b)
            above = [probs[a] / max(probs[a] + probs[b], 1e-12) for a, b in itertools.pairwise(order)]
            answers[qid] = RankAnswer(order=[p.option_keys[j] for j in order], probabilities=dict(zip(p.option_keys, probs, strict=True)), above=above)
        elif isinstance(q, MatchQuestion):
            n = p.n_options
            matches = {}
            for row, item in enumerate(p.item_keys):
                dist = dict(zip(p.option_keys, probs[row * n : (row + 1) * n], strict=True))
                best = max(dist, key=dist.__getitem__)
                matches[item] = Matched(match=best, probabilities=dist, confidence=confidence_from_probabilities(list(dist.values())))
            answers[qid] = MatchAnswer(matches=matches)
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
    qids = spec.questions or [q for q in req.questions if req.questions[q].type not in ("locate", "match")]
    unknown = [q for q in qids if q not in req.questions]
    if unknown:
        raise HTTPException(status_code=422, detail=f"explain: no such question {unknown[0]!r}")
    if any(req.questions[q].type == "locate" for q in qids):
        # removing a segment removes one of locate's candidates, so there is no fixed answer to compare
        raise HTTPException(status_code=422, detail="explain: locate questions already point at the text; explain the others")
    if any(req.questions[q].type == "match" for q in qids):
        raise HTTPException(status_code=422, detail="explain: match questions answer per item; explain the others")
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
        telemetry.setup()
        if app.state.scorer is None:
            with telemetry.span("s1.load_model", **{"s1.model": os.environ.get("S1_MODEL")}) as sp:
                t0 = time.perf_counter()
                app.state.scorer = load_scorer(os.environ.get("S1_MODEL"))
                sp.set_attribute("s1.load_seconds", round(time.perf_counter() - t0, 2))
        # Opt-in: batching trades a reproducible answer for throughput, because a
        # bf16 batched pass reduces differently than a single-row one (~0.01 on a
        # probability). See s1proto/batching.py.
        if int(os.environ.get("S1_BATCH_MAX", "1")) > 1:
            app.state.scorer = Batcher(app.state.scorer)
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
    app.state.max_inflight = int(os.environ.get("S1_MAX_INFLIGHT", "0"))  # running + waiting; raise it when batching so batches can fill
    app.state.queue_wait_s = float(os.environ.get("S1_QUEUE_WAIT_S", "10"))
    # With batching on, the batcher owns the device and callers must be allowed to
    # reach it concurrently — a semaphore of 1 in front would mean every batch holds
    # exactly one request. Without it, one pass at a time is the safe default.
    _batch_max = int(os.environ.get("S1_BATCH_MAX", "1"))
    app.state.device_slots = threading.Semaphore(int(os.environ.get("S1_CONCURRENCY", str(max(1, _batch_max)) if _batch_max > 1 else "1")))
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
            "batching": getattr(app.state.scorer, "stats", None),
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
        split = split_media_state(req.state)
        attrs = {
            "gen_ai.operation.name": "systemone",
            "gen_ai.request.model": req.model,
            "gen_ai.response.model": getattr(app.state.scorer, "name", None),
            "s1.questions": list(req.questions),
            "s1.question_types": [q.type for q in req.questions.values()],
            "s1.gates": len(req.gates or {}),
            **telemetry.state_attrs(req.state, split),
        }
        with telemetry.span("s1.systemone", headers=request.headers, **attrs) as sp:
            resp = queued(req, sp)
            sp.set_attribute("gen_ai.response.id", resp.headers.get("x-request-id", ""))
            sp.set_attribute("gen_ai.usage.input_tokens", int(resp.headers.get("x-s1-input-tokens", "0")))
            return resp

    def queued(req: SystemOneRequest, sp: Any) -> Any:
        if not take_slot():
            sp.set_attribute("s1.busy", True)
            raise HTTPException(
                status_code=503,
                detail=f"at capacity: {app.state.max_inflight} questions already in flight",
                headers={"x-s1-busy": "1", "retry-after": "1"},
            )
        try:
            with telemetry.span("s1.queue"):
                got = app.state.device_slots.acquire(timeout=app.state.queue_wait_s)
            if not got:
                sp.set_attribute("s1.busy", True)
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
        supported = getattr(app.state.scorer, "question_types", ("noul", "choice", "score"))
        media = split_media_state(req.state) is not None
        for qid, q in req.questions.items():
            if media and q.type in V2_KINDS:  # they point into, order or pair text; images and audio come later if at all
                raise HTTPException(status_code=422, detail=f"question {qid!r}: {q.type} questions take text states only")
            if q.type not in supported:
                raise HTTPException(status_code=422, detail=f"question {qid!r}: model {app.state.scorer.name} does not answer {q.type} questions")
            if isinstance(q, LocateQuestion):
                continue  # candidates come from the state; render_locate enforces its own cap
            n = 2 if isinstance(q, NoulQuestion) else len(q.criteria) + isinstance(q, MatchQuestion)
            if n > cap:
                raise HTTPException(status_code=422, detail=f"question {qid!r}: {n} options exceeds this model's cap of {cap}")
        t0 = time.perf_counter()
        sc = app.state.scorer
        with telemetry.span(
            "s1.score",
            **{
                "s1.prompts": len(req.questions),
                "s1.prefix_cache": getattr(sc, "prefix_cache", None),
                "s1.parallel_options": getattr(sc, "parallel_options", None),
            },
        ) as sp:
            answers, tokens = build_answers(req, sc, app.state.temperatures)
            sp.set_attribute("gen_ai.usage.input_tokens", tokens)
        resp = SystemOneResponse(model=app.state.scorer.name, answers=answers, usage=Usage(input_tokens=tokens, output_tokens=0))
        body = resp.model_dump()
        if req.gates:
            # Gates are evaluated by code from the calibrated answers; the
            # model never sees them. Only present in the response when asked
            # for, so the TypeSafe SDK's response parser is unaffected.
            try:
                with telemetry.span("s1.gates", **{"s1.gates": list(req.gates)}):
                    gate_results = evaluate_gates({k: Gate.model_validate(v) for k, v in req.gates.items()}, body["answers"])
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"gates: {e}") from e
            body["gates"] = {k: v.model_dump() for k, v in gate_results.items()}
        if req.explain:
            with telemetry.span("s1.explain"):
                explanations, extra = build_explanations(req, app.state.scorer, app.state.temperatures, answers, text_state_of(req))
            if explanations:
                body["explanations"] = explanations
                body["usage"]["input_tokens"] += extra
        headers = {
            "x-s1-latency-ms": f"{(time.perf_counter() - t0) * 1000:.1f}",
            "x-request-id": body["request_id"],
            "x-s1-input-tokens": str(body["usage"]["input_tokens"]),
        }
        return JSONResponse(content=body, headers=headers)

    return app


app = create_app()
