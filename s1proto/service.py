"""FastAPI service exposing POST /v1/systemone in TypeSafe's format."""

from __future__ import annotations

import os
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
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    confidence_from_probabilities,
)
from s1proto.scorer import ScorerProtocol, load_scorer
from s1proto.template import render

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


def create_app(scorer: ScorerProtocol | None = None, temperatures: dict[str, float] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.scorer is None:
            app.state.scorer = load_scorer(os.environ.get("S1_MODEL"))
        yield

    app = FastAPI(title="s1proto", version=__version__, lifespan=lifespan)
    app.state.scorer = scorer
    app.state.temperatures = temperatures or parse_temperatures(os.environ.get("S1_TEMPERATURES"))

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        s = app.state.scorer
        return {"ok": s is not None, "model": getattr(s, "name", None), "max_options": getattr(s, "max_options", None)}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest, request: Request, authorization: str | None = Header(default=None)) -> Any:
        # TypeSafe requires a bearer key. With S1_API_KEY set the token must
        # match it; without it any non-empty token is accepted (local runs).
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        expected = os.environ.get("S1_API_KEY")
        if expected and authorization[7:].strip() != expected:
            raise HTTPException(status_code=401, detail="invalid api key")
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
        headers = {"x-s1-latency-ms": f"{(time.perf_counter() - t0) * 1000:.1f}"}
        return JSONResponse(content=body, headers=headers)

    return app


app = create_app()
