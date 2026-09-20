"""Request/response models matching TypeSafe's /v1/systemone contract.

Kept byte-compatible with https://docs.typesafe.ai/api so the official
`typesafe_sdk` client works against us with only `base_url` changed.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]

# TypeSafe caps Choice at 255 options. Phase 1 scores with single-letter
# labels so the effective cap is 26; we still validate to 255 so the
# request schema doesn't diverge, and the scorer rejects > its own cap.
MAX_CHOICE_OPTIONS = 255


class NoulCriteria(BaseModel):
    true: str | None = None
    false: str | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONValue
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONValue
    criteria: dict[str, JSONValue]

    @field_validator("criteria")
    @classmethod
    def _at_least_two(cls, v: dict[str, JSONValue]) -> dict[str, JSONValue]:
        if len(v) < 2:
            raise ValueError("choice needs at least 2 options")
        if len(v) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice supports at most {MAX_CHOICE_OPTIONS} options")
        for k in v:
            if not k or not k.strip():
                raise ValueError("option names must be non-empty")
        return v


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONValue
    criteria: list[JSONValue] = Field(min_length=2)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class Explain(BaseModel):
    """Ask what in the state moved the answer.

    The state is cut into segments and the same question is asked again with
    each one removed; the drop in probability is that segment's contribution.
    It is a counterfactual on the input, so the caller can re-run any line of
    it against this same endpoint and get the same number — no access to the
    weights and no trust in an explanation the model wrote about itself.

    One forward pass per segment, and no text is generated for any of them."""

    method: Literal["ablation"] = "ablation"
    unit: Literal["sentence", "line"] = "sentence"
    max_segments: int = Field(default=12, ge=1, le=40)
    questions: list[str] | None = None  # default: every question in the request


class Segment(BaseModel):
    text: str
    p_without: float
    delta: float  # p(as given) - p(without this segment); positive means it argued for the answer


class Explanation(BaseModel):
    method: Literal["ablation"] = "ablation"
    unit: str
    option: str  # which option the deltas are measured against
    p: float  # that option's probability with the whole state
    segments: list[Segment]


class SystemOneRequest(BaseModel):
    state: JSONValue
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    # Extension over the TypeSafe schema: deterministic gates evaluated
    # server-side from the answers (see circuits.py). Optional; absent
    # in the response unless supplied.
    gates: dict[str, Any] | None = None
    # Extension: per-segment attribution by ablation. Optional; absent in
    # the response unless supplied. Costs one forward pass per segment.
    explain: Explain | None = None

    @model_validator(mode="after")
    def _state_present(self) -> SystemOneRequest:
        if self.state is None:
            raise ValueError("state is required")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, JSONValue]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def confidence_from_probabilities(probs: list[float]) -> float:
    """1 - H(p) / log(N). 1.0 when all mass is on one option, 0.0 when
    uniform. N=1 is undefined; we never emit it (min 2 options)."""
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    c = 1.0 - h / math.log(n)
    # Clamp float noise so pydantic's [0, 1] bound never trips.
    return min(1.0, max(0.0, c))
