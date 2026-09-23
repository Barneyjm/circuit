"""Request/response models matching TypeSafe's /v1/systemone contract.

Kept byte-compatible with https://docs.typesafe.ai/api so the official
`typesafe_sdk` client works against us with only `base_url` changed.
"""

from __future__ import annotations

import math
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]

# TypeSafe caps Choice at 255 options. Phase 1 scores with single-letter
# labels so the effective cap is 26; we still validate to 255 so the
# request schema doesn't diverge, and the scorer rejects > its own cap.
MAX_CHOICE_OPTIONS = 255


def _check_options(v: dict[str, JSONValue], kind: str) -> dict[str, JSONValue]:
    if len(v) < 2:
        raise ValueError(f"{kind} needs at least 2 options")
    if len(v) > MAX_CHOICE_OPTIONS:
        raise ValueError(f"{kind} supports at most {MAX_CHOICE_OPTIONS} options")
    for k in v:
        if not k or not k.strip():
            raise ValueError("option names must be non-empty")
    return v


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
        return _check_options(v, "choice")


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONValue
    criteria: list[JSONValue] = Field(min_length=2)


class MultiQuestion(BaseModel):
    """Every option that applies, each with its own probability. None may apply, or all.
    Extension over the TypeSafe schema (circuit v2 models)."""

    type: Literal["multi"]
    instructions: JSONValue
    criteria: dict[str, JSONValue]

    @field_validator("criteria")
    @classmethod
    def _options(cls, v: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return _check_options(v, "multi")


class LocateQuestion(BaseModel):
    """Which part of the state answers the instructions. Candidates are the state's string
    values (a list of strings gives one candidate per element; a plain-text state is split
    into sentences), plus "none" when the state does not say. `criteria` optionally
    describes what "none" means. Extension over the TypeSafe schema (circuit v2 models)."""

    type: Literal["locate"]
    instructions: JSONValue
    criteria: str | None = None


class RankQuestion(BaseModel):
    """Order the options, best first, by the instructions. Extension over the TypeSafe
    schema (circuit v2.1 models)."""

    type: Literal["rank"]
    instructions: JSONValue
    criteria: dict[str, JSONValue]

    @field_validator("criteria")
    @classmethod
    def _options(cls, v: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return _check_options(v, "rank")


MAX_MATCH_ITEMS = 64


class MatchQuestion(BaseModel):
    """Match each of `items` to one option in `criteria`, or to "none". Items are answered
    independently, so two items may match the same option. `none` optionally describes what
    no match means. Extension over the TypeSafe schema (circuit v2.1 models)."""

    type: Literal["match"]
    instructions: JSONValue
    items: dict[str, JSONValue]
    criteria: dict[str, JSONValue]
    none: str | None = None

    @field_validator("items")
    @classmethod
    def _items(cls, v: dict[str, JSONValue]) -> dict[str, JSONValue]:
        if not v:
            raise ValueError("match needs at least 1 item")
        if len(v) > MAX_MATCH_ITEMS:
            raise ValueError(f"match supports at most {MAX_MATCH_ITEMS} items")
        if any(not k or not k.strip() for k in v):
            raise ValueError("item names must be non-empty")
        return v

    @field_validator("criteria")
    @classmethod
    def _options(cls, v: dict[str, JSONValue]) -> dict[str, JSONValue]:
        if not v:
            raise ValueError("match needs at least 1 option")
        if "none" in v:
            raise ValueError('"none" is reserved for no match; describe it with the `none` field')
        _check_options({**v, "none": None}, "match")  # "none" counts toward the cap
        return v


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion | MultiQuestion | LocateQuestion | RankQuestion | MatchQuestion, Field(discriminator="type")]


QUESTION_MODELS: dict[str, type[BaseModel]] = {
    "noul": NoulQuestion,
    "choice": ChoiceQuestion,
    "score": ScoreQuestion,
    "multi": MultiQuestion,
    "locate": LocateQuestion,
    "rank": RankQuestion,
    "match": MatchQuestion,
}


def parse_question(q: dict[str, Any]) -> BaseModel:
    """A question dict (as stored in a JSONL row) as its validated model."""
    return QUESTION_MODELS[q["type"]].model_validate(q)


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
    # Unknown top-level fields are refused, as TypeSafe's API refuses them (measured:
    # any extra key there answers 400). A misspelled "explains" or "gate" is a
    # request that quietly does nothing otherwise, which is worse than an error.
    # Per-question fields stay permissive, also matching them: a question carrying
    # "weight" is accepted and ignored by both.
    model_config = ConfigDict(extra="forbid")

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


class MultiAnswer(BaseModel):
    type: Literal["multi"] = "multi"
    selected: list[str]  # options at probability 0.5 or above, most likely first
    probabilities: dict[str, float]  # independent per option; they need not sum to 1


class Located(BaseModel):
    path: str  # where in the state, e.g. "paragraphs[3]" or "sentence[2]"
    text: str
    probability: float = Field(ge=0.0, le=1.0)


class LocateAnswer(BaseModel):
    type: Literal["locate"] = "locate"
    located: list[Located]  # most likely first, up to three
    none: float = Field(ge=0.0, le=1.0)  # probability that the state does not answer it
    confidence: float = Field(ge=0.0, le=1.0)


class RankAnswer(BaseModel):
    type: Literal["rank"] = "rank"
    order: list[str]  # best first
    probabilities: dict[str, float]  # of being ranked first; sums to 1
    above: list[float]  # P(order[i] ranks above order[i + 1]), one per adjacent pair


class Matched(BaseModel):
    match: str  # an option, or "none"
    probabilities: dict[str, float]  # over the options and "none"; sums to 1
    confidence: float = Field(ge=0.0, le=1.0)


class MatchAnswer(BaseModel):
    type: Literal["match"] = "match"
    matches: dict[str, Matched]  # one per item


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer | LocateAnswer | RankAnswer | MatchAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    # An id for this answer, returned in the body and as x-request-id. TypeSafe's
    # SDK reads one off every result and error; it is also the handle to quote when
    # an answer has to be accounted for later, since nothing about the request is
    # stored server-side.
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


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
