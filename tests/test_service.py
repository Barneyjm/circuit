import math

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from s1proto.schema import SystemOneResponse
from s1proto.scorer import FakeScorer
from s1proto.service import create_app, parse_temperatures
from tests.test_schema_and_template import HANDOFF_EXAMPLE

AUTH = {"Authorization": "Bearer test"}


@pytest.fixture(scope="module")
def client():
    app = create_app(scorer=FakeScorer())
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] and r.json()["model"] == "fake"


def test_requires_bearer(client):
    assert client.post("/v1/systemone", json=HANDOFF_EXAMPLE).status_code == 401


def test_handoff_example_response_shape(client):
    r = client.post("/v1/systemone", json=HANDOFF_EXAMPLE, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    SystemOneResponse.model_validate(body)  # schema-valid
    assert body["model"] == "fake"
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    assert set(body["answers"]) == {"department", "frustration", "urgent"}

    dept = body["answers"]["department"]
    assert dept["type"] == "choice"
    assert set(dept["probabilities"]) == {"billing", "technical", "sales"}
    assert math.isclose(sum(dept["probabilities"].values()), 1.0, abs_tol=1e-6)
    assert dept["choice"] == max(dept["probabilities"], key=dept["probabilities"].__getitem__)
    assert 0.0 <= dept["confidence"] <= 1.0

    fr = body["answers"]["frustration"]
    assert fr["type"] == "score"
    assert fr["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(fr["probabilities"]) == {"0", "1", "2"}
    expected = sum(int(k) * v for k, v in fr["probabilities"].items())
    assert math.isclose(fr["score"], expected, abs_tol=1e-9)
    assert 0.0 <= fr["score"] <= 2.0

    ur = body["answers"]["urgent"]
    assert ur["type"] == "noul" and 0.0 <= ur["noul"] <= 1.0
    assert "confidence" not in ur
    assert "x-s1-latency-ms" in r.headers


def test_answers_are_independent(client):
    """Adding a question must not change another's answer."""
    one = {**HANDOFF_EXAMPLE, "questions": {"department": HANDOFF_EXAMPLE["questions"]["department"]}}
    a = client.post("/v1/systemone", json=one, headers=AUTH).json()["answers"]["department"]
    b = client.post("/v1/systemone", json=HANDOFF_EXAMPLE, headers=AUTH).json()["answers"]["department"]
    assert a == b


def test_too_many_options_is_422():
    from s1proto.scorer import FakeScorer

    capped = FakeScorer()
    capped.max_options = 26
    with TestClient(create_app(scorer=capped)) as c:
        req = {"state": "s", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(30)}}}}
        r = c.post("/v1/systemone", json=req, headers=AUTH)
        assert r.status_code == 422 and "cap" in r.text


def test_parse_temperatures():
    assert parse_temperatures(None) == {"noul": 1.0, "choice": 1.0, "score": 1.0, "multi": 1.0, "locate": 1.0, "rank": 1.0, "match": 1.0}
    assert parse_temperatures("noul=1.5, choice=0.8, bogus=3")["noul"] == 1.5
    assert parse_temperatures("noul=1.5, choice=0.8")["choice"] == 0.8


# --- fuzz: any schema-valid request yields a schema-valid response ----------

_text = st.text(min_size=1, max_size=60).filter(lambda s: s.strip())
_json_leaf = st.one_of(st.none(), st.booleans(), st.integers(-1000, 1000), st.floats(allow_nan=False, allow_infinity=False, width=32), _text)
_json = st.recursive(_json_leaf, lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(_text, c, max_size=4)), max_leaves=12)
_opt_names = st.lists(_text, min_size=2, max_size=26, unique=True)

_noul = st.fixed_dictionaries(
    {"type": st.just("noul"), "instructions": _json},
    optional={"criteria": st.fixed_dictionaries({}, optional={"true": _text, "false": _text})},
)
_choice = st.builds(
    lambda names, descs, instr: {"type": "choice", "instructions": instr, "criteria": dict(zip(names, descs))},
    _opt_names,
    st.lists(st.one_of(st.none(), _text), min_size=26, max_size=26),
    _json,
)
_score = st.fixed_dictionaries({"type": st.just("score"), "instructions": _json, "criteria": st.lists(_json, min_size=2, max_size=26)})
_request = st.fixed_dictionaries(
    {
        "state": st.one_of(_text, _json.filter(lambda v: v is not None)),
        "model": _text,
        "questions": st.dictionaries(_text, st.one_of(_noul, _choice, _score), min_size=1, max_size=6),
    }
)


@settings(max_examples=int(__import__("os").environ.get("S1_FUZZ_EXAMPLES", "500")), deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_request)
def test_fuzz_schema_valid_requests_yield_schema_valid_responses(req):
    app = create_app(scorer=FakeScorer())
    with TestClient(app) as c:
        r = c.post("/v1/systemone", json=req, headers=AUTH)
    assert r.status_code == 200, r.text
    body = SystemOneResponse.model_validate(r.json())
    assert set(body.answers) == set(req["questions"])
    for qid, ans in body.answers.items():
        q = req["questions"][qid]
        if q["type"] == "choice":
            assert set(ans.probabilities) == set(q["criteria"])
            assert math.isclose(sum(ans.probabilities.values()), 1.0, abs_tol=1e-6)
        elif q["type"] == "score":
            assert len(ans.probabilities) == len(q["criteria"])
            assert 0.0 <= ans.score <= len(q["criteria"]) - 1


def test_explain_returns_one_segment_per_sentence():
    """Ablation attribution: each sentence removed once, deltas against the
    answer the model actually gave."""
    client = TestClient(create_app(FakeScorer()))
    state = "The pipe burst on Elm. Water is in the street. The meter reads zero."
    r = client.post(
        "/v1/systemone",
        headers={"authorization": "Bearer x"},
        json={
            "model": "fake",
            "state": state,
            "questions": {"urgent": {"type": "noul", "instructions": "Does this need someone today?"}},
            "explain": {"method": "ablation", "unit": "sentence"},
        },
    )
    assert r.status_code == 200, r.text
    ex = r.json()["explanations"]["urgent"]
    assert ex["option"] in ("yes", "no")
    assert [s["text"] for s in ex["segments"]] == [
        "The pipe burst on Elm.",
        "Water is in the street.",
        "The meter reads zero.",
    ]
    for seg in ex["segments"]:
        assert abs(seg["delta"] - (ex["p"] - seg["p_without"])) < 1e-9


def test_explain_is_absent_unless_asked_for():
    client = TestClient(create_app(FakeScorer()))
    r = client.post(
        "/v1/systemone",
        headers={"authorization": "Bearer x"},
        json={
            "model": "fake",
            "state": "One sentence. And another.",
            "questions": {"q": {"type": "noul", "instructions": "Well?"}},
        },
    )
    assert r.status_code == 200
    assert "explanations" not in r.json()


def test_explain_needs_a_text_state():
    client = TestClient(create_app(FakeScorer()))
    r = client.post(
        "/v1/systemone",
        headers={"authorization": "Bearer x"},
        json={
            "model": "fake",
            "state": {"rows": [1, 2]},
            "questions": {"q": {"type": "noul", "instructions": "Well?"}},
            "explain": {"method": "ablation"},
        },
    )
    assert r.status_code == 422
    assert "segmented" in r.json()["detail"]


def test_every_answer_carries_a_request_id():
    """In the body and on the header, so an answer can be quoted later — nothing
    about the request itself is kept server-side."""
    client = TestClient(create_app(FakeScorer()))
    body = {"model": "fake", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Well?"}}}
    r1 = client.post("/v1/systemone", headers={"authorization": "Bearer x"}, json=body)
    r2 = client.post("/v1/systemone", headers={"authorization": "Bearer x"}, json=body)
    assert r1.status_code == 200
    assert r1.json()["request_id"] == r1.headers["x-request-id"]
    assert r1.json()["request_id"] != r2.json()["request_id"]


def test_an_unknown_top_level_field_is_refused():
    """A misspelled extension is a request that silently does nothing; say so."""
    client = TestClient(create_app(FakeScorer()))
    r = client.post(
        "/v1/systemone",
        headers={"authorization": "Bearer x"},
        json={
            "model": "fake",
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "Well?"}},
            "explains": {"method": "ablation"},  # the real field is "explain"
        },
    )
    assert r.status_code == 422
    assert "explains" in r.text


def test_an_unknown_per_question_field_is_ignored():
    """TypeSafe accepts and ignores these (measured), so a client written against
    their raw-dict form keeps working here."""
    client = TestClient(create_app(FakeScorer()))
    r = client.post(
        "/v1/systemone",
        headers={"authorization": "Bearer x"},
        json={"model": "fake", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Well?", "weight": 2}}},
    )
    assert r.status_code == 200
