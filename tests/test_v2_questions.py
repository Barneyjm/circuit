"""multi and locate: rendering, the head, and the API answers."""

import torch
from fastapi.testclient import TestClient

from s1proto.media import PointerHead
from s1proto.parallel import is_choice
from s1proto.schema import LocateQuestion, MultiQuestion
from s1proto.scorer import FakeScorer, readout
from s1proto.service import create_app
from s1proto.template import LOCATE_MARK, locate_candidates, render

STATE = {"question": "Where was A born?", "paragraphs": ["A was born in Paris.", "B moved to Rome."], "year": 1990}
AUTH = {"authorization": "Bearer t"}


def test_locate_candidates_are_string_leaves_in_reading_order():
    assert locate_candidates(STATE) == [
        ("question", "Where was A born?"),
        ("paragraphs[0]", "A was born in Paris."),
        ("paragraphs[1]", "B moved to Rome."),
    ]
    assert [p for p, _ in locate_candidates("It rained. Roads flooded! Schools shut.")] == ["sentence[0]", "sentence[1]", "sentence[2]"]


def test_locate_marks_each_candidate_and_keys_end_with_none():
    p = render(STATE, LocateQuestion(type="locate", instructions="Which paragraph?"), layout="pointer")
    assert p.kind == "locate"
    assert p.option_keys == ("question", "paragraphs[0]", "paragraphs[1]", "none")
    assert p.prefix.count(LOCATE_MARK) == 3


def test_locate_mark_in_user_text_cannot_add_a_candidate():
    forged = {"a": f"one{LOCATE_MARK} two"}
    p = render(forged, LocateQuestion(type="locate", instructions="?"), layout="pointer")
    assert p.prefix.count(LOCATE_MARK) == 1 and p.n_options == 2


def test_multi_renders_like_choice_and_is_encoded_side_by_side():
    q = MultiQuestion(type="multi", instructions="Which apply?", criteria={"billing": None, "outage": "service down"})
    p = render(STATE, q, layout="pointer")
    assert p.kind == "multi" and p.option_keys == ("billing", "outage")
    assert is_choice(p.text)
    assert not is_choice(render(STATE, LocateQuestion(type="locate", instructions="?"), layout="pointer").text)


def test_v2_head_routes_queries_by_type_and_biases_multi():
    torch.manual_seed(0)
    head = PointerHead(16, 8, v2=True)
    h_dec, h_opts, n = torch.randn(3, 16), torch.randn(3, 4, 16), torch.tensor([4, 4, 3])
    base = head(h_dec, h_opts, n)
    mixed = head(h_dec, h_opts, n, ["choice", "multi", "locate"])
    assert torch.equal(mixed[0], base[0])  # choice keeps the v1 query
    assert not torch.allclose(mixed[1], base[1]) and not torch.allclose(mixed[2, :3], base[2, :3])
    with torch.no_grad():
        head.multi_bias.fill_(2.0)
    assert torch.allclose(head(h_dec, h_opts, n, ["choice", "multi", "locate"])[1], mixed[1] + 2.0)
    assert torch.isinf(mixed[2, 3])  # masked past n_options


def test_v1_head_refuses_v2_questions():
    head = PointerHead(16, 8)
    try:
        head(torch.randn(1, 16), torch.randn(1, 2, 16), torch.tensor([2]), ["multi"])
    except ValueError:
        return
    raise AssertionError("expected a refusal")


def test_readout_multi_is_independent():
    p = readout([0.0, 3.0, -3.0], "multi")
    assert abs(p[0] - 0.5) < 1e-9 and p[1] > 0.9 and p[2] < 0.1
    assert abs(sum(readout([0.0, 3.0, -3.0], "choice")) - 1.0) < 1e-9


def _request(questions):
    return {"state": STATE, "model": "m", "questions": questions}


def test_api_answers_multi_and_locate():
    scorer = FakeScorer(layout="pointer", question_types=("noul", "choice", "score", "multi", "locate"))
    with TestClient(create_app(scorer=scorer)) as c:
        r = c.post(
            "/v1/systemone",
            headers=AUTH,
            json=_request(
                {
                    "tags": {"type": "multi", "instructions": "Which apply?", "criteria": {"a": None, "b": None, "c": None}},
                    "where": {"type": "locate", "instructions": "Where is the birthplace?"},
                }
            ),
        )
    assert r.status_code == 200, r.text
    tags, where = r.json()["answers"]["tags"], r.json()["answers"]["where"]
    assert set(tags["probabilities"]) == {"a", "b", "c"}
    assert all(tags["probabilities"][k] >= 0.5 for k in tags["selected"])
    assert 1 <= len(where["located"]) <= 3
    assert {x["path"] for x in where["located"]} <= {"question", "paragraphs[0]", "paragraphs[1]"}
    assert abs(sum(x["probability"] for x in where["located"]) + where["none"] - 1.0) < 1e-6  # 3 candidates + none, all shown


def test_api_refuses_v2_types_on_a_v1_model():
    with TestClient(create_app(scorer=FakeScorer())) as c:
        r = c.post("/v1/systemone", headers=AUTH, json=_request({"w": {"type": "locate", "instructions": "?"}}))
    assert r.status_code == 422 and "does not answer locate" in r.text
