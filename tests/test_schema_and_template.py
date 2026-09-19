import json
import math

import pytest
from pydantic import ValidationError

from s1proto.schema import SystemOneRequest, confidence_from_probabilities
from s1proto.template import LABELS, render

HANDOFF_EXAMPLE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "s1-proto",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    },
}


def test_request_parses_handoff_example():
    req = SystemOneRequest.model_validate(HANDOFF_EXAMPLE)
    assert set(req.questions) == {"department", "frustration", "urgent"}


@pytest.mark.parametrize(
    "bad",
    [
        {"state": "x", "model": "m", "questions": {}},
        {"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"only": None}}}},
        {"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["one"]}}},
        {"state": "x", "model": "m", "questions": {"q": {"type": "banana", "instructions": "i"}}},
        {"model": "m", "questions": {"q": {"type": "noul", "instructions": "i"}}},
    ],
)
def test_request_rejects_malformed(bad):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(bad)


def test_confidence_bounds():
    assert confidence_from_probabilities([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence_from_probabilities([1 / 3] * 3) == pytest.approx(0.0, abs=1e-9)
    mid = confidence_from_probabilities([0.85, 0.08, 0.07])
    assert 0.0 < mid < 1.0


def test_choice_prompt_layout():
    req = SystemOneRequest.model_validate(HANDOFF_EXAMPLE)
    p = render(req.state, req.questions["department"])
    lines = p.text.splitlines()
    assert lines[0] == "Help! My payouts have been failing for 3 days."
    assert lines[1] == ""
    assert lines[-1] == "Answer:"
    assert "A. billing — Payments, invoicing, refunds" in lines
    assert "C. sales — Pricing, upgrades, new accounts" in lines
    assert p.n_options == 3 and p.option_keys == ("billing", "technical", "sales")


def test_score_prompt_uses_levels_in_order():
    req = SystemOneRequest.model_validate(HANDOFF_EXAMPLE)
    p = render(req.state, req.questions["frustration"])
    assert "A. Calm\nB. Frustrated\nC. Very angry" in p.text
    assert p.option_keys == ("0", "1", "2")


def test_noul_prompt_is_yes_then_no():
    req = SystemOneRequest.model_validate(HANDOFF_EXAMPLE)
    p = render(req.state, req.questions["urgent"])
    assert "A. yes\nB. no" in p.text
    assert p.option_keys == ("yes", "no")


def test_json_state_is_rendered_pretty():
    state = {"ticket": {"messages": [{"text": "hi"}]}}
    q = SystemOneRequest.model_validate({"state": state, "model": "m", "questions": {"q": {"type": "noul", "instructions": "i"}}})
    p = render(q.state, q.questions["q"])
    assert p.text.startswith(json.dumps(state, ensure_ascii=False, indent=2))


def test_structured_criteria_render_as_json():
    crit = {"a": {"covers": "x", "excludes": "y"}, "b": None}
    q = SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {"q": {"type": "choice", "instructions": ["step 1", "step 2"], "criteria": crit}}})
    p = render(q.state, q.questions["q"])
    assert 'A. a — {"covers": "x", "excludes": "y"}' in p.text
    assert "B. b\n" in p.text
    assert '["step 1", "step 2"]' in p.text


def test_option_labels_extend_past_26_for_head_scorers():
    crit = {f"opt{i}": None for i in range(30)}
    q = SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": crit}}})
    p = render(q.state, q.questions["q"])
    assert "AA. opt26" in p.text and "AD. opt29" in p.text
    assert len(LABELS) == 255 and math.isfinite(1.0)
    too_many = {f"opt{i}": None for i in range(256)}
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": too_many}}})
