import pytest
from decision_circuits.backends import SystemOne
from fastapi.testclient import TestClient

from s1proto.dsl import Circuit, G, Q, argmax, majority, order, verify
from s1proto.scorer import FakeScorer
from s1proto.service import create_app

ANSWERS = {
    "pii": {"type": "noul", "noul": 0.92},
    "business": {"type": "noul", "noul": 0.10},
    "angry": {"type": "noul", "noul": 0.93},
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7, "technical": 0.2, "other": 0.1}, "confidence": 0.55},
    "dept2": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "technical": 0.3, "other": 0.1}, "confidence": 0.4},
    "supported": {"type": "noul", "noul": 0.85},
    "urgency": {"type": "score", "score": 2.88, "legend": {}, "probabilities": {"0": 0.0, "1": 0.02, "2": 0.08, "3": 0.90}, "confidence": 0.73},
}


def test_operators_compile_and_evaluate():
    c = Circuit()
    c.gate("redact", ((Q("pii") >= 0.7) & ~Q("business")) >= 0.6).on_uncertain("escalate")
    c.gate("human", (Q("angry") | Q("urgency:3")) >= 0.6)
    c.gate("route", argmax("dept", min_confidence=0.35))
    c.gate("vote", majority("dept", "dept2"))
    c.gate("checked", verify("dept", Q("supported"), tau=0.8)).on_uncertain("escalate")
    c.gate("tier", order("urgency", [1.0, 2.0, 2.6]))
    c.gate("bill_and_hot", (G("route:billing") & G("tier:3")) >= 0.5)

    compiled = c.compile()
    assert compiled["redact"]["op"] == "and" and compiled["redact"]["tau"] == 0.6 and compiled["redact"]["on_uncertain"] == "escalate"
    # helpers exist for the threshold and the not
    helper_ops = sorted(v["op"] for k, v in compiled.items() if k.startswith("_"))
    assert helper_ops == ["not", "threshold"]

    r = c.evaluate(ANSWERS)
    assert set(r) == {"redact", "human", "route", "vote", "checked", "tier", "bill_and_hot"}  # helpers hidden
    assert r["redact"]["value"] is True and r["redact"]["p"] == pytest.approx(0.92 * 0.90, abs=1e-6)
    assert r["human"]["value"] is True and r["human"]["p"] > 0.99
    assert r["route"]["value"] == "billing"
    assert r["vote"]["value"] == "billing"
    assert r["checked"]["value"] == "billing" and r["checked"]["outcome"] == "decided"
    assert r["tier"]["value"] == 3
    assert r["bill_and_hot"]["value"] is True


def test_chained_and_flattens():
    c = Circuit()
    c.gate("all3", (Q("pii") & Q("angry") & Q("supported")) >= 0.5)
    spec = c.compile()["all3"]
    assert spec["op"] == "and" and spec["inputs"] == ["pii", "angry", "supported"]


def test_bare_reference_thresholds_at_half():
    c = Circuit()
    c.gate("is_pii", Q("pii"))
    assert c.compile()["is_pii"] == {"op": "threshold", "input": "pii", "tau": 0.5, "on_uncertain": "abstain", "band": 0.1}


def test_default_policy_carries_value():
    c = Circuit()
    c.gate("tier", order("urgency", [2.85])).on_uncertain("default", default=2).band(0.1)
    r = c.evaluate(ANSWERS)["tier"]
    assert r["outcome"] == "default" and r["value"] == 2


def test_request_round_trip_through_service():
    c = Circuit()
    c.noul("urgent", "Is this urgent?", true="Needs action today", false="Can wait")
    c.choice("dept", "Which team?", {"billing": None, "technical": None, "other": None})
    c.gate("rush", Q("urgent") >= 0.5)
    c.gate("route", argmax("dept"))
    with TestClient(create_app(scorer=FakeScorer())) as client:
        out = c.run(SystemOne("/v1/systemone", api_key="x", client=client), "Card charged twice, please refund today.")
    assert set(out["answers"]) == {"urgent", "dept"}
    assert set(out["gates"]) == {"rush", "route"}
    assert isinstance(out["gates"]["rush"]["value"], bool)
    assert out["gates"]["route"]["value"] in {"billing", "technical", "other"}


def test_indexing_and_keyword_policy_match_string_forms():
    a = Circuit()
    a.gate("hot", (Q("urgency:3") | G("route:billing")) >= 0.5).on_uncertain("escalate").band(0.05)
    b = Circuit()
    b.gate("hot", (Q("urgency")[3] | G("route")["billing"]).at(0.5), on_uncertain="escalate", band=0.05)
    assert a.compile() == b.compile()
    with pytest.raises(KeyError):
        Q("urgency:3")["x"]


def test_to_mermaid_is_a_method():
    c = Circuit()
    c.noul("pii", "PII?")
    c.gate("redact", Q("pii") >= 0.7)
    assert "redact" in c.to_mermaid(plain=True)
