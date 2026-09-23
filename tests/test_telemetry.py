"""The server continues the caller's trace, with the queue, the forward pass and the gates as
children, and records nothing of the state."""

import json

from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from s1proto.scorer import FakeScorer
from s1proto.service import create_app

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


def test_request_joins_the_callers_trace_with_queue_score_and_gates():
    EXPORTER.clear()
    body = {
        "state": "My card 4532 0151 1283 0366 was charged twice.",
        "model": "fake",
        "questions": {"dup": {"type": "noul", "instructions": "Charged twice?"}},
        "gates": {"refund": {"op": "threshold", "input": "dup", "tau": 0.5}},
    }
    headers = {"authorization": "Bearer t", "traceparent": f"00-{TRACE_ID}-00f067aa0ba902b7-01"}
    with TestClient(create_app(scorer=FakeScorer())) as c:
        r = c.post("/v1/systemone", json=body, headers=headers)
    assert r.status_code == 200
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    root = spans["s1.systemone"]
    assert format(root.context.trace_id, "032x") == TRACE_ID and root.kind == trace.SpanKind.SERVER
    for child in ("s1.queue", "s1.score", "s1.gates"):
        assert spans[child].parent.span_id == root.context.span_id, child
    a = root.attributes
    assert a["gen_ai.request.model"] == "fake" and a["s1.state.kind"] == "text" and a["gen_ai.response.id"] == r.json()["request_id"]
    assert a["gen_ai.usage.input_tokens"] > 0
    recorded = json.dumps([dict(s.attributes) for s in spans.values()], default=str)
    assert "4532" not in recorded and "Charged twice" not in recorded
