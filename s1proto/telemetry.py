"""OpenTelemetry for the server: a span per request that joins the caller's trace.

A `decision-circuits` client sends `traceparent`; the request span here continues that trace,
so one trace shows the circuit, the HTTP call, and inside it where the time went:

    s1.systemone           the request (gen_ai.request.model, questions, state kind and size)
    ├── s1.queue           waiting for the device (one forward pass at a time)
    ├── s1.score           rendering and the forward pass (path: shared prefix or independent)
    ├── s1.gates           gate evaluation, when the request carries gates
    └── s1.explain         the ablation passes, when asked for
    s1.load_model          at startup: loading the base and adapter (a cold start's cost)

Off unless OpenTelemetry is installed (`uv sync --extra otel`). Spans are exported when the
standard `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, with
`OTEL_SERVICE_NAME` (default `s1proto`) and headers from `OTEL_EXPORTER_OTLP_HEADERS`; with
neither, spans go to whatever tracer provider the host process set, or nowhere. Never
recorded: the state, the instructions, the options' text. Recorded: model names, question
ids and types, the state's kind and length, token counts, timings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import propagate, trace
except ImportError:
    trace = propagate = None  # type: ignore[assignment]

_configured = False


class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...


def setup() -> None:
    """Install an OTLP exporter when the standard environment asks for one. Idempotent."""
    global _configured
    if _configured or trace is None:
        return
    _configured = True
    if not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")):
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "s1proto")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def _clean(attrs: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, str | bool | int | float) or (isinstance(v, list | tuple) and all(isinstance(x, str) for x in v)) else str(v)
    return out


@contextmanager
def span(name: str, /, headers: Mapping[str, str] | None = None, **attrs: Any) -> Iterator[Any]:
    """A span under the current one; with `headers`, a server span continuing the caller's trace."""
    if trace is None:
        yield _NoSpan()
        return
    from s1proto import __version__

    tracer = trace.get_tracer("s1proto", __version__)
    if headers is not None:
        ctx = propagate.extract(dict(headers))
        with tracer.start_as_current_span(name, context=ctx, kind=trace.SpanKind.SERVER, attributes=_clean(attrs)) as s:
            yield s
        return
    with tracer.start_as_current_span(name, attributes=_clean(attrs)) as s:
        yield s


def state_attrs(state: Any, split: Any) -> dict[str, Any]:
    """What kind of state a request carried and how big, never what it says."""
    if split:
        _, modality, _ = split
        return {"s1.state.kind": modality}
    if isinstance(state, str):
        return {"s1.state.kind": "text", "s1.state.chars": len(state)}
    return {"s1.state.kind": "json", "s1.state.fields": len(state) if isinstance(state, dict) else None}
