"""The shared-prefix scorer concatenates token ids of `prefix` and
`tail`. That is only equivalent to scoring the full text if the
tokenizer never merges across the boundary. Check with the real
tokenizer on a spread of states (tokenizer only, no model load)."""

import json

import pytest

from s1proto.schema import SystemOneRequest
from s1proto.template import render
from tests.test_schema_and_template import HANDOFF_EXAMPLE

STATES = [
    "Help! My payouts have been failing for 3 days.",
    "trailing whitespace and newlines\n\n\n",
    {"ticket": {"messages": [{"text": "hi"}], "id": 42}},
    ["a", 1, None, {"b": "c"}],
    "unicode — café — 東京 — 🚀",
    "ends with a colon:",
    "ends with a letter A",
]


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.parametrize("state", STATES, ids=[json.dumps(s)[:20] for s in STATES])
def test_prefix_plus_tail_tokenizes_like_full_text(tok, state):
    req = SystemOneRequest.model_validate({**HANDOFF_EXAMPLE, "state": state})
    for q in req.questions.values():
        p = render(req.state, q)
        assert p.text == p.prefix + p.tail
        full = tok.encode(p.text, add_special_tokens=False)
        joined = tok.encode(p.prefix, add_special_tokens=False) + tok.encode(p.tail, add_special_tokens=False)
        assert full == joined


def test_shared_prefix_matches_scoring_each_question_alone():
    """The cached path must agree with the plain one. It agrees exactly — and
    better than the left-padded batch does, which drifts by up to 0.02 depending
    on what else shared the request."""
    pytest.importorskip("torch")
    from pathlib import Path

    from s1proto.scorer import LoRAScorer

    run = Path("runs/circuit-1.7b")
    if not run.exists():
        pytest.skip("no local weights")
    from s1proto.schema import ChoiceQuestion, NoulQuestion

    sc = LoRAScorer(run_dir=str(run))
    state = "Eligibility: randomised trials in adults with COPD. " + "Background: a multicentre trial of 1,204 adults over 52 weeks. " * 4
    qs = [
        NoulQuestion(type="noul", instructions="Is the population adults with COPD?"),
        NoulQuestion(type="noul", instructions="Does it report exacerbation rates?"),
        ChoiceQuestion(type="choice", instructions="What design is this?", criteria={"trial": None, "cohort": None, "other": None}),
    ]
    prompts = [render(state, q, layout="pointer") for q in qs]

    sc.prefix_cache = False
    alone = [sc.score([p])[0].probabilities for p in prompts]
    sc.prefix_cache = True
    cached = [r.probabilities for r in sc.score(prompts)]

    for a, c in zip(alone, cached, strict=True):
        for x, y in zip(a, c, strict=True):
            assert abs(x - y) < 1e-6, (a, c)
