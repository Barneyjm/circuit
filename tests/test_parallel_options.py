"""Side-by-side option encoding makes the answer independent of option order, exactly."""

import itertools

import pytest
import torch

from s1proto.parallel import is_choice, parallel_inputs

transformers = pytest.importorskip("transformers")

START, END, DECIDE, PAD = 5, 6, 7, 0


@pytest.fixture(scope="module")
def model():
    cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return transformers.Qwen3Model(cfg).eval()


def sequence(prefix, options):
    ids = list(prefix)
    for o in options:
        ids += [START, *o, END, 9]  # 9 stands in for the newline between options
    return ids + [DECIDE]


def states(model, prefix, options, parallel, pad_to=None):
    ids = sequence(prefix, options)
    pad = (pad_to or len(ids)) - len(ids)
    x = torch.tensor([[PAD] * pad + ids])
    m = torch.tensor([[0] * pad + [1] * len(ids)])
    mask, pos = parallel_inputs(x, m, [parallel], START, DECIDE, torch.float32)
    with torch.no_grad():
        hs = model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[0]
    ends = (x[0] == END).nonzero(as_tuple=True)[0]
    return hs[-1], [hs[e] for e in ends]


PREFIX = [11, 12, 13, 14, 15, 16]
OPTIONS = [[21, 22], [31, 32, 33, 34], [41], [51, 52, 53]]


def test_every_order_gives_the_same_states(model):
    decide0, ends0 = states(model, PREFIX, OPTIONS, parallel=True)
    by_option0 = {tuple(o): e for o, e in zip(OPTIONS, ends0, strict=True)}
    for order in itertools.permutations(range(len(OPTIONS))):
        opts = [OPTIONS[j] for j in order]
        decide, ends = states(model, PREFIX, opts, parallel=True)
        assert torch.allclose(decide, decide0, atol=1e-5)
        for o, e in zip(opts, ends, strict=True):
            assert torch.allclose(e, by_option0[tuple(o)], atol=1e-5)


def test_the_ordinary_causal_mask_is_not_invariant(model):
    decide0, _ = states(model, PREFIX, OPTIONS, parallel=False)
    decide1, _ = states(model, PREFIX, OPTIONS[::-1], parallel=False)
    assert not torch.allclose(decide1, decide0, atol=1e-3)


def test_left_padding_does_not_change_anything(model):
    a, ends_a = states(model, PREFIX, OPTIONS, parallel=True)
    b, ends_b = states(model, PREFIX, OPTIONS, parallel=True, pad_to=40)
    assert torch.allclose(a, b, atol=1e-5)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(ends_a, ends_b, strict=True))


def test_unflagged_rows_keep_the_causal_mask():
    x = torch.tensor([sequence(PREFIX, OPTIONS)])
    mask, pos = parallel_inputs(x, torch.ones_like(x), [False], START, DECIDE, torch.float32)
    n = x.shape[1]
    assert torch.equal(mask[0, 0] == 0, torch.tril(torch.ones(n, n, dtype=torch.bool)))
    assert pos[0].tolist() == list(range(n))


def test_only_choice_questions_are_flagged():
    assert is_choice("state\n\nQuestion (pick exactly one option):\nWhich?")
    assert not is_choice("state\n\nQuestion (answer yes or no):\nIs it?")
