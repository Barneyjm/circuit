"""Encode a choice question's options side by side instead of one after another.

A causal model reads option 3 having read options 1 and 2, and option 1 having read
neither, so the same options in a different order are different inputs and the answer
moves: 8% of top answers for Jev, 14 to 18% for our circuits, 23% for an untuned letter
readout (`scripts/eval_permutations.py`). Shuffling options in training shrinks that and
cannot remove it.

This removes it. Each option attends to the state, the question and itself, never to
another option, and its positions restart where the question ended, so its hidden states
do not depend on where in the list it was written. The decide token attends to all of
them from one position past the longest. Attention over keys whose positions carry no
order is a sum over a set, so the decide token's state, the option states the head reads,
and the probabilities are the same under any ordering, up to the order floating point
adds them in. Still one forward pass.

Only `choice` is encoded this way. A score's levels are ordered by meaning and a noul's
two options are always yes then no, so there is no order to be robust to.
"""

from __future__ import annotations

import torch

CHOICE_LEAD = "Question (pick exactly one option):"


def is_choice(text: str) -> bool:
    return CHOICE_LEAD in text


def parallel_inputs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, rows: list[bool], start_id: int, decide_id: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """A 4D additive attention mask [B, 1, L, L] and position ids [B, L].

    Rows flagged in `rows` get side-by-side options; the rest get the ordinary causal
    mask, so one batch can mix question types. Padding is on the left and is masked for
    every query; a padded position attends to itself so no softmax row is empty."""
    b, n = input_ids.shape
    dev = input_ids.device
    real = attention_mask.bool()
    allowed = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev)).unsqueeze(0).repeat(b, 1, 1) & real.unsqueeze(1)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    for i in range(b):
        if not rows[i]:
            continue
        starts = (input_ids[i] == start_id).nonzero(as_tuple=True)[0].tolist()
        decide = (input_ids[i] == decide_id).nonzero(as_tuple=True)[0]
        if len(starts) < 2 or len(decide) == 0:
            continue
        d = int(decide[-1])
        bounds = starts + [d]  # option j owns [bounds[j], bounds[j+1]): its text, its closing delimiter, the newline after
        first = starts[0]
        base = int(position_ids[i, first])  # where the question ended
        for j in range(len(starts)):
            lo, hi = bounds[j], bounds[j + 1]
            allowed[i, lo:hi, first:lo] = False  # not the options written before it
            position_ids[i, lo:hi] = base + torch.arange(hi - lo, device=dev)
        longest = max(bounds[j + 1] - bounds[j] for j in range(len(starts)))
        position_ids[i, d:] = base + longest + torch.arange(n - d, device=dev)
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    allowed = allowed | (eye & ~real.unsqueeze(2))
    mask = torch.zeros(b, 1, n, n, dtype=dtype, device=dev).masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask, position_ids
