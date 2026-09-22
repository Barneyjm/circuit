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


def option_spans(input_ids: torch.Tensor, rows: list[bool], start_id: int, decide_id: int) -> list[list[tuple[int, int]] | None]:
    """Per row: [(lo, hi)] for each option, plus the decide position as the last span's hi;
    None for rows not flagged or without at least two options and a decide token."""
    out: list[list[tuple[int, int]] | None] = []
    for i in range(input_ids.shape[0]):
        if not rows[i]:
            out.append(None)
            continue
        starts = (input_ids[i] == start_id).nonzero(as_tuple=True)[0].tolist()
        decide = (input_ids[i] == decide_id).nonzero(as_tuple=True)[0]
        if len(starts) < 2 or len(decide) == 0:
            out.append(None)
            continue
        bounds = starts + [int(decide[-1])]
        out.append([(bounds[j], bounds[j + 1]) for j in range(len(starts))])
    return out


def parallel_mask(attention_mask: torch.Tensor, spans: list, dtype: torch.dtype) -> torch.Tensor:
    """A 4D additive mask [B, 1, L, L]: causal, padding masked for every query (a padded
    position attends to itself so no row is empty), and each option blind to the others."""
    b, n = attention_mask.shape
    dev = attention_mask.device
    real = attention_mask.bool()
    allowed = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev)).unsqueeze(0).repeat(b, 1, 1) & real.unsqueeze(1)
    for i, sp in enumerate(spans):
        if sp:
            first = sp[0][0]
            for lo, hi in sp:
                allowed[i, lo:hi, first:lo] = False  # not the options written before it
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    allowed = allowed | (eye & ~real.unsqueeze(2))
    return torch.zeros(b, 1, n, n, dtype=dtype, device=dev).masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


def parallel_positions(position_ids: torch.Tensor, spans: list) -> torch.Tensor:
    """Restart every option's positions where the question ended, and put the decide token
    one past the longest option. Works on [B, L] and on Qwen3-VL's [3, B, L] multimodal
    positions, whose three axes agree on text tokens and are all moved the same way."""
    pos = position_ids.clone()
    lead = pos.ndim == 3
    for i, sp in enumerate(spans):
        if not sp:
            continue
        first, d = sp[0][0], sp[-1][1]
        base = pos[..., i, first].clone()  # scalar, or one value per axis
        for lo, hi in sp:
            off = torch.arange(hi - lo, device=pos.device)
            pos[..., i, lo:hi] = (base.unsqueeze(-1) if lead else base) + off
        longest = max(hi - lo for lo, hi in sp)
        tail = torch.arange(pos.shape[-1] - d, device=pos.device)
        pos[..., i, d:] = (base.unsqueeze(-1) if lead else base) + longest + tail
    return pos


def parallel_inputs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, rows: list[bool], start_id: int, decide_id: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and [B, L] positions for a text batch; rows not flagged keep the causal mask."""
    spans = option_spans(input_ids, rows, start_id, decide_id)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    return parallel_mask(attention_mask, spans, dtype), parallel_positions(position_ids, spans)
