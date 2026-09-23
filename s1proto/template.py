"""Prompt templates: one per question type, one sequence per question.

Layout (Phase 1):

    <state>

    <instructions>
    A. <option 1>
    B. <option 2>
    ...
    Answer:

The scorer reads the next-token logits after "Answer:" and keeps only
the label tokens (" A", " B", ...). Everything about the prompt that
the model could be biased by (label letters, option order, the word
"Answer") is a known Phase 1 weakness and is what Phase 2 corrects.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any

from s1proto.schema import ChoiceQuestion, LocateQuestion, MatchQuestion, MultiQuestion, NoulQuestion, RankQuestion, ScoreQuestion

# A..Z are single tokens and are what the label-token scorer reads
# (Phase 1, 26-option cap). Beyond 26 the labels are two letters; only a
# head-based scorer (Phase 3) can answer those, and it ignores the
# label text anyway, so the letters just keep the option list legible.
LETTER_LABELS = list(string.ascii_uppercase)
LABELS = LETTER_LABELS + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
LABELS = LABELS[:255]
MAX_OPTIONS = len(LABELS)  # 255, the TypeSafe cap
MAX_LETTER_OPTIONS = len(LETTER_LABELS)

NOUL_OPTIONS = ("yes", "no")

# Pointer layout: each option is wrapped in delimiter tokens and the
# sequence ends with a decide token. The head reads the hidden state at
# every option's closing delimiter and at the decide token, so options
# need no letters, the count is unbounded, and their order carries no
# slot position. These are Qwen reserved tokens that never occur in
# ordinary text; user text is sanitized so it cannot forge them.
OPT_START = "<|box_start|>"
OPT_END = "<|box_end|>"
DECIDE = "<|fim_middle|>"
# Alternatives for tokenizers without the box tokens. Qwen2-Audio's vocabulary
# drops them but carries Whisper-style timestamp tokens that the processor
# never emits for an input, so three of those serve as delimiters.
POINTER_TOKEN_SETS: dict[str, tuple[str, str, str]] = {
    "qwen": ("<|box_start|>", "<|box_end|>", "<|fim_middle|>"),
    "qwen2-audio": ("<|29.98|>", "<|29.99|>", "<|30.00|>"),
}
_RESERVED_EXTRA = (
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
)
RESERVED = (OPT_START, OPT_END, DECIDE, *_RESERVED_EXTRA)
# `locate` marks the end of every candidate in the state with this token; the head reads
# the hidden state there, as it reads an option's closing delimiter. Text models only.
LOCATE_MARK = "<|object_ref_end|>"
MAX_LOCATE_CANDIDATES = 512
# `match` writes its items after the options, each wrapped in these; the head reads one query
# per item, at its closing token. After the options so each item has read all of them.
ITEM_START = "<|quad_start|>"
ITEM_END = "<|quad_end|>"
_SENTINEL = "\ue000"  # private-use; stands in for LOCATE_MARK until the state is sanitized
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

CHOICE_LEAD = "Question (pick exactly one option):"
MULTI_LEAD = "Question (pick every option that applies; none may apply):"
LOCATE_LEAD = "Question (point to the part of the input that answers this, or none):"
# Question types past noul/choice/score, each read with its own query in a v2 head.
V2_KINDS = ("multi", "locate", "rank", "match")
RANK_LEAD = "Question (order the options, best first):"
MATCH_LEAD = "Question (match each item below to one option, or none):"


def pointer_tokens_for(tokenizer) -> tuple[str, str, str]:
    """The first delimiter set whose three tokens are single ids in this tokenizer."""
    for name, toks in POINTER_TOKEN_SETS.items():
        if all(len(tokenizer.encode(t, add_special_tokens=False)) == 1 for t in toks):
            return toks
    raise ValueError("no pointer delimiter set is a single token in this tokenizer; add one to POINTER_TOKEN_SETS")


def use_pointer_tokens(start: str, end: str, decide: str) -> None:
    """Switch the pointer layout's delimiters process-wide (trainer and
    scorer call this with the set recorded in a run's config.json)."""
    global OPT_START, OPT_END, DECIDE, RESERVED
    OPT_START, OPT_END, DECIDE = start, end, decide
    RESERVED = (OPT_START, OPT_END, DECIDE, *_RESERVED_EXTRA)


LAYOUTS = ("letters", "pointer")


def sanitize(text: str) -> str:
    for tok in RESERVED:
        text = text.replace(tok, tok.replace("<|", "<").replace("|>", ">"))
    return text


@dataclass(frozen=True)
class Prompt:
    """A rendered sequence plus how to map label positions back.

    `text == prefix + tail`. `prefix` is the rendered state (shared by
    every question in a request, so the scorer can prefill it once);
    `tail` is the question-specific part ending in the answer slot."""

    text: str
    n_options: int
    option_keys: tuple[str, ...]  # what each label position means to the caller
    prefix: str = ""
    tail: str = ""
    layout: str = "letters"
    kind: str = "choice"  # noul | choice | score | multi | locate | rank | match: temperature and readout differ by type
    item_keys: tuple[str, ...] = ()  # match: one row of option logits per item

    @property
    def n_logits(self) -> int:
        """Logits the head returns: one per option, or for match one per option per item."""
        return self.n_options * max(1, len(self.item_keys))


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_text(value: Any) -> str:
    """Instructions and criteria may be strings or JSON structure.
    Strings pass through; structure is rendered as compact JSON so the
    model sees the nesting without a wall of indentation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _options_block(options: list[tuple[str, str]]) -> str:
    """options: [(name, description)] -> 'A. name — description' lines.
    Description omitted when empty so null-criteria options stay tidy."""
    if len(options) > MAX_OPTIONS:
        raise ValueError(f"at most {MAX_OPTIONS} options in the letters layout, got {len(options)}")
    lines = []
    for label, (name, desc) in zip(LABELS, options, strict=False):
        lines.append(f"{label}. {name} — {desc}" if desc else f"{label}. {name}")
    return "\n".join(lines)


def _pointer_block(options: list[tuple[str, str]]) -> str:
    lines = []
    for name, desc in options:
        body = f"{name} — {desc}" if desc else name
        lines.append(f"{OPT_START}{sanitize(body)}{OPT_END}")
    return "\n".join(lines)


def _assemble(state: Any, instructions: Any, options: list[tuple[str, str]], lead: str, layout: str = "letters") -> tuple[str, str]:
    """Returns (prefix, tail). The prefix ends with the blank line so
    the tail starts at a clean token boundary ("\\n\\n" is its own
    token in the Qwen/Llama vocabularies)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if layout == "pointer":
        prefix = f"{sanitize(render_state(state))}\n\n"
        tail = f"{lead}\n{sanitize(render_text(instructions))}\n{_pointer_block(options)}\n{DECIDE}"
        return prefix, tail
    prefix = f"{render_state(state)}\n\n"
    tail = f"{lead}\n{render_text(instructions)}\n{_options_block(options)}\nAnswer:"
    return prefix, tail


def render_noul(state: Any, q: NoulQuestion, layout: str = "letters") -> Prompt:
    yes_desc = render_text(q.criteria.true) if q.criteria else ""
    no_desc = render_text(q.criteria.false) if q.criteria else ""
    options = [("yes", yes_desc), ("no", no_desc)]
    prefix, tail = _assemble(state, q.instructions, options, lead="Question (answer yes or no):", layout=layout)
    return Prompt(text=prefix + tail, n_options=2, option_keys=NOUL_OPTIONS, prefix=prefix, tail=tail, layout=layout, kind="noul")


def render_choice(state: Any, q: ChoiceQuestion, layout: str = "letters") -> Prompt:
    names = list(q.criteria.keys())
    options = [(name, render_text(q.criteria[name])) for name in names]
    prefix, tail = _assemble(state, q.instructions, options, lead=CHOICE_LEAD, layout=layout)
    return Prompt(text=prefix + tail, n_options=len(names), option_keys=tuple(names), prefix=prefix, tail=tail, layout=layout)


def render_score(state: Any, q: ScoreQuestion, layout: str = "letters") -> Prompt:
    # Levels are ordered; the level index is the option key so the
    # caller can compute the expected value and the legend.
    options = [(render_text(level), "") for level in q.criteria]
    prefix, tail = _assemble(state, q.instructions, options, lead="Question (pick the level that fits best; levels are in increasing order):", layout=layout)
    return Prompt(
        text=prefix + tail,
        n_options=len(options),
        option_keys=tuple(str(i) for i in range(len(options))),
        prefix=prefix,
        tail=tail,
        layout=layout,
        kind="score",
    )


def render_multi(state: Any, q: MultiQuestion, layout: str = "pointer") -> Prompt:
    if layout != "pointer":
        raise ValueError("multi needs a pointer-layout model")
    names = list(q.criteria.keys())
    options = [(name, render_text(q.criteria[name])) for name in names]
    prefix, tail = _assemble(state, q.instructions, options, lead=MULTI_LEAD, layout=layout)
    return Prompt(text=prefix + tail, n_options=len(names), option_keys=tuple(names), prefix=prefix, tail=tail, layout=layout, kind="multi")


def render_rank(state: Any, q: RankQuestion, layout: str = "pointer") -> Prompt:
    if layout != "pointer":
        raise ValueError("rank needs a pointer-layout model")
    names = list(q.criteria.keys())
    options = [(name, render_text(q.criteria[name])) for name in names]
    prefix, tail = _assemble(state, q.instructions, options, lead=RANK_LEAD, layout=layout)
    return Prompt(text=prefix + tail, n_options=len(names), option_keys=tuple(names), prefix=prefix, tail=tail, layout=layout, kind="rank")


def render_match(state: Any, q: MatchQuestion, layout: str = "pointer") -> Prompt:
    if layout != "pointer":
        raise ValueError("match needs a pointer-layout model")
    names = list(q.criteria.keys())
    options = [(name, render_text(q.criteria[name])) for name in names] + [("none", render_text(q.none) or "no option matches")]
    items = "\n".join(f"{ITEM_START}{sanitize(f'{k} — {render_text(v)}' if render_text(v) else k)}{ITEM_END}" for k, v in q.items.items())
    prefix, tail = _assemble(state, q.instructions, options, lead=MATCH_LEAD, layout=layout)
    tail = tail.removesuffix(DECIDE) + f"Items:\n{items}\n{DECIDE}"
    keys = (*names, "none")
    return Prompt(text=prefix + tail, n_options=len(keys), option_keys=keys, prefix=prefix, tail=tail, layout=layout, kind="match", item_keys=tuple(q.items))


def locate_candidates(state: Any) -> list[tuple[str, str]]:
    """(path, text) for every candidate a locate question can point at, in reading order:
    each non-empty string value of a JSON state, or each sentence of a plain-text one."""
    if isinstance(state, str):
        return [(f"sentence[{i}]", t) for i, t in enumerate(x for x in _SENTENCE.split(state.strip()) if x.strip())]
    out: list[tuple[str, str]] = []

    def walk(v: Any, path: str) -> None:
        if isinstance(v, str):
            if v.strip():
                out.append((path or "state", v))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else str(k))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")

    walk(state, "")
    return out


def _marked_state(state: Any) -> str:
    """The state rendered as usual with LOCATE_MARK after each candidate."""
    if isinstance(state, str):
        body = " ".join(t.replace(_SENTINEL, "") + _SENTINEL for _, t in locate_candidates(state))
    else:

        def mark(v: Any) -> Any:
            if isinstance(v, str):
                return v.replace(_SENTINEL, "") + _SENTINEL if v.strip() else v
            if isinstance(v, dict):
                return {k: mark(x) for k, x in v.items()}
            if isinstance(v, list):
                return [mark(x) for x in v]
            return v

        body = json.dumps(mark(state), ensure_ascii=False, indent=2)
    return sanitize(body).replace(_SENTINEL, LOCATE_MARK)


def render_locate(state: Any, q: LocateQuestion, layout: str = "pointer") -> Prompt:
    if layout != "pointer":
        raise ValueError("locate needs a pointer-layout model")
    cands = locate_candidates(state)
    if not cands:
        raise ValueError("locate needs a state with text in it")
    if len(cands) > MAX_LOCATE_CANDIDATES:
        raise ValueError(f"locate supports at most {MAX_LOCATE_CANDIDATES} candidates, the state has {len(cands)}")
    prefix = f"{_marked_state(state)}\n\n"
    none = render_text(q.criteria) or "the input does not say"
    tail = f"{LOCATE_LEAD}\n{sanitize(render_text(q.instructions))}\n{_pointer_block([('none', none)])}\n{DECIDE}"
    keys = tuple(p for p, _ in cands) + ("none",)
    return Prompt(text=prefix + tail, n_options=len(keys), option_keys=keys, prefix=prefix, tail=tail, layout=layout, kind="locate")


def render(
    state: Any, q: NoulQuestion | ChoiceQuestion | ScoreQuestion | MultiQuestion | LocateQuestion | RankQuestion | MatchQuestion, layout: str = "letters"
) -> Prompt:
    if isinstance(q, NoulQuestion):
        return render_noul(state, q, layout)
    if isinstance(q, ChoiceQuestion):
        return render_choice(state, q, layout)
    if isinstance(q, MultiQuestion):
        return render_multi(state, q, layout)
    if isinstance(q, LocateQuestion):
        return render_locate(state, q, layout)
    if isinstance(q, RankQuestion):
        return render_rank(state, q, layout)
    if isinstance(q, MatchQuestion):
        return render_match(state, q, layout)
    return render_score(state, q, layout)
