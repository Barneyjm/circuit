# The generalization grid

There is an infinite space of questions. You cannot chase it with
datasets. But the *structure* of a judgment is finite: what operation
the question asks for, how the state is laid out, what the answer
space looks like. The base model already knows about the world; the
fine-tune teaches it the format. So the training distribution should
span structure, not subject matter, and generalization should be
measured by holding out structure, not items.

`s1proto/data/grid.py` generates one cell per (operation, format) with
labels **computed by code**. No teacher, no dataset license, no label
noise. `scripts/build_grid.py` builds it; `--holdout "temporal/*,*/thread"`
keeps whole rows or columns out of training so the eval answers "can it
do an operation it never saw" and "can it read a format it never saw".

## Operations (rows)

| operation | asks | answer |
|---|---|---|
| extract | does the text mention a phone / email / date / price / order id? | noul |
| classify | which of these categories is the message? (3, 8, or 24 options; with or without descriptions; sometimes "other") | choice |
| compare | is A's field strictly lower than B's? which of these has the lowest? | noul / choice |
| consistency | is the customer's claim consistent with the order record? | noul |
| count | how many listed items are flagged? | score 0 / 1 / 2 / 3+ |
| rule | does the record satisfy a two- or three-clause rule? | noul |
| temporal | did event A happen before event B? (five date formats) | noul |
| negation | is the reply declining the offer? (with double negatives and "can't say no") | noul |
| ordinal | how severe is the incident? | score, 4 levels |

## Formats (columns)

`string` prose with the fields inline; `json` flat object; `nested`
object with sub-objects; `list` several records, the question names one
by id; `thread` a chat transcript with the fact in one turn and filler
around it; `document` the fact buried in unrelated paragraphs.

## Ambiguity

About 5% of items are built to be undecidable (a vague claim, equal
values under a strict "lower than", a date with no year, "let me think
about it") and carry a soft label of 0.5. A model that learns the grid
learns that a flat distribution is sometimes the right answer, which is
what the uncertainty bands in decision circuits depend on.

## Sizes

150 train + 40 eval per cell: 8,100 / 2,125 items over 54 cells. Items
are short (100 to 400 tokens), so a 1.7B epoch is under two hours on a
laptop.

## Protocol

1. Baseline every model on the full eval grid (no grid training).
2. Train on the grid with one operation row and one format column held
   out; report the held-out row, the held-out column, and the rest.
3. Train on the full grid; re-run the old scoreboard and Cole's bench to
   see whether structural training transfers to real questions (the
   refund-desk consistency gap is the first thing to check).

## Results

Filled in by `scripts/grid_report.py --md`.

### Baselines, 2026-09-19 (no grid training)

**jev-latest  accuracy overall 95%, 178 ms/item**

| operation | string | json | nested | list | thread | document | mean |
|---|---|---|---|---|---|---|---|
| extract | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| classify | 90% | 92% | 92% | 95% | 92% | 62% | 87% |
| compare | 100% | 100% | 100% | 100% | 100% | 97% | 100% |
| consistency | 100% | 94% | 97% | 100% | 94% | 100% | 98% |
| count | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| rule | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| temporal | 97% | 100% | 100% | 100% | 100% | 100% | 100% |
| negation | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| ordinal | 91% | 90% | 93% | 92% | 92% | 82% | 90% |
| **mean** | 98% | 97% | 98% | 99% | 98% | 94% | |

Ambiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): compare (heldout) conf 0.95 ece 0.05, consistency (heldout) conf 0.76 ece 0.47, extract (heldout) conf 0.97 ece 0.42, negation (heldout) conf 0.82 ece 0.78, temporal (heldout) conf 0.94 ece 0.39

**Qwen3-1.7B-Base  accuracy overall 59%, 255 ms/item**

| operation | string | json | nested | list | thread | document | mean |
|---|---|---|---|---|---|---|---|
| extract | 77% | 97% | 89% | 75% | 95% | 69% | 84% |
| classify | 82% | 67% | 75% | 65% | 60% | 52% | 67% |
| compare | 54% | 45% | 41% | 44% | 61% | 45% | 48% |
| consistency | 43% | 55% | 56% | 48% | 50% | 45% | 49% |
| count | 45% | 52% | 60% | 48% | 55% | 32% | 49% |
| rule | 28% | 32% | 40% | 50% | 40% | 35% | 38% |
| temporal | 43% | 44% | 61% | 41% | 56% | 54% | 50% |
| negation | 67% | 48% | 53% | 84% | 97% | 83% | 72% |
| ordinal | 71% | 55% | 73% | 78% | 48% | 60% | 64% |
| **mean** | 57% | 55% | 61% | 59% | 62% | 53% | |

Ambiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): compare (heldout) conf 0.66 ece 0.42, consistency (heldout) conf 0.73 ece 0.27, extract (heldout) conf 0.75 ece 0.13, negation (heldout) conf 0.60 ece 0.30, temporal (heldout) conf 0.65 ece 0.31

**lora:1.7b-real  accuracy overall 68%, 210 ms/item**

| operation | string | json | nested | list | thread | document | mean |
|---|---|---|---|---|---|---|---|
| extract | 91% | 97% | 100% | 83% | 95% | 85% | 92% |
| classify | 70% | 64% | 57% | 45% | 55% | 42% | 56% |
| compare | 84% | 57% | 62% | 72% | 58% | 63% | 66% |
| consistency | 77% | 91% | 86% | 70% | 64% | 71% | 77% |
| count | 57% | 70% | 82% | 78% | 68% | 55% | 68% |
| rule | 78% | 78% | 75% | 62% | 60% | 65% | 70% |
| temporal | 57% | 62% | 61% | 59% | 44% | 49% | 55% |
| negation | 61% | 48% | 76% | 79% | 91% | 83% | 73% |
| ordinal | 74% | 87% | 80% | 70% | 80% | 68% | 76% |
| **mean** | 72% | 73% | 76% | 69% | 68% | 64% | |

Ambiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): compare (heldout) conf 0.67 ece 0.37, consistency (heldout) conf 0.73 ece 0.59, extract (heldout) conf 0.86 ece 0.20, negation (heldout) conf 0.78 ece 0.66, temporal (heldout) conf 0.56 ece 0.20

Water-utility calls (the article's 100): Jev 98% / ECE 0.02; 1.7B raw 79% / 0.15; 1.7B real-data fine-tune 61% / 0.12.

### Held-out run, 2026-09-19: 1.7B trained on the grid minus the temporal row and the thread column

**lora:1.7b-grid-holdout  accuracy overall 93%, 203 ms/item**

| operation | string | json | nested | list | thread | document | mean |
|---|---|---|---|---|---|---|---|
| extract | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| classify | 95% | 92% | 95% | 98% | 85% | 85% | 92% |
| compare | 97% | 92% | 95% | 79% | 97% | 95% | 93% |
| consistency | 100% | 100% | 100% | 90% | 97% | 100% | 98% |
| count | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| rule | 100% | 100% | 100% | 98% | 95% | 98% | 98% |
| temporal | 54% | 62% | 67% | 62% | 47% | 51% | 57% |
| negation | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| ordinal | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **mean** | 94% | 94% | 95% | 92% | 91% | 92% | |

Ambiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): compare (heldout) conf 0.95 ece 0.05, consistency (heldout) conf 0.94 ece 0.06, extract (heldout) conf 0.89 ece 0.17, negation (heldout) conf 0.73 ece 0.17, temporal (heldout) conf 0.78 ece 0.12

Format generalizes (held-out thread column 91% vs 94% in-distribution); operation does not (held-out temporal row 57% vs 50% raw). Transfer to real data: water calls 93% / ECE 0.06 (raw 79%, real-data fine-tune 61%, Jev 98%); cold eval 41% (grid-only training narrows general NLP; the publish run mixes grid, wide-mix, and real data). Ambiguous items stayed overconfident (conf 0.73 to 0.95) at 5% soft-label share; the publish set oversamples them 3x.

### kev-0.5b (jaredpalmer/kev-0.5b) on the grid

**kev-latest  accuracy overall 48%, 244 ms/item**

| operation | string | json | nested | list | thread | document | mean |
|---|---|---|---|---|---|---|---|
| extract | 74% | 55% | 89% | 58% | 65% | 72% | 69% |
| classify | 70% | 56% | 65% | 40% | 35% | 35% | 50% |
| compare | 32% | 35% | 38% | 33% | 45% | 24% | 35% |
| consistency | 60% | 76% | 61% | 35% | 44% | 55% | 55% |
| count | 20% | 22% | 12% | 20% | 15% | 15% | 18% |
| rule | 28% | 30% | 38% | 42% | 38% | 38% | 35% |
| temporal | 57% | 59% | 47% | 65% | 44% | 46% | 53% |
| negation | 52% | 58% | 56% | 55% | 41% | 54% | 53% |
| ordinal | 79% | 77% | 77% | 68% | 45% | 52% | 66% |
| **mean** | 52% | 52% | 54% | 46% | 41% | 43% | |

Ambiguous items (soft label 0.5; ideal: mean confidence near 0, ECE near 0): compare (heldout) conf 0.43 ece 0.24, consistency (heldout) conf 0.70 ece 0.12, extract (heldout) conf 0.77 ece 0.45, negation (heldout) conf 0.75 ece 0.22, temporal (heldout) conf 0.72 ece 0.69
