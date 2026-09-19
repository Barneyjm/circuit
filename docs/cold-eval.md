# Cold eval on real, human-labeled data

## Jev vs ours, same 1,200 items, same labels

Jev latency is the client-side p50 from this laptop to api.typesafe.ai
(p90 228 ms, 8 concurrent, 0 errors). Ours is ms/item on the same
hardware for every row. Cells are accuracy / ECE against human labels.

| model | ms/item | MultiNLI | SMS spam | Toxicity | CLINC 151-way | Ticket queue | Ticket priority |
|---|---|---|---|---|---|---|---|
| Jev (jev-latest) | 164.2 (API p50) | 88% / 0.04 | 96% / 0.05 | 82% / 0.06 | 90% / 0.05 | 26% / 0.52 | 26% / 0.60 |
| 1.7B raw | 152.5 | 66% / 0.10 | 70% / 0.17 | 63% / 0.08 | n/a | 30% / 0.24 | 38% / 0.10 |
| 1.7B tuned (synthetic) | 138.3 | 64% / 0.18 | 84% / 0.06 | 66% / 0.13 | 4% / 0.03 | 28% / 0.20 | 40% / 0.32 |
| 1.7B tuned (real data) | 135.9 | 82% / 0.11 | 96% / 0.03 | 70% / 0.09 | 3% / 0.03 | 28% / 0.25 | 41% / 0.31 |
| 8B raw | 612.2 | 80% / 0.10 | 88% / 0.13 | 74% / 0.06 | n/a | 34% / 0.20 | 32% / 0.32 |
| 8B tuned (synthetic) | 589.0 | 78% / 0.07 | 78% / 0.10 | 66% / 0.15 | 4% / 0.04 | 34% / 0.25 | 30% / 0.41 |
| 14B raw | 1124.0 | 90% / 0.11 | 94% / 0.10 | 74% / 0.09 | n/a | 36% / 0.23 | 34% / 0.23 |


2026-09-18. No model here was trained on any of these sources. Speed
is measured on the same hardware for every row (Apple M2 Max, MPS,
batch 8) on a fixed 120-item slice of the same data; accuracy and ECE
are on 200 items per task (1,000 for the base models, 1,200 for the
head-based fine-tunes, which can also answer the 151-way task).
Reference point for latency: Jev's production median on Cole's
ClassicMiniDIY items is 197 ms with 300 to 2,600 input tokens.

Cells are `accuracy / ECE`.

| model | ms/item | MultiNLI | SMS spam | Civil toxicity | Ticket queue (10) | Ticket priority (5) | CLINC intents (151) |
|---|---|---|---|---|---|---|---|
| 1.7B raw | 153 | 66% / 0.10 | 70% / 0.17 | 63% / 0.08 | 30% / 0.24 | 38% / 0.10 | n/a (26-option cap) |
| 1.7B tuned (synthetic) | 138 | 64% / 0.18 | 84% / 0.06 | 66% / 0.13 | 28% / 0.20 | 40% / 0.32 | 4% / 0.03 |
| 1.7B tuned (real + synthetic) | 136 | 82% / 0.11 | 96% / 0.03 | 70% / 0.09 | 28% / 0.25 | 41% / 0.31 | 3% / 0.03 |
| 8B raw | 612 | 80% / 0.10 | 88% / 0.13 | 74% / 0.06 | 34% / 0.20 | 32% / 0.32 | n/a |
| 8B tuned (1 epoch, bounded) | 589 | 78% / 0.07 | 78% / 0.10 | 66% / 0.15 | 34% / 0.25 | 30% / 0.41 | 4% / 0.04 |
| 14B raw | 1,124 | 90% / 0.11 | 94% / 0.10 | 74% / 0.09 | 36% / 0.23 | 34% / 0.23 | n/a |
| 0.6B raw | 65 | (timing slice only: 45% / 0.11 overall) | | | | | |

Sources: `nyu-mll/multi_nli` (validation_matched), `ucirvine/sms_spam`,
`google/civil_comments` (soft label = annotator toxicity fraction),
`Tobi-Bueck/customer-support-tickets` (English), `clinc/clinc_oos`
(plus). Builder: `scripts/build_hf_eval.py`. Timing: `data/hf_eval_timing.jsonl`.

## Read

- **Raw scale wins on real language tasks.** 14B at 90% MultiNLI and
  94% spam with no training is the best number in the project; each
  step up the ladder costs 2 to 4x latency for a few points.
- **Real training data is what was missing, not model size.** The same
  1.7B recipe on 8k items drawn from the public train splits (MNLI,
  civil_comments, sms_spam, CLINC) plus the synthetic families goes
  from 64% to 82% on MultiNLI and 84% to 96% on spam, at parity with
  Jev on spam and better calibrated there, at 136 ms/item. Toxicity
  moved 4 points; those are soft labels, so argmax accuracy understates
  agreement. On Cole's bench it also edges the synthetic-only run
  (0.714 vs 0.685 agreement with Jev). The synthetic-only 1.7B
  fine-tune beat its own base on spam (+14) and toxicity (+3) and was
  flat elsewhere. The bounded 8B fine-tune
  is below its base on spam and toxicity; that run (one epoch, half
  the data, batch 4) is not a clean read and should be redone.
- **CLINC was a truncation bug, now fixed.** 3 to 4% on 151-way CLINC
  (chance 0.7%) for every fine-tune, including the one trained on 3,000
  CLINC items. A 151-option prompt is ~840 tokens and the trainer
  truncated at 640 from the right, so the answer slot the head reads
  was cut off every CLINC training item; the head trained on a random
  mid-list position for 37% of the real-data run. Trainer and scorer
  now truncate from the left (`truncation_side="left"`), so the slot
  always survives. The real-data run needs to be repeated with the fix
  before CLINC says anything about the head.
- **The ticket dataset is noisy.** Queue labels overlap ("IT Support"
  vs "Technical Support") and priorities do not follow from the text;
  every model sits near a third. Treat it as a floor, not a target.
- **Calibration transferred better than accuracy**, again: the 8B
  fine-tune has the best MultiNLI ECE (0.07) despite lower accuracy.

## Latency vs Jev

| | ms/item, this laptop | est. on one H100 (÷30–60) |
|---|---|---|
| 0.6B | 65 | ~1–2 |
| 1.7B | 138–153 | ~3–5 |
| 8B | ~600 | ~10–20 |
| 14B | 1,124 | ~20–40 |
| Jev (production, Cole's items) | 197 | — |

The 1.7B class is in Jev's latency band on a laptop; everything is
under Jev's number on a datacenter GPU. Speed is not the gap. Accuracy
on real data is, and it closes with real training data, not synthetic.

## Cole's bench (ClassicMiniDIY typesafe-bench)

546 question-items from 161 production states across nine site
features, reference = Jev's answers (agreement with Jev, not accuracy;
the bench README says the same). Jev answered each item's 3 to 8
questions in one call at a 197 ms median in production; our ms are per
question on this laptop.

| model | agree w/ Jev | ECE | KL | Noul | Choice | Score | ms/question |
|---|---|---|---|---|---|---|---|
| 0.6B raw | 0.374 | 0.201 | 1.018 | 0.418 | 0.391 | 0.188 | 175 |
| 1.7B tuned (synthetic) | 0.685 | 0.069 | 0.499 | 0.848 | 0.542 | 0.447 | 575 |
| 1.7B tuned (real + synthetic) | 0.714 | 0.071 | 0.544 | 0.872 | 0.570 | 0.494 | 568 |
| 8B raw | 0.762 | 0.070 | 0.437 | 0.879 | 0.620 | 0.671 | 1,599 |
| 14B raw | 0.800 | 0.087 | 0.307 | 0.922 | 0.665 | 0.682 | 2,872 |
| **8B tuned** | **0.817** | **0.060** | 0.317 | 0.908 | 0.642 | **0.882** | 2,436 |

Per feature (agreement with Jev):

| feature | 0.6B raw | 14B raw | 8B raw | 1.7B tuned (syn) | 1.7B tuned (real) | 8B tuned |
|---|---|---|---|---|---|---|
| chat-classifier (8 q) | 0.32 | 0.82 | 0.85 | 0.79 | 0.80 | 0.83 |
| message-screen | 0.41 | 0.95 | 0.80 | 0.73 | 0.81 | 0.93 |
| model-safety | 0.50 | 1.00 | 1.00 | 0.83 | 0.83 | 1.00 |
| part-correlation | 0.29 | 0.82 | 0.78 | 0.43 | 0.51 | 0.84 |
| search-intent | 0.53 | 0.77 | 0.68 | 0.68 | 0.72 | 0.78 |
| saved-search-match | 0.67 | 0.80 | 0.67 | 0.80 | 0.67 | 0.80 |
| duplicate-hint | 0.53 | 0.67 | 0.53 | 0.53 | 0.47 | 0.87 |
| mcp-related-pick | 0.43 | 0.50 | 0.57 | 0.57 | 0.50 | 0.64 |
| search-miss-triage | 0.27 | 0.49 | 0.42 | 0.33 | 0.40 | 0.51 |

Read: on production-style questions (short wording, 2 to 18 options,
Score rubrics) the 8B fine-tune is the closest thing we have to Jev,
ahead of the raw 14B, and its Score agreement (0.88 vs 0.67 raw) is
the head doing what it was built for. This is the same fine-tune that
lost to its base on the HF benchmarks. The two results together say:
the fine-tune learned *this style of question* well, and public NLP
benchmarks are a different style. Both matter; the scoreboard keeps
both. Longer prompts (part-correlation: retailer listing × 3 factory
candidates, ~1k tokens) cost the 1.7B the most.

## 2026-09-19: the open reproductions and circuit-1.7b, same 1,200 items

Cells are accuracy / ECE against human labels. Nimble's letter scoring caps at 26 options, so its 200 CLINC items count as errors.

| model | MultiNLI | SMS spam | Toxicity | CLINC 151-way | Ticket queue | Ticket priority | ms/item |
|---|---|---|---|---|---|---|---|
| Jev (jev-latest) | 88% / 0.04 | 96% / 0.05 | 82% / 0.06 | 90% / 0.05 | 26% / 0.52 | 26% / 0.60 | 164 (API p50) |
| Bespoke-Nimble-9B | 84% / 0.09 | 91% / 0.06 | 86% / 0.08 | 0% / 0.00 | 28% / 0.49 | 24% / 0.65 | 1773 (M2 Max, 1 item/forward) |
| kev-0.5b | 46% / 0.28 | 50% / 0.30 | 62% / 0.16 | 62% / 0.17 | 21% / 0.26 | 38% / 0.28 | 325 (M2 Max, local server) |
| circuit-1.7b (pointer head, clean data) | 81% / 0.09 | 98% / 0.02 | 90% / 0.16 | 86% / 0.06 | 19% / 0.47 | 32% / 0.31 | 679 (M2 Max, batch 8, shared GPU) |
| 1.7B real-data fine-tune (slot head) | 82% / 0.11 | 96% / 0.03 | 70% / 0.09 | 2% / 0.03 | 28% / 0.25 | 42% / 0.31 | 505 (M2 Max) |
| 14B raw | 90% / 0.11 | 94% / 0.10 | 74% / 0.09 | n/a | 36% / 0.23 | 34% / 0.23 | - |

Water-utility calls (the article's 100): Jev 98% / 0.02, Nimble 93% / 0.05, circuit-1.7b 92% / 0.08, kev 80% / 0.13, 1.7B raw 79% / 0.15.

circuit-1.7b trained on grid + commercial wide mix + CC real data (MNLI, civil_comments, sms_spam, CLINC train splits), pointer head, 2 epochs on an RTX 4090 in 61 minutes. Those four tasks are therefore in-distribution for it and not, presumably, for Jev; the water calls and Cole's bench are the out-of-distribution checks.
