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

## Cole's bench, 2026-09-19 (546 production questions, agreement with Jev / ECE)

| model | agreement | ECE | noul | choice | score |
|---|---|---|---|---|---|
| Bespoke-Nimble-9B | 0.84 | 0.05 | | | |
| 8B tuned on synthetic (Jev-derived targets; not published) | 0.82 | 0.06 | 0.91 | 0.64 | 0.88 |
| 14B raw | 0.80 | 0.09 | 0.92 | 0.66 | 0.68 |
| 1.7B real-data fine-tune (slot head) | 0.71 | 0.07 | 0.87 | 0.57 | 0.49 |
| circuit-1.7b | 0.70 | 0.05 | 0.83 | 0.53 | 0.64 |
| kev-0.5b | 0.49 | 0.11 | 0.56 | 0.42 | 0.38 |

Nimble leads the open models on production-style questions. circuit-1.7b is the best calibrated but mid-pack on agreement: clean data bought calibration and the public benchmarks, not this style of question. circuit-8b (clean data, pointer head) is the direct test of whether size closes it.

## circuit-8b (added 2026-09-19)

Same recipe as circuit-1.7b on Qwen3-8B-Base, 1 epoch, bf16, 75 minutes on
an RTX A6000; best validation ECE 0.014 at step 2,800. Accuracy / ECE.

| | MultiNLI | SMS spam | toxicity | CLINC 151-way | water calls | grid | typesafe-bench (agree / ECE) |
|---|---|---|---|---|---|---|---|
| circuit-8b | 86% / 0.08 | 98% / 0.02 | 93% / 0.14 | 95% / 0.03 | 93% / 0.05 | 98% | 0.84 / 0.04 |
| circuit-1.7b | 81% / 0.09 | 98% / 0.02 | 90% / 0.16 | 86% / 0.06 | 92% / 0.08 | 97% | 0.70 / 0.05 |
| Jev | 88% / 0.04 | 96% / 0.05 | 82% / 0.06 | 90% / 0.05 | 98% / 0.02 | 95% | 1.0 by definition |

Throughput on the A6000, batch 8: 72 ms per grid item, 178 ms per cold-eval
item, 198 ms per typesafe-bench item. Results: `results/*_circuit-8b.json`.

## circuit-vl-4b (added 2026-09-19)

LoRA on the language model of Qwen3-VL-4B-Instruct plus the pointer head,
trained on the vision grid (1,083 items, 2 epochs, 7 minutes on an A6000;
best validation ECE 0.016 at step 500). 300 held-out grid items:

| model | accuracy | ECE | ms/item (A6000) |
|---|---|---|---|
| Qwen3-VL-4B-Instruct raw, letter logits | 96.0% | 0.041 | 61 |
| circuit-vl-4b | 98.3% | 0.018 | 90 |

Weak cells: count/table 93%, negation/scene 97%. On the blurred ambiguous
compare items mean confidence is 0.93 (raw: 1.0); still overconfident.
Results: `results/vgrid_circuit-vl-4b.json`, `results/vgrid_qwen3vl4b_raw_gpu.json`.

## circuit-audio-7b (added 2026-09-19)

LoRA on the language model of Qwen2-Audio-7B-Instruct plus the pointer
head (timestamp tokens as delimiters), trained on the audio grid (1,000
synthesized clips, 1 epoch, 35 minutes on an Apple laptop GPU; best
validation ECE 0.082 at step 400). 300 held-out clips:

| model | decidable (279) acc / ECE | all 300 acc / ECE | ambiguous mean conf (21) |
|---|---|---|---|
| Qwen2-Audio-7B-Instruct raw, letter logits | 72.0% / 0.188 | 68.7% / 0.214 | 0.67 |
| circuit-audio-7b | 93.5% / 0.072 | 90.0% / 0.109 | 0.63 |

Biggest gains: count/sounds 15% -> 67%, count/list 63% -> 89%, negation/list
55% -> 100%, compare/numbers 46% -> 89%. Weakest cell after training:
count/sounds. Results: `results/agrid_circuit-audio-7b.json`,
`results/agrid_qwen2audio_raw.json`.

## circuit-14b (trained 2026-09-19, not published)

Same recipe on Qwen3-14B-Base, 1 epoch at batch 2 with gradient checkpointing,
132 minutes on an RTX A6000, best validation ECE 0.032 at step 4,000. It does
not beat circuit-8b, so it is not in the family. Accuracy / ECE:

| | MultiNLI | SMS spam | toxicity | CLINC 151-way | water calls | grid | typesafe-bench (agree / ECE) |
|---|---|---|---|---|---|---|---|
| circuit-14b | 85.5% / 0.120 | 96.0% / 0.039 | 85.0% / 0.143 | 93.5% / 0.051 | 93% / 0.037 | 98% | 0.857 / 0.070 |
| circuit-8b | 86.0% / 0.083 | 97.5% / 0.023 | 92.5% / 0.135 | 95.0% / 0.031 | 93% / 0.048 | 98% | 0.841 / 0.039 |

Reading: a 1.6-point gain in agreement on the production questions, paid for
with worse calibration everywhere (Brier 0.140 vs 0.128 there) and lower
accuracy on all four cold-eval tasks. The likelier cause is the training
budget (half the batch size over the same single epoch, early-stopped at
step 4,000 of 6,800) than the base; with the 8B already at or above Jev on
most rows, the next spend goes to data, not parameters. Timing on the A6000:
n/a ms per cold-eval item batched. Results: `results/*_circuit-14b.json`.

## circuit-audio-7b v2 (added 2026-09-19, published over v1)

Audio grid v2: 14 cells, 1,400 training clips. Scripted calls, lists, and
readbacks in 27 Kokoro voices with random blends; real LibriSpeech
dev-clean recordings (CC BY 4.0) with mention / which-sentence / word-order
questions labeled from transcripts; Free Spoken Digit Dataset (CC BY-SA
4.0) digits stitched into account numbers; generated sounds. Same recipe,
1 epoch, 55 minutes on the laptop, best validation ECE 0.052 at step 300.
420 held-out clips:

| model | decidable (385) acc / ECE | all 420 acc / ECE | ambiguous mean conf (35) |
|---|---|---|---|
| Qwen2-Audio-7B-Instruct raw, letter logits | 73.2% / 0.199 | 68.1% / 0.232 | 0.69 |
| circuit-audio-7b v2 | 94.5% / 0.039 | 88.8% / 0.073 | 0.54 |

Weak cell: count/sounds 44% (raw 8%). Results:
`results/agrid2_circuit-audio-7b-v2.json`, `results/agrid2_qwen2audio_raw.json`.

## circuit-vl-4b v2 (added 2026-09-19, published over v1)

Vision grid v2 adds three real-photo cells from 700 Open Images V7
validation photos (images CC BY 2.0 by their Flickr authors, labels CC BY
4.0; per-image attribution in `data/vision/sources/openimages/index.json`):
present/photo, classify/photo, negation/photo, labeled by the human-verified
image labels. 16 cells, 1,408 training items, 2 epochs, 85 minutes on the
laptop, best validation ECE 0.035 at step 600. 390 held-out items:

| model | all 390 acc / ECE | real photos (86) | rendered (293) | ambiguous mean conf (11) |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct raw, letter logits | 92.6% / 0.079 | 81.4% / 0.192 | ~96% | 0.96 |
| circuit-vl-4b v2 | 96.4% / 0.036 | 89.5% / 0.105 | ~99% | 0.88 |

Photo cells: classify 97% (raw 79%), present 86% (raw 76%), negation 86%
(raw 89%). Open Images' verified labels are not exhaustive, which caps the
presence cells. Results: `results/vgrid2_circuit-vl-4b-v2.json`,
`results/vgrid2_qwen3vl4b_raw.json`.

## router-0.6b and router-1.7b (trained 2026-09-20, not published)

LiteLLM ships an auto-router that asks a System One model which tier of LLM
should answer a prompt, and published a benchmark of it against Jev with the
evidence archive attached: 80 authored cases, the rubric, and the recorded
wire requests. Replaying their exact request bodies against our models, only
the model name changed:

| system | tier match | cost per classification |
|---|---|---|
| Jev (their figure) | 228/240 = 95.00% | $0.0000321, their price |
| circuit-8b | 223/244 = 91.39% | $0.0000425, our GPU cost |
| circuit-1.7b | 201/244 = 82.38% | $0.0000201 |
| Claude Haiku 4.5 (their baseline) | 177/240 = 73.75% | $0.000827 |

circuit-8b lands 3.6 points behind Jev zero-shot, having never seen a routing
example. It is also more expensive per call than Jev's retail price, because
94% of a routing prompt is a rubric that never changes (376 characters of
instructions and 609 of tier criteria against 61 characters of actual
message), and our layout puts the state first, so that fixed text cannot be
prefix-cached.

The obvious answer was a small model trained for the task. That is
`s1proto/data/router_grid.py`: 2,874 items, tiers balanced, labels from two
places — prompts lifted from public benchmarks whose task fixes the tier
(GSM8K and LogiQA are REASONING, BoolQ and TriviaQA SIMPLE, MBPP MEDIUM), and
constructed items for the shapes no benchmark supplies (follow-ups, tool
output, long-but-easy padding, boundary items with soft labels). Tier names
and descriptions vary per item so the model has to read the rubric it is
given rather than memorise four names.

Qwen3-0.6B-Base, LoRA rank 16 plus the pointer head, 2 epochs, batch 8, 512
tokens, 18 minutes on an Apple laptop GPU. It reached **99.3% on our own eval
split and 47.5% on LiteLLM's 80 cases**, and every error is downward:

|  | SIMPLE | MEDIUM | COMPLEX | REASONING |
|---|---|---|---|---|
| SIMPLE | 61 | 0 | 0 | 0 |
| MEDIUM | 18 | 43 | 0 | 0 |
| COMPLEX | 48 | 10 | 3 | 0 |
| REASONING | 43 | 0 | 9 | 9 |

43 REASONING cases routed to SIMPLE, 48 COMPLEX to SIMPLE, and not one
request over-routed. Training loss reached 0.0000 halfway through the second
epoch, which is the whole story: the model learned to recognise the
generators, not to read criteria. circuit-8b's confusion on the same 244
requests is nearly diagonal.

The obvious next question was whether 0.6B was simply too small, so the same
data trained Qwen3-1.7B-Base under the same settings: 37 minutes, 98.95% on our
eval split, **and the identical 116/244 = 47.54% on theirs**. Not similar —
identical. The two models return the same tier on all 244 requests, and their
confusion matrices match cell for cell. Tripling the parameters changed
nothing, which is as clean a statement as this kind of experiment ever gives:
the data is the binding constraint, not capacity.

Reading a supplied rubric *is* the task, and it is the capability a 0.6B does
not have and an 8B already does without being taught. A smaller model trained
on synthetic routing data cannot serve operator-defined tiers, which was the
design constraint that made the idea worth trying. Not published. The
generator stays, because the negative result depends on it being reproducible
and the next attempt should start from real prompt distributions rather than
templates.

### v2: real prompts instead of templates (2026-09-20)

The next attempt did start from real prompt distributions. `router_grid.py` v2 draws
about 5,000 prompts from ten public sets, labels each by the set it came from (GSM8K
and LogiQA are REASONING, BoolQ and TriviaQA SIMPLE, and so on), and keeps synthetic
items to 14%. Qwen3-1.7B-Base, same head and settings, 60 minutes on the laptop GPU.

**93.4% on our eval split, 83/244 = 34.0% on LiteLLM's cases** — worse than v1, and
the confusion says why:

|  | SIMPLE | MEDIUM | COMPLEX | REASONING |
|---|---|---|---|---|
| SIMPLE | 16 | 39 | 6 | 0 |
| MEDIUM | 0 | 58 | 3 | 0 |
| COMPLEX | 0 | 55 | 6 | 0 |
| REASONING | 0 | 52 | 6 | 3 |

Everything it does not recognise goes to MEDIUM. A label that comes from which dataset
a prompt was drawn from teaches the model to recognise datasets; LiteLLM's authored
cases come from none of them. v1 memorised generators and v2 memorised sources, and
the gap between our split and theirs (99/48, then 93/34) did not close. Two attempts,
same conclusion: without per-prompt tier labels from someone who read the rubric, a
small model does not learn routing, and circuit-8b zero-shot at 91.4% remains the
answer. The final checkpoint is `runs/router-1.7b-v2-final`; `train_lora.py` keeps the
lowest-ECE checkpoint as "best", which here was step 100 at 26% accuracy, so the
last one was used.

Results: `scratchpad/replay_circuit-8b.json`, `replay_router-0.6b.json`,
`replay_router-1.7b.json`;
their archive is `jev-live-evidence-20260918.tar.gz` from docs.litellm.ai.

## Reproducibility as a measurable property (2026-09-20)

Batching concurrent requests into one forward pass is worth about 3x on
routing-shaped traffic: 3.3 answers a second at concurrency 16 becomes 10.1,
and median latency falls from 4.3 s to 1.6 s because the queue drains faster.
It is off by default anyway, and the reason is what it does to the answer.

A bf16 matmul reduces in a different order at a different batch size, so the
same prompt scored in a batch does not give the same number as scored alone.
Measured on circuit-1.7b with four **identical** prompts, no padding involved:

| | p(yes) |
|---|---|
| alone | 0.5050005437563114 |
| batched with four identical prompts | 0.4954083064618685 |
| batched with one short prompt | 0.4923591602016146 |
| batched with one long prompt | 0.4946115911018874 |

About 0.01, and that example crosses 0.5, which flips a noul. With batching on,
an answer depends on what other callers sent in the same instant. For an API
that offers a number someone may have to account for later, that is the wrong
trade; for a screening run over thousands of items on one machine, it is the
right one. `S1_BATCH_MAX=8` turns it on.

**The same effect is visible in Jev.** LiteLLM's benchmark archive records three
repeats of 80 identical inputs with the per-tier probabilities attached
(`attempts.jsonl`). Of the 80 cases, **46 varied between repeats** and 34 were
identical; where it varies the median spread is **0.0100** (mean 0.0207, max
0.0800). The chosen tier never changed, so it never showed up in their accuracy
figure. Independently, the ainergiz systematic-review write-up reports "we ran
the same input twice, scores moved by 0.009 on average" on a completely
different workload.

**Batch-invariant kernels fix half of it.** Thinking Machines' library
(`thinking-machines-lab/batch_invariant_ops`, MIT) substitutes four ATen ops —
`mm`, `addmm`, `_log_softmax`, `mean.dim` — through `torch.Library`, no model
changes. Measured on an L40S with circuit-1.7b (`deploy/batch_invariance_probe.py`):

| | alone vs batch of 5 identical | alone vs batch of mixed lengths | time |
|---|---|---|---|
| standard kernels | 1.93e-02 | 6.90e-03 | 62 ms |
| batch-invariant | **0.00e+00** | 2.58e-02 | 67 ms |

Batch size stops mattering entirely, and it costs about 5 ms rather than the
1.6x they report, because we prefill once and never decode. Sequence length
still matters, because attention is not one of the four ops they swap — their
attention work goes through vLLM's FlexAttention backend, which we do not use.
So reproducibility under batching needs either a batch-invariant attention path
or prompts bucketed by token length so every pass is uniform. The second is
probably cheaper for us and is untried.

Our batched drift is 0.0096 and their median is 0.0100. That is the same
phenomenon at the same size, and it reframes the cost comparison:

| | cost per classification | same input, twice |
|---|---|---|
| Jev | $0.0000321, their price | moves ~0.010 |
| circuit-8b, batching off | $0.0000425, our GPU cost | identical |
| circuit-8b, batching on | ~$0.0000137 | moves ~0.010 |

Batched we would be under half their price at the same reproducibility, and
unbatched we are dearer than them and exactly repeatable. The gap was never
really a cost gap; it is a choice about whether an answer has to come back the
same way twice, and it is worth stating as a property rather than leaving it
implicit. Where it matters most is a threshold: a probability within 0.01 of
the line can fall either side of it depending on who else was being served.

## The column race (2026-09-20, third-party benchmark, we lose)

`goodrahstar/jev-column-race` labels 1,000 real Google Play reviews with four
typed columns — sentiment (score), topic (choice), bug (noul), churn (score) —
and publishes Jev against Gemini 3.8 Flash. Twenty reviews go in one shared
state and each gets four questions, so a request carries eighty questions: the
exact shape our prefix caching exists for. Star ratings are kept out of the
state and used afterwards as the independent check on sentiment, which is the
part of this benchmark worth respecting — the ground truth is what the reviewer
themselves chose, not anyone's label.

Their request builder replayed against our endpoints, 200 of the 1,000 reviews:

| | sentiment vs the writer's own stars (Spearman) | cost per 1,000 | wall per 1,000 |
|---|---|---|---|
| Gemini 3.8 Flash (their figure) | 0.82 | $0.158 | 18.8 s |
| Jev (their figure) | 0.80 | $0.023 | 4.6 s |
| circuit-8b | 0.71 | $0.078 | ~144 s |
| circuit-1.7b | 0.57 | $0.038 | ~174 s |

We lose on quality and lose badly on speed — roughly 30x slower per review than
Jev, and dearer than them too. Unlike the routing benchmark, where circuit-8b
sits 3.6 points behind zero-shot, there is no reading of this where we are
close. Eighty questions over a shared state is a serving problem as much as a
model one: we run each question as its own prompt against a cached state, and
whatever Jev does, it is not that.

The run paid for itself anyway. The first attempt returned 500s: eighty
questions is eighty prompts of about 1,300 tokens, and scoring them in one pass
exhausts a 22 GB L4 with a CUDA out-of-memory. Any caller using the API the way
both published benchmarks use it would have hit it. Scoring is now chunked
(`S1_SCORE_CHUNK`, 16), so memory is bounded by the chunk rather than by how
many questions a caller asks.

Caveats on our numbers: 200 reviews rather than 1,000, and Spearman against
stars measures the sentiment column only. Topic, bug and churn have no ground
truth in this corpus, so agreement there would only be agreement with Jev.

Results: `scratchpad/race_circuit-8b.json`, `race_circuit-1.7b.json`.

## Deterministic batching, and what it costs (2026-09-20)

Batch-invariant kernels remove the batch-size half of the drift. The other half
is sequence length, because attention is not among the four ops they swap. So
the batcher now buckets by token length: a job joins a pass only if its prompts
match the lengths already in it. Measured on an L40S with circuit-1.7b, eight
concurrent requests of assorted lengths through the batcher itself:

| | same prompt, alone vs in traffic | passes for 8 requests | wall |
|---|---|---|---|
| uniform lengths + invariant kernels | **0.00e+00** | 9 (1.0 prompts each) | 1,382 ms |
| mixed lengths + invariant kernels | 6.47e-02 | 2 (4.5 prompts each) | 622 ms |

The guarantee holds exactly: batched with seven other requests, the answer is
bit-identical to the same prompt scored alone. The cost is that on traffic of
assorted lengths nothing batches — eight lengths, eight passes — so determinism
was bought by turning the feature off in all but name. Where lengths cluster,
which is the screening shape, it should pay; on a public endpoint taking
whatever arrives, it will not.

FlexAttention is not the way out. Switching the backend and keeping the
invariant kernels made the identical-length case *worse*, 1.37e-02 against
bit-identical on SDPA, because transformers does not expose the block sizes
their attention path needs (`get_batch_invariant_attention_block_size` returns
16x16 and nothing consumes it). A batch-invariant attention kernel we control
is the general fix and is not written.

Probes: `deploy/batch_invariance_probe.py`, `deploy/deterministic_batching_probe.py`.

## Qwen3.5 as the base (2026-09-21, not published)

The shipped circuits sit on Qwen3. Qwen3.8 has nothing under 27B, so the newest small
bases are Qwen3.5 2B and 9B. Same data, head, settings and checkpoint rule as the
publish run (`results/pipeline26.sh`), one H100, both at once. Accuracy / ECE / KL:

| set | circuit-1.7b | Qwen3.5 2B | circuit-8b | Qwen3.5 9B |
|---|---|---|---|---|
| grid, 2,125 | .969 / .013 / .069 | .964 / .018 / .068 | .980 / .007 / .029 | .970 / .009 / .030 |
| held-out public, 1,200 | .677 / .123 / 1.03 | .694 / .160 / 1.29 | .709 / .181 / 1.37 | .705 / .168 / 1.15 |
| water calls, 100 | .920 / .077 / .246 | .870 / .097 / .384 | .930 / .048 / .297 | .930 / .044 / .186 |
| DIY, 546 | .700 / .053 / .569 | .758 / .065 / .508 | .841 / .039 / .318 | **.886 / .036 / .165** |
| ms per item, batch 1, same card | 343 | 580 | 350 | 592 |

Mostly a wash, with one real difference: the 9B is 4.5 points better than circuit-8b on
the DIY set at half the KL, and the 2B is 5.9 points better than circuit-1.7b there.
That is the set furthest from the training distribution. Everywhere else the pairs are
within noise, and the 2B is worse on water calls and less calibrated on held-out data.

It costs latency and the caching. Qwen3.5 interleaves linear-attention layers, and
three things follow. Its backward pass returns NaN gradients on a left-padded batch
(each row is fine alone), so training runs rows unpadded and accumulates
(`--micro 1`). It compiles once per sequence length, about 1.4 s each, so lengths are
rounded up to 64 in training and serving would need the same. And its recurrent state
cannot be copied per question, so shared-prefix scoring — 2.8x on multi-question
requests — is off for it (`_kv_cache_only` in `scorer.py`). Forward passes on padded
batches are sound: 300 held-out items score the same at batch 1 and batch 16.
Per-item latency on the same card is about 1.7x the Qwen3 circuits; both were bound
by the host rather than the GPU, so treat the ratio as indicative.

Not worth switching the small model. The 9B's DIY result is worth a second look if
out-of-distribution accuracy becomes the thing to buy and the latency is affordable.
Weights: `runs/circuit35-2b`, `runs/circuit35-9b`. About 3.3 H100-hours.

## Datasets nobody prepared for (2026-09-21)

> **Correction, same day.** The KL columns in this section and the two after it must not be
> read across to Jev. Its API rounds probabilities to two decimals, a reported 0.00 meets
> our `eps = 1e-9`, and one such item against a 20% human vote adds 3.8 to the mean. Jev's
> "KL 2.00" on ChaosNLI is that, not miscalibration. Brier is bounded and tells the real
> story: ChaosNLI Jev .269, SemIf .254, Nimble .331, circuit-8b v1.0 .357, circuit-1.7b
> v1.0 .322, circuit-8b v1.1 **.177**, circuit-1.7b v1.1 .292. So "every open model is
> closer to the human vote than Jev" is false; only circuit-8b v1.1 is. KL between the
> open models, which report full precision, stands. Found via Haixun Wang's "What's Inside
> Jev?", which points out the reporting floor.

`hf_eval.jsonl` is held-out rows of datasets that also supply training rows.
`scripts/build_unseen_eval.py` builds the harder thing: 300 items each from four
datasets no circuit has seen in any split, plus 300 POPE image questions.
Accuracy / ECE / KL to the human reference:

| set | Jev | Nimble | circuit-1.7b | circuit-8b |
|---|---|---|---|---|
| BFCL: can any listed tool do this? | .813 / .069 / 0.40 | **.827 / .060 / 0.37** | .580 / .228 / 0.76 | .807 / .107 / 0.59 |
| HaluEval QA: does the passage support the answer? | **.910 / .029 / 0.24** | .840 / .085 / 0.37 | .730 / .052 / 0.53 | .720 / .143 / 0.60 |
| ChaosNLI: 100 annotators per item | **.600** / .254 / 2.00 | .563 / .315 / **0.87** | .523 / .242 / 0.98 | .560 / .314 / 0.96 |
| HWU64: 64 intents | **.800 / .087** / 1.29 | cannot (option cap) | .723 / .107 / **1.01** | .777 / .140 / 1.12 |

This is a loss, and a more honest picture than the held-out rows gave. Jev leads on
accuracy everywhere it competes; circuit-8b is level with it on tool relevance and
close on intents, and 19 points behind on groundedness, where the 1.7B and 8B score
the same, so it is the data and not the size. circuit-1.7b on tool relevance is
barely above a coin. Calibration also degrades off-distribution: ECE of .11 to .31
here against .01 to .05 on the grid.

Two things go the other way. On ChaosNLI every open model is about twice as close to
the distribution of human answers as Jev is (KL 0.87 to 0.98 against 2.00): Jev picks
the plurality label more often and is confident on items people split on. And the
pointer head takes 64 options where Nimble, the other open model, takes none of them.

POPE, object presence in COCO photos: circuit-vl-4b-v2 .923 / ECE .049 against the
untuned Qwen3-VL-4B at .913 / .072. Tuning on rendered documents and charts did not
cost anything on photographs, and did not buy much either.

What to train next is written in the table: groundedness and tool-call judgments,
neither of which has a family in the mix. Results: `results/unseen_*.json`,
`results/pope_*.json`.

### Closing the gap: groundedness and tool-call families (2026-09-21, not yet published)

`scripts/build_grounded_tools.py` adds 3,000 rows to the publish mix, all human-labelled:
VitaminC and SQuAD v2 for whether evidence supports a claim or an answer, and CLINC and
SNIPS utterances against hand-written tool specs for whether a tool fits. HaluEval and
BFCL contribute nothing and stay the test. Same recipe otherwise (`pipeline27.sh`).
Accuracy / ECE / KL on the unseen sets:

| | BFCL | HaluEval | ChaosNLI | HWU64 |
|---|---|---|---|---|
| Jev | .813 / .069 / 0.40 | **.910 / .029 / 0.24** | .600 / .254 / 2.00 | **.800** / .087 / 1.29 |
| circuit-1.7b | .580 / .228 / 0.76 | .730 / .052 / 0.53 | .523 / .242 / 0.98 | .723 / .107 / 1.01 |
| circuit-1.7b v1.1 | .817 / .112 / 0.54 | .767 / .126 / 0.49 | .587 / .238 / 0.70 | .780 / .065 / 0.96 |
| circuit-8b | .807 / .107 / 0.59 | .720 / .143 / 0.60 | .560 / .314 / 0.96 | .777 / .140 / 1.12 |
| circuit-8b v1.1 | **.857** / .099 / 0.45 | .840 / .048 / 0.34 | **.710 / .098 / 0.38** | .777 / **.071 / 0.97** |

Tool relevance moved the most: the 1.7B from a coin to level with Jev, the 8B past it.
Groundedness closed from 19 points behind to 7 at 8B; the 1.7B gained less (.73 to .77),
so here size does matter once the data exists. In distribution the 1.7B is unchanged or
better everywhere (grid .970, DIY .700 to .747, water .930); the 8B gives back a little
(grid .980 to .959, water .930 to .890, DIY .841 to .828).

**The checkpoint rule turned out to matter more than expected.** `train_lora.py` keeps the
lowest validation ECE, which for the 8B was step 1,200 of 4,446. The last checkpoint is
more accurate where the training data lives (grid .972, HaluEval .877, HWU64 .820) and
much worse calibrated where it does not: ChaosNLI falls from .710 / KL 0.38 to .577 / KL
1.11, and for the 1.7B to KL 3.11, worse than Jev. Training longer on one-hot labels buys
accuracy and spends calibration, and the ChaosNLI result above is mostly early stopping,
not the new rows. An earlier note here called the rule a bug after it kept a 26%-accurate
router checkpoint; the rule is right for a calibrated model and wrong only when accuracy
has not arrived yet. The fix is a floor, not a different metric.

These are v1.1, not v2: same base, head and recipe, more data. Published as tag `v1.1` on
`jbarney/circuit-1.7b` and `jbarney/circuit-8b` (2026-09-21; the 8B is the step-1,200
checkpoint, chosen for calibration off distribution over accuracy on it; a retrain on a
different GPU peaked late instead, so the best step is a property of a run, not of the
recipe); `v1.0` tags the original weights, and Modal and
`REPRODUCE.md` pin `v1.0` until the 8B is settled. Weights: `runs/circuit-{1.7b,8b}-v1.1`
(kept) and `-v1.1-final` (last). About 1.9 H100-hours.

## SemIf: what an untuned 4B does (2026-09-21)

SemIf (TheoLeeCJ/SemIf, MIT; LangSmith hosts it as `semif-qwen3.5-4b`) is not a trained
decision model. It asks stock `Qwen/Qwen3.5-4B` a lettered multiple-choice question and
reads the logits of the answer letters, one pass, nothing generated. That makes it the
control the rest of this document was missing: what do you get with no training at all?
`scripts/eval_semif.py`, MLX backend on the Mac. Accuracy / ECE / KL:

| set | SemIf | Jev | circuit-1.7b v1.1 | circuit-8b v1.1 |
|---|---|---|---|---|
| BFCL tool relevance | .813 / .118 / 0.52 | .813 / .069 / 0.40 | .817 / .112 / 0.54 | **.857** / .099 / 0.45 |
| HaluEval groundedness | .847 / .065 / 0.42 | **.910 / .029 / 0.24** | .767 / .126 / 0.49 | .840 / .048 / 0.34 |
| ChaosNLI | .623 / .205 / 0.61 | .600 / .254 / 2.00 | .587 / .238 / 0.70 | **.710 / .098 / 0.38** |
| HWU64, 64 intents | cannot ask | **.800** | .780 | .777 |
| grid, 2,125 | .868 / .020 / 0.35 | .947 / .014 / 0.28 | **.970 / .011 / 0.06** | .959 / .019 / 0.09 |
| water calls | .910 / .071 | **.980** / .015 | .930 / .055 | .890 / .051 |
| DIY, 546 | .802 / .056 | | .747 / .063 | **.828** / .038 |
| MNLI / SMS spam / civil toxicity | .79 / .92 / .79 | .88 / .96 / .82 | .82 / .97 / .87 | .86 / .98 / .86 |
| ms per item, same Mac | 250 to 690 | | 184 | 777 |

A recent 4B instruct model already does most of this. With no training it equals Jev on
tool relevance, equals our 8B on groundedness, beats our 1.7B on groundedness by 8 points
and on the DIY set by 5. What training buys is narrower than "accuracy": the grid (.97
against .87, and the gap is in classify, ordinal and temporal, the operations that need
a rubric read rather than a fact recalled), calibration off distribution (ECE .05 to .10
for the 8B against .07 to .21), and the things its construction rules out. Answers are
letters A to P, so more than 16 options cannot be asked at all: every HWU64 and CLINC
item, 30 DIY items and 69 grid items were unaskable and are scored as wrong above. Score
questions read as five unrelated letters, and ticket priority shows it (.25, ECE .52).

Nimble, which is this same base fine-tuned, scores .853 on the grid, slightly *below*
the untuned model. Whatever Nimble's training added, it was not this.

The obvious conclusion is the one Qwen3.5 already suggested twice: the base matters more
than anything done to it so far, and a circuit on Qwen3.5-4B would start from .85 on
groundedness rather than .72. The costs are the ones measured above in this document:
no shared-prefix caching, per-length compilation, about 1.7x the latency.

## Three probes of the hosted model (2026-09-22, overnight)

All against `jev-latest`, all reproducible with the scripts named; each costs cents.

**Same request, four times** (`scripts/probe_jev.py`, 300 choice items). Jev's top answer
differs between identical requests on **3.7%** of items, with a mean probability shift of
.016. Under six copies in flight at once it is 5.0% / .016, and one at a time also 5.0% /
.016 (`probe_jev_concurrency.py`, 120 items), so our own concurrency is not the cause;
whether it is batch composition on their side, with other customers' traffic, cannot be
told from outside. Our 1.7B on the Mac, same probe: 0.0% / .000. The permutation test's
8% for Jev therefore has a floor of about 4% that has nothing to do with option order.

**A second question in the same request.** Jev's answer to the first question moves by
2.0% flips / .015 when an unrelated noul question rides along, which is within its
repeat noise: the branches are isolated, as Hume's reconstruction says. Ours moves 1.3% /
.005 with zero repeat noise, which is the shared-prefix path batching the two tails
together (bf16 again); a real, small, fixable drift.

**Two decimals.** 100% of Jev's reported probabilities lie on 0.01 steps, and the true
label is reported as exactly 0.00 on 2% of items overall and 12% of 151-way ones. So a
downstream `p > 0` gate sees the truth as impossible one time in eight on a big list.

**Option count** (`probe_option_count.py`, 60 CLINC items, the same item at 4 to 151
options, truth always present):

| options | Jev acc / conf / p50 | circuit-1.7b v1.1 (Mac) acc / conf / p50 |
|---|---|---|
| 4 | .983 / .989 / 159 ms | .983 / .987 / 202 ms |
| 16 | .967 / .970 / 158 ms | .983 / .976 / 306 ms |
| 32 | .950 / .976 / 162 ms | .967 / .973 / 444 ms |
| 64 | .933 / .948 / 164 ms | .967 / .954 / 650 ms |
| 151 | .900 / .923 / 157 ms | .917 / .951 / 1,328 ms |

Accuracy declines smoothly for both, no step anywhere, and ours is level or ahead at
every size. Jev's latency is flat across a 38x range of option count while ours grows
6.6x, as a decoder that reads every option in context must. Flat is consistent with a
big card where 900 tokens of prefill hide under the network round trip, and also with
options not being read in context at all; the probe cannot tell those apart. Both models
grow overconfident with size (conf above p(truth) by .07 for Jev and .05 for ours at 151).

## Four more probes, both models (2026-09-22, overnight)

Same scripts run against `jev-latest` and against circuit-1.7b v1.1 on the Mac. Jev's
floor of 3.7% flips from repeating an identical request applies to every Jev number here.

**Does it read the rubric?** (`probe_rubric.py`) Each option's description is rotated onto
the next option's name, so the description-reader and the name-reader disagree. Water
calls, 100 items, share following the description / the name: Jev .95 / .02, ours .93 /
.01. With names replaced by `opt_1..opt_k` both hold (.95, .92); with descriptions removed
both fall (.87, .80). Both models read the criteria you write. When name and description
conflict Jev's confidence drops (.97 to .93 on water calls, .87 to .60 on the DIY tool
question): it notices. Ours drops less.

**Text that has nothing to do with the question** (`probe_distractor.py`, 356 items, a
paragraph about a library and a walking group). Flip rate with it appended / prepended:
Jev .051 / .090, ours .076 / .171. Prepended filler is worse for both, and much worse for
ours: mean probability shift .152 against Jev's .096. Ours loses no accuracy on average
(.680 to .688), so the flips are churn on near-ties, but a state that begins with
irrelevant material moves our numbers by a tenth.

**Where the evidence sits** (`probe_position.py`, the call transcript among five filler
paragraphs, about 900 tokens). Jev: .98 / .98 / .98 / .97 alone / start / middle / end.
Ours: .93 / .93 / .90 / .90. No lost-in-the-middle for Jev at this length; a 3-point dip
for ours once the transcript is not first.

**The image and audio circuits** (`eval_permutations.py --media`, own grid evals, 4 orders).
circuit-vl-4b: 3.6% flips on 140 items at 99% accuracy, all in classify and the
deliberately ambiguous cells. circuit-audio-7b: 11.7% on 240 items, 22% on count questions
and 75% on the ambiguous ones. Same affliction as the text models, same fix available: the
mask and positions in `s1proto/parallel.py` touch only the option spans, which sit after
the image or clip.

Taken together with the earlier probes: the hosted model is more robust than ours to
everything except option order, where after tonight's training neither has an edge on us,
and it reads a rubric as well as it is claimed to. Its weaknesses are nondeterminism, two
decimals, and confidence on items people disagree about. Ours are distractibility and
calibration off distribution, both of which are training data, not architecture.
