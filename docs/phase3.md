# Phase 3: LoRA fine-tune with a proper scoring rule

2026-09-18. Trainer: `scripts/train_lora.py`. Data: `data/train.jsonl`
(5,100 items, 17 families, Jev+Gemini averaged references; 10% held
back per family for validation). Eval: `data/eval.jsonl` (2,199 items,
22 families; the 5 held-out families never appear in training).

## Setup

- LoRA rank 16 on q/k/v/o/gate/up/down (1.7B: 17.4M trainable params,
  1.0%), plus a fresh linear head from the answer-slot hidden state to
  256 outputs, masked to the question's N options. Label letters are no
  longer read; the head replaces them.
- Loss: cross-entropy with soft targets = the reference distribution.
  Option order shuffled per example for Choice.
- AdamW, lr 1e-4 (LoRA) / 1e-3 (head), 30-step warmup, linear decay,
  batch 8, 2 epochs = 1,148 steps, early stopping on validation ECE.
- 10.3 s/step on the M2 Max (MPS); 217 min total for the 1.7B.

## 1.7B result

Validation (training families, 510 items): ECE 0.106 → 0.045, argmax
agreement 0.71 → 0.84, KL 0.44 → 0.21 over 800 steps; later steps
reached 0.85 agreement at slightly worse ECE and were not kept.

Full reference set, all 2,199 items, vs the raw base models:

| model | acc | ECE | KL | Brier | sel@90 | agree Jev |
|---|---|---|---|---|---|---|
| Qwen3-8B-Base, raw | 0.720 | 0.039 | 0.432 | 0.245 | 0.745 | 0.718 |
| Qwen3-14B-Base, raw | 0.731 | 0.033 | 0.366 | 0.209 | 0.772 | 0.750 |
| **Qwen3-1.7B + LoRA + head** | **0.798** | **0.023** | **0.238** | **0.139** | **0.835** | **0.792** |

By type:

| type | 8B raw acc / ECE | 14B raw | 1.7B tuned |
|---|---|---|---|
| Noul (799) | 0.766 / 0.029 | 0.780 / 0.032 | 0.772 / 0.037 |
| Choice (1000) | 0.765 / 0.085 | 0.778 / 0.052 | 0.850 / 0.024 |
| Score (400) | 0.517 / 0.141 | 0.517 / 0.139 | **0.723 / 0.073** |

Transfer, n-weighted over families:

| model | training-family acc / ECE | held-out acc / ECE / KL |
|---|---|---|
| 8B raw | 0.732 / 0.139 | 0.681 / 0.170 / 0.462 |
| 14B raw | 0.752 / 0.131 | 0.659 / 0.160 / 0.469 |
| 1.7B tuned | 0.827 / 0.085 | 0.701 / 0.101 / 0.365 |

Held-out families one by one (acc / ECE):

| family | 8B raw | 14B raw | 1.7B tuned |
|---|---|---|---|
| expense_category | 0.86 / 0.12 | 0.73 / 0.14 | 0.79 / 0.08 |
| deadline_realistic (Score) | 0.44 / 0.25 | 0.38 / 0.26 | 0.58 / 0.11 |
| claim_supported | 0.83 / 0.07 | 0.85 / 0.05 | 0.84 / 0.06 |
| log_severity | 0.78 / 0.07 | 0.83 / 0.09 | 0.84 / 0.05 |
| scheduling_conflict | 0.50 / 0.33 | 0.51 / 0.26 | 0.46 / 0.20 |

## Read

- **The head fixes Score.** 52% → 72% argmax agreement and KL 0.68 →
  0.26 on ordinal questions, which neither 8B nor 14B could move with
  letter-token scoring. Ordinality is learnable from soft targets in a
  few hundred steps.
- **Transfer is real but smaller.** On families never seen in training
  the 1.7B is 2 points above the raw 8B on agreement and a third lower
  on ECE and KL; on trained families it is 10 points above. The
  calibration transfers better than the accuracy: held-out ECE 0.10 vs
  0.17. That is the useful half for a gate.
- **Two held-out families need computation, not judgment.**
  `scheduling_conflict` (interval overlap across time zones) sits at
  chance for every model, and `deadline_realistic` needs arithmetic on
  hours, days and team size. Single-pass scoring is the wrong tool for
  these; a circuit would route them to code or to a reasoning model.
  Worth noting the tuned model is *less* confident on them (ECE 0.20
  vs 0.33), which is the right failure.
- **Noul did not improve** over the raw 14B (0.77 vs 0.78). The raw
  base models were already decent at yes/no, and the head's gain went
  to Choice and Score. More Noul data or a Noul-specific temperature
  would likely close it.
- **ECE 0.023 overall is below the Phase 3 target (0.05)** and every
  type is under 0.08. Against the *human* subset this is still
  unverified; the labeling bench is in progress.

## 8B, bounded run

1 epoch over 2,400 items (543 steps, batch 4, max length 512, gradient
checkpointing, lm_head skipped), 102 min. Best validation at step 500:
ECE 0.043, agreement 0.870, KL 0.139. The first attempt at batch 8 with
full-vocab logits swapped the machine to a standstill; reading the slot
hidden state straight from the decoder was the fix.

Full reference set:

| model | acc | ECE | KL | Brier | sel@90 | agree Jev |
|---|---|---|---|---|---|---|
| 1.7B tuned (2 epochs, 5,100 items) | 0.798 | 0.023 | 0.238 | 0.139 | 0.835 | 0.792 |
| **8B tuned (1 epoch, 2,400 items)** | **0.820** | 0.029 | **0.215** | **0.117** | **0.868** | **0.820** |

By type (acc / ECE): Noul 0.825 / 0.038 (1.7B: 0.772 / 0.037), Choice
0.871 / 0.034 (0.850 / 0.024), Score 0.685 / 0.062 (0.723 / 0.073).

Transfer:

| model | training-family acc / ECE | held-out acc / ECE / KL |
|---|---|---|
| 1.7B tuned | 0.827 / 0.085 | **0.701 / 0.101 / 0.365** |
| 8B tuned | **0.859 / 0.075** | 0.687 / 0.170 / 0.390 |

Held-out by family (acc / ECE), 1.7B → 8B: expense_category 0.79 → 0.86,
claim_supported 0.84 → 0.94, log_severity 0.84 → 0.82,
deadline_realistic 0.58 → 0.36, scheduling_conflict 0.46 → 0.46.

Read:

- The bigger base wins in-family on every type except Score, and on
  three of five held-out families, on half the data and half the steps.
  Noul finally moved (0.77 → 0.83).
- The held-out aggregate is *worse* than the 1.7B, entirely because of
  `deadline_realistic`, where the 8B became confidently wrong (ECE
  0.29). That family needs arithmetic; the 8B learned a stronger prior
  from the ordinal families it did see and applies it where it doesn't
  fit. Same lesson as before: route computation to code, and a circuit
  should treat a high-ECE family as a signal to do so.
- Score lagging the 1.7B is most likely the training budget (400
  Score items seen once vs twice); worth the extra epoch before
  concluding anything about scale.
- Net: for this data size the two models are close, and the small one
  is the better deal per dollar. The next lever is more reference data,
  particularly Score and reasoning-shaped families, not a bigger base.
