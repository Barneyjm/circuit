# Phase 1: prefill-only logit scoring

2026-09-18. Built on an Apple M2 Max (64 GB, no discrete GPU); torch 2.14 on MPS, transformers 5.17.

## What exists

- `POST /v1/systemone` in TypeSafe's exact request/response format. The official `typesafe_sdk` 0.6.0 drives it with only `base_url` changed.
- One prompt per question: state, blank line, lead, instructions, `A.`/`B.`/`C.` options, `Answer:`. The scorer reads next-token logits at the slot, keeps the 26 label tokens, softmaxes over the first N. No sampling, no generation; `output_tokens` is 0.
- Shared-prefix KV cache: the state is prefilled once per request and each question's tail runs against the cached KV. Same argmax as the independent path, probabilities within bf16 noise (max |Δp| 0.03 observed).
- Confidence = `1 - H(p)/log N`. Score = expected level index. Noul = P(yes).
- Backends: `HFScorer` (torch/transformers, any device) and `FakeScorer` (deterministic, model-free).

## Acceptance

| Criterion (handoff) | Status | Note |
| --- | --- | --- |
| TypeSafe wrapper runs against our endpoint with only the base URL changed | **Pass**, with a substitution | `system-one-adapter-python` is a wrapper that makes chat LLMs answer System One questions; it has no endpoint to point at. The official `typesafe_sdk` with `base_url=` is the like-for-like test and passes against both the fake and the real scorer (`tests/test_sdk_acceptance.py`). |
| p50 < 100 ms, p99 < 300 ms at 512 tokens, 4 questions | **Not on this hardware** | Table below. Target was written for one H100 with vLLM; this is a laptop on MPS. The structural wins (prefix cache 2.2x, fp16 1.6x) carry over. |
| Always in schema; fuzz 10k random requests, zero parse failures | **Pass** | Hypothesis, 10,000 examples across all three types, nested JSON state/instructions, 2..26 options: 0 failures, 104 s. |
| Accuracy on TypeSafe's four workflow evals recorded as baseline | **Blocked** | evals.typesafe.ai publishes result charts only; no data or harness is downloadable. Substituted two public labeled tasks (below) as the Phase 1 baseline. |

## Latency (512-token state, 4 questions, one batch)

| Model | dtype | prefix cache | batch tokens | p50 ms | p99 ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B-Base | bf16 | off | 2209 | 775 | 777 |
| Qwen3-0.6B-Base | bf16 | on | 2209 | 348 | 379 |
| Qwen3-0.6B-Base | fp16 | on | 2209 | 220 | 222 |
| Qwen3-1.7B-Base | bf16 | on | 2209 | 746 | 750 |
| Qwen3-8B-Base | bf16 | off | 2209 | 7379 | 7448 |
| Qwen3-8B-Base | bf16 | on | 2209 | 2762 | 2767 |
| Qwen3-14B-Base | bf16 | on | 2209 | 4891 | 4898 |

MPS prefill throughput is roughly 3k tokens/s for the 0.6B model and 0.8k for the 8B. An H100 running vLLM prefill on an 8B model is 30 to 60x that; the handoff's 25 to 40 ms prefill estimate is consistent with these token counts. The prefix cache is the part that matters on any hardware: it turns the cost from `n_questions × state` into `state + Σ tails`.

## Calibration baseline (raw logits, T = 1, n = 200 per task, seed 7)

Two public labeled tasks, no prompt tuning:

- **SST-2** (Noul): "Is this movie review positive?" with true/false criteria.
- **AG News** (Choice, 4-way): world / sports / business / sci_tech, each with a one-line description.

| Model | SST-2 acc | SST-2 ECE | P(yes) predicted vs true | AG acc | AG ECE | Position-0 win rate | Argmax flips on option reversal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B-Base | 73.0% | 0.115 | 0.77 vs 0.51 | 84.0% | 0.068 | 32% | 10.5% |
| Qwen3-1.7B-Base | 69.0% | 0.154 | 0.81 vs 0.51 | 83.0% | 0.058 | 33% | 4.5% |
| Qwen3-8B-Base | 91.5% | 0.043 | 0.55 vs 0.51 | 87.0% | 0.075 | 32.5% | 9.5% |
| Qwen3-14B-Base | 92.0% | 0.033 | 0.56 vs 0.51 | 88.5% | 0.062 | 29% | 6.5% |

Read:

- The 8B base model is already under the Phase 3 ECE target (0.05) on the binary task with no calibration at all, and the yes-bias that dominates the small models is gone.
- On the 4-way task all three sizes over-pick the first option (true rate for that label is 27%) and about one item in ten changes its answer when the options are reversed. This is the label/order bias the handoff predicted; it does not shrink with scale.
- Temperature sweep from stored logits, 8B: SST-2 is flat around T = 0.7 to 1.0 (ECE 0.040 to 0.044). AG News wants *sharpening*: ECE 0.075 at T = 1.0, 0.054 at T = 0.5, accuracy unchanged. The base model is underconfident on the multi-way task, not overconfident.
- 0.6B → 1.7B accuracy drop on SST-2 is within the n = 200 standard error (about 3 points) and is driven by a larger yes-prior, not less knowledge.
- 14B continues the trend: better on both tasks, less position bias, and sharpening to T = 0.7 takes AG News ECE from 0.062 to 0.032.
- Content-free debiasing (Zhao et al. 2021 style, prior measured on "N/A" / "" / "[MASK]") did **not** help at 8B or 14B: SST-2 ECE moved 0.043 → 0.059 (8B) and 0.033 → 0.031 (14B); AG News got slightly worse on both, and the order-flip rate rose. The label prior on an empty state is not the prior the model applies on real states at this scale. Permutation averaging is the remaining post-hoc lever for order bias.

## Next: Phase 2 decision

Go. The prototype already meets the Phase 2 exit bar on the binary task with the 8B model (ECE 0.043 < 0.10) and is close on the 4-way task; the remaining error is structural (position bias) rather than model capacity, and the two cheapest Phase 2 methods target it directly:

1. Content-free label debiasing (implemented as `--debias`, results appended below when the run finishes).
2. Per-type temperature, fitted on a validation split rather than eyeballed from the sweep.

Then build the 2,000-item eval set. The two-task check here is a smoke test, not the eval set; it has hard labels, not reference distributions, so it measures ECE against accuracy rather than KL to a reference.

Open items carried forward:

- A GPU box for the latency criterion. Everything else in Phase 1 is done on the laptop.
- Qwen3-14B-Base fits in 64 GB (bf16, ~30 GB) and is being evaluated; Qwen3.5-9B-Base is the newer-generation candidate at the same cost as the 8B.
- Licensing check on Qwen (Apache 2.0 for Qwen3; confirm for Qwen3.5) before anything ships.
