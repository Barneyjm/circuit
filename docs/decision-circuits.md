# From one model to a decision-circuit family

Draft, 2026-09-18. Numbers marked TBD are filled from `results/` when
the Phase 2/3 runs finish.

## The idea, restated

Barney's "AI Decision Circuits" (Towards Data Science, May 2025)
borrowed error correction from electronics: put independent
categorizers, a schema validator, and a negative checker in a circuit,
and combine them with fixed logic so that undetected errors fall like a
product of independent failure rates. The result there was a system
that traded three points of overall accuracy for a high-confidence
tier at 92.5%, from components that were each around 91%.

The blocker in that design was the components. Each gate was a chat
LLM asked to produce a label, so (a) every gate cost a generation,
(b) agreement between two gates was the only confidence signal, and
(c) the "probability" that the error math needs was a number nobody
actually had; 0.1 × 0.1 assumed independence and calibration that the
components could not certify.

A System One model fixes (a) and (c) at the component level. One
prefill pass returns a calibrated distribution per question, so a gate
is a thresholded probability rather than a sampled label, and the
circuit's error math has real inputs. Independence between questions
is a property of the architecture (separate answer slots, no
conditioning on each other's answers), not a hope.

## Gates

A gate is a boolean or categorical output computed by code from one or
more calibrated distributions. The model supplies the numbers; the
gate is deterministic. First-order gates:

| Gate | Inputs | Output | Use |
| --- | --- | --- | --- |
| THRESHOLD(q, τ) | one Noul or one Choice option | bool | "act if P(refund_eligible) ≥ 0.9" |
| AND / OR | k Nouls | bool, plus P(all) = Π p_i under independence | "PII present AND not a public figure" |
| MAJORITY(k) | k Nouls or k Choices over one state | option, plus vote margin | Redundant phrasings of the same question |
| ARGMAX+CONF(q, c) | one Choice | option or `abstain` if confidence < c | Confidence-gated routing |
| VERIFY(q_answer, q_check) | a Choice plus a Noul asking "is that answer supported?" | option or `escalate` | The negative checker, as a second question over the same state |
| ORDER(q_score, cutpoints) | one Score | ordinal bucket | Priority tiers from an expected level |

Every gate has a defined behavior on low confidence: abstain, escalate,
or fall through to a default. That is what makes a circuit inspectable:
each item's path through the circuit is a list of (question, probability,
gate, outcome) tuples that a human can audit.

Error math with calibrated inputs. If P(pass) for a THRESHOLD gate is
calibrated, the expected false-accept rate at τ is simply the mass of
wrong items above τ, which the reliability diagram gives directly. For
an AND of k independent Nouls, false-accept ≈ Π (1 - p_i) only if the
questions are independent given the state; they are independent at the
model level, not necessarily at the world level, so the training set
should include correlated pairs and the eval should report the joint
calibration of the gate output, not only the marginals.

## Why "family"

Three axes fall out of the prototype and each argues for more than one
model:

1. **Size vs latency.** The 0.6B model answers a 4-question, 512-token
   request in 220 ms on a laptop and gets 84% on 4-way routing; the 8B
   takes 2.8 s and gets 87% with ECE 0.043 on yes/no. A circuit has
   cheap gates (is this spam? is a field empty?) and expensive gates
   (does this code have a bug?). Route each question to the smallest
   model whose held-out calibration on that family is under the gate's
   tolerance. This is the "cascade" pattern but with a principled
   trigger: the small model's *calibrated* uncertainty, not a heuristic.
2. **Domain.** Post-hoc calibration transfers imperfectly across task
   families (Phase 2 held-out test, TBD). A per-domain LoRA on a shared
   base is cheap (10M trainable params on the 0.6B, ~80M on the 8B) and
   the head is 256 outputs regardless of domain. A family is a base
   plus a set of adapters, each with its own reliability diagram.
3. **Cardinality.** The label-token trick caps at 26 options; the
   trained head lifts that to 255; retrieval over embeddings handles the
   rest. Different heads, same base.

## What a circuit model would add beyond N questions

Today a circuit is client code over the `/v1/systemone` API: ask k
questions in one request, compose in code. Two things are worth moving
into the model or the server:

- **Single-sequence multi-slot inference.** One prefill of the state
  with a block-diagonal attention mask over question segments, so a
  20-gate circuit costs one state prefill plus 20 short tails, not 20
  cached-prefix passes. The prefix cache in `HFScorer` is the
  laptop-grade version of this; the production version needs a custom
  mask in the serving stack.
- **Gate-aware training.** Train not only per-question calibration but
  the calibration of gate outputs: sample circuits (AND of two Nouls,
  VERIFY pairs, MAJORITY over paraphrases) during training and add a
  loss on the composed output against the composed reference. This is
  the step that makes the electronics analogy honest: the reliability
  of the composed circuit is measured and optimized, not derived under
  an independence assumption.

## Proposed spec for a circuit request

Backward compatible with the TypeSafe schema: `questions` stays, and an
optional `gates` map references question ids.

```json
{
  "state": {...},
  "model": "s1-8b",
  "questions": {
    "pii": {"type": "noul", "instructions": "Does the text contain PII?"},
    "public": {"type": "noul", "instructions": "Is the named person a public figure?"},
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {...}}
  },
  "gates": {
    "redact": {"op": "and", "inputs": ["pii", "not:public"], "threshold": 0.85, "on_uncertain": "escalate"},
    "route": {"op": "argmax", "input": "dept", "min_confidence": 0.6, "on_uncertain": "abstain"}
  }
}
```

Response adds `gates: {redact: {value: true, p: 0.91, path: [...]}, route: {value: "billing", confidence: 0.72}}`.
The server evaluates gates deterministically; the model only ever sees
questions. Circuits are therefore versionable, testable offline against
the eval set, and their false-accept rates are reportable per gate.

## Worked circuits (implemented; `scripts/circuits_demo.py`)

`gates` is now an optional block on `/v1/systemone` (`s1proto/circuits.py`).
Ops: `threshold`, `not`, `and`, `or`, `majority`, `argmax`, `verify`,
`order`. Inputs reference nouls, `choice:option`, `score:level`,
earlier boolean gates, or `gate:value` for categorical gates.
Uncertainty (a probability inside `tau ± band`, a confidence below the
floor, a split vote) propagates downstream and resolves by the gate's
`on_uncertain`: abstain, escalate, or a default. Every result carries
a trace. Run on the fine-tuned 1.7B (`results/circuits_demo_1.7B.txt`):

**Support triage** (route + tier + OR(angry, critical) → human).
"Your app deleted every invoice… audit Friday… fix this now or we
cancel": department split three ways at confidence 0.20, so `route`
abstained; urgency 2.88/3 → tier 3; P(critical)=0.90, P(angry)=0.93 →
`human` = true at 0.99. Exactly the item a person should see, and the
circuit says why. The SSO pricing question routed to sales at tier 0
with `human` = false at 0.05; the password reset routed to account at
0.96.

**PII redaction** (AND(pii, NOT business-only)). Shipping label with a
name, address and phone: redact at 0.87. Business contact block:
P(pii)=0.03, redact at 0.00. "Reset requested for user j.patel — see
ticket #9921": P(pii)=0.41, outside the band, redact = false at 0.36.
Arguable, and the trace shows the whole argument.

**Refund decision** (majority over three phrasings, a negative checker
on exclusions, and a code-owned date gate). The lesson case: purchase
July 1, request September 12, "never used it, unopened". All three
phrasings said *eligible* at 0.79 to 0.90. The model cannot subtract
dates. `within_window`, computed in code and passed in as a noul,
was 0.0, so `approve` = false at 0.00. The negative checker was the
weak link: on the genuinely eligible case it said "policy excludes
this" at 0.50, dragging `approve` to 0.44. A 1.7B model reading a
one-line policy for an explicit exclusion is not yet reliable; that
question needs its own family in training data, or a bigger model.

**Moderation** (decision, then corroboration by an independent "is it
clean?" question in both directions). Dishwasher question: allow at
0.96, clean 0.69, `allow_ok` true at 0.66. Insult: remove_harassment
at 0.96, clean 0.09, `remove_ok` true at 0.87. Spam: remove_spam at
0.96, `remove_ok` true at 0.84. The first version of this circuit
applied the removal checker to an allow decision and escalated the
dishwasher post; making the branches explicit (`decision:allow` as a
gate input) fixed it. Circuit bugs are visible in traces the same way
code bugs are.

Three things the demos establish:

1. **Abstention is the product.** In every circuit the interesting
   output was a gate that declined: `route` abstaining on the
   three-way split, `approve` failing on a checker it couldn't trust.
   A generator would have produced an answer in each case.
2. **Computation stays in code.** The date gate caught what three
   redundant model votes missed. Redundancy does not fix arithmetic;
   routing does.
3. **The independence assumption is printed, not hidden.** Every AND/OR
   trace says "under independence". Whether that holds is an empirical
   question per gate, and the human-labeled set is where to answer it.

## Results so far (details in phase2.md / phase3.md)

- Post-hoc: temperature is the only method that helps; debias and
  permutation averaging do nothing on the reference set. Raw 8B agrees
  with the teachers 72%, raw 14B 73%.
- LoRA + 256-way head, soft targets: 1.7B reaches 0.798 agreement /
  ECE 0.023 / KL 0.24; 8B (half the data, one epoch) 0.820 / 0.029 /
  0.215. Score questions go from 52% to 72% with the head.
- Transfer to five never-seen families: 0.70 (1.7B) and 0.69 (8B) vs
  0.68 raw. Two of the five need arithmetic and sit at chance for
  every model; the tuned models are least confident there, which is
  the behavior a gate wants.
- Not yet measured: joint calibration of a composed gate (AND over two
  Nouls) vs the composed reference. First circuit experiment to run.
