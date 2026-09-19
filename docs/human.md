# Human check on the reference set

> Updated after fixing two question defects the labeler found (see
> "Two data problems" below; `agent_next_action` rewritten and its
> `refund` option widened to `action`; pick_* candidates reshuffled).
> Numbers below are from the corrected set; the first-pass numbers are
> in git history.

2026-09-18. One labeler (the project owner), 191 answered items of a
198-item sample, 7 marked unsure and excluded. The sample is
**stratified across all 22 families and deliberately weighted toward
items where Jev and Gemini disagree** (95 of 191), so raw aggregates
understate teacher accuracy on the natural distribution; the two
subsets are reported separately. Labeler notes: a handful of early
items were answered before the option descriptions were read as the
definition of the question and may be revisited; the scheduling
overlap family was hard for a person too (5 of 9 unsure).

## Teachers vs human

| subset | n | human = Jev | human = Gemini | human = averaged ref |
|---|---|---|---|---|
| teachers agree | 101 | 0.871 | 0.871 | 0.871 |
| teachers disagree | 90 | 0.489 | 0.422 | 0.544 |
| all (disagreement-weighted sample) | 191 | 0.691 | 0.660 | 0.717 |

Reweighted to the eval set's natural mix (87.9% agreement items), the
averaged reference matches the human on about **83%** of items.

Tiebreaks on the 95 disputed items: human sided with Jev 44 times,
Gemini 38, neither 8. The two teachers are about equally right when
they disagree; averaging them is the right call and the human is the
only way to know which one to trust on a given family.

Teacher confidence vs human agreement:

| teacher | subset | mean stated confidence | hit rate vs human | ECE |
|---|---|---|---|---|
| Jev | agree | 0.924 | 0.871 | 0.079 |
| Gemini | agree | 0.936 | 0.871 | 0.098 |
| Jev | disagree | 0.674 | 0.489 | 0.212 |
| Gemini | disagree | 0.769 | 0.422 | 0.351 |

Both teachers are over-confident against a human, Gemini more so: on
disputed items Gemini says 77% and is right 42%; Jev says 67% and is
right 49%. Jev's confidence is the better-calibrated of the two, which
is its whole pitch, and this is the first independent evidence for it
here. Neither is calibrated enough on hard items to be used as truth
without a human tier.

## Models vs human (191 items)

| model | agreement with human | on disputed items | mean conf | ECE vs human |
|---|---|---|---|---|
| Qwen3-8B raw | 0.555 | 0.444 | 0.683 | 0.143 |
| Qwen3-14B raw | 0.607 | 0.478 | 0.717 | 0.136 |
| **1.7B tuned** | **0.649** | **0.533** | 0.772 | 0.136 |
| 8B tuned (bounded) | 0.602 | 0.444 | 0.785 | 0.194 |

The 1.7B fine-tune is the closest to the human of anything we built,
and on the disputed items it beats both teachers individually (0.533
vs Jev 0.489, Gemini 0.422) and matches their average (0.544). It
learned the consensus, and the consensus is what a person agrees with.

## By family (human vs averaged reference, n≈9 each; noisy)

Lowest: agent_next_action 0.22, content_policy 0.33, then
deadline_realistic / expense_category / same_place / log_severity at
0.56. Highest: review_positive 1.00, claim_supported / pick_total /
pick_meeting_time / email_folder 0.89.

`agent_next_action` was 0.22 on the first pass because the question was
ambiguous (did "answer" mean send the pending offer or write something
new?) and `refund` was too narrow for states where the accepted action
is a plan change. After rewriting the instructions and widening the
option to `action`, then re-querying both teachers, human agreement on
those 9 items is 7/9. `content_policy` remains genuinely subjective.

## Two data problems found by the labeler

1. **`pick_total` is a position leak.** The reference answer is the
   *last* candidate in 246 of 291 training items (91 of 100 eval),
   because candidates are listed in document order and totals come
   last on invoices. A model can score 85% by picking the last
   candidate. Fix: shuffle `candidates` in the state and remap the
   reference keys (order-invariant by construction); no re-labeling
   needed. `scripts/fix_candidate_order.py`, then retrain.
2. **`scheduling_conflict` is hard for people.** Interval overlap
   across time zones is arithmetic. Keep the family as a held-out
   "should abstain" test rather than something to train toward.

## What this changes

- The Phase 3 numbers are agreement with a teacher consensus that a
  human agrees with ~81% of the time on the natural distribution and
  ~54% on the hard half. ECE 0.02 against the reference is real; ECE
  against a person is ~0.15 for every model, and the teachers
  themselves are at 0.10 to 0.34. The calibration target should be a
  human-labeled set, not the teacher average, before any gate is
  trusted at a threshold.
- Next data spend: more human labels (this took one person a morning
  for 191), concentrated on disputed items and on the three weakest
  families; and a second labeler on a shared 50 to measure human
  agreement itself.
