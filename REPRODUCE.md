# Reproduce the unseen-dataset benchmark

Four public datasets that none of these models was trained on, 300 items each, scored
the same way for every model. One command:

```bash
git clone https://github.com/Barneyjm/circuit && cd circuit
scripts/reproduce_unseen.sh
```

It needs [uv](https://docs.astral.sh/uv/) and nothing else. It builds the test set from
Hugging Face (fixed seed, so you get the same 1,200 items we did), downloads the open
weights, scores them, and prints the table below. About ten minutes on an Apple-silicon
laptop or any GPU with 20 GB; `MODELS="circuit-1.7b" scripts/reproduce_unseen.sh` runs
in five on much less.

The table was measured on the `v1.0` weights, and the script pins them. `REVISION=v1.1
scripts/reproduce_unseen.sh` scores the current ones, which were trained after this
benchmark found the gaps (tool relevance .58 to .82 for the 1.7B), so they are a
response to it and not a blind result.

To include Jev, TypeSafe's hosted model, bring your own key. It takes about a minute
and costs a few cents:

```bash
TYPESAFE_API_KEY=... scripts/reproduce_unseen.sh
```

## What you should see

Accuracy / expected calibration error / Brier score. Lower is better for the last two.
Run on 2026-09-21.

| | tool relevance (BFCL) | groundedness (HaluEval QA) | human disagreement (ChaosNLI) | 64 intents (HWU64) |
|---|---|---|---|---|
| Jev (`jev-latest`) | .813 / .069 / .259 | **.910 / .029 / .140** | **.600 / .254 / .269** | **.800 / .087 / .276** |
| Bespoke-Nimble-9B | **.827 / .060 / .238** | .840 / .085 / .236 | .563 / .315 / .331 | does not run (26-option cap) |
| circuit-8b `v1.0` | .807 / .107 / .314 | .720 / .143 / .384 | .560 / .314 / .357 | .777 / .140 / .355 |
| circuit-1.7b `v1.0` | .580 / .228 / .539 | .730 / .052 / .352 | .523 / .242 / .322 | .723 / .107 / .381 |

Jev is the strongest model here. Nimble edges it on tool relevance; nothing open is close
on groundedness. Its weakest set is ChaosNLI, where it agrees with the majority of 100
annotators 60% of the time and its calibration error is three to nine times what it is
elsewhere, but it is still the best of these four there too.

**A correction.** An earlier version of this page reported KL divergence and said Jev was
twice as far from the annotators' votes as any open model. That was an artifact. Jev's
API reports probabilities to two decimals, so an option it gives 0.4% comes back as 0.00,
and KL against a human vote of 20% on that option is then enormous. Brier score is
bounded and does not have the problem; on it Jev leads. The script still prints KL for
the open models, where it is meaningful, and should not be read across to Jev.

The open models are deterministic on fixed hardware; expect the third decimal to move
between a Mac and a CUDA card. Jev is a hosted model that can change under its alias.
With 300 items a set, a gap of two or three points of accuracy is noise.

## What each set asks

- **BFCL**, Berkeley Function-Calling Leaderboard, live relevance and irrelevance
  splits: a user request and a tool description. Can the tool do it?
- **HaluEval QA**: a passage, a question and an answer. Does the passage support it?
- **ChaosNLI** (MNLI part): premise and hypothesis, with 100 annotators' labels. The
  reference is their distribution, not a single right answer.
- **HWU64**: a request to a home assistant and 64 intents to choose from.

Sources, licences and why each is evaluation-only are in `docs/wide-mix.md`. The built
files are not committed, because ChaosNLI is non-commercial; the script rebuilds them.

## Pieces, if you want them separately

| step | command |
|---|---|
| build the test set | `uv run python scripts/build_unseen_eval.py --n 300` |
| score an open circuit | `uv run python scripts/eval_set.py lora:runs/hub/circuit-8b data/unseen_eval.jsonl --out out.json` |
| score Jev | `uv run python scripts/eval_jev.py data/unseen_eval.jsonl --out out.json` |
| score Nimble | `uv run python scripts/eval_nimble.py data/unseen_eval.jsonl --nimble <dir> --out out.json` |

Our own result files are in `results/unseen_*.json`, and the write-up, including where
our models lose, is the section "Datasets nobody prepared for" in `docs/cold-eval.md`.
Weights: `jbarney/circuit-1.7b` and `jbarney/circuit-8b` on Hugging Face, tags `v1.0` and `v1.1`.
