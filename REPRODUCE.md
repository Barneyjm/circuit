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

To include Jev, TypeSafe's hosted model, bring your own key. It takes about a minute
and costs a few cents:

```bash
TYPESAFE_API_KEY=... scripts/reproduce_unseen.sh
```

## What you should see

Accuracy / expected calibration error / KL divergence to the human reference. Lower is
better for the last two. Run on 2026-09-21.

| | tool relevance (BFCL) | groundedness (HaluEval QA) | human disagreement (ChaosNLI) | 64 intents (HWU64) |
|---|---|---|---|---|
| Jev (`jev-latest`) | .813 / .069 / 0.40 | **.910 / .029 / 0.24** | .600 / .254 / 2.00 | **.800 / .087** / 1.29 |
| Bespoke-Nimble-9B | **.827 / .060 / 0.37** | .840 / .085 / 0.37 | .563 / .315 / 0.87 | does not run |
| circuit-8b | .807 / .107 / 0.59 | .720 / .143 / 0.60 | .560 / .314 / 0.96 | .777 / .140 / 1.12 |
| circuit-1.7b | .580 / .228 / 0.76 | .730 / .052 / 0.53 | .523 / .242 / 0.98 | .723 / .107 / **1.01** |

The open models are deterministic on fixed hardware; expect the third decimal to move
between a Mac and a CUDA card. Jev is a hosted model that can change under its alias.
With 300 items a set, a gap of two or three points of accuracy is noise. The ChaosNLI
KL column is not: its reference is how 100 people actually voted on each item, and the
hosted model is about twice as far from that as any open one.

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
Weights: `jbarney/circuit-1.7b` and `jbarney/circuit-8b` on Hugging Face.
