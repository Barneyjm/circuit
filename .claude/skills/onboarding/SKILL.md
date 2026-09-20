---
name: onboarding
description: Orientation for this repo (package name s1proto, GitHub name "circuit"). Load this FIRST in a new session before making changes, running training, touching GPU pods, or asking the user what's going on — it answers that. Triggers on "what is this repo", "get me oriented", "where do I start", "what's the current state", cold starts in general, or any request touching training/eval/data-generation/deployment you don't already have context for.
---

# Orientation: the circuit project

You're in `s1proto` (package name) / **circuit** (GitHub repo name, public: `Barneyjm/circuit`). This is the training and evaluation harness for **System One models**: small open-weights models that answer typed yes/no, multiple-choice, or leveled questions about a state with a calibrated probability distribution, in one forward pass, no text generation. They speak the same `POST /v1/systemone` contract as TypeSafe's commercial model "Jev."

**The sibling repo you also need to know about:** `~/Documents/code/decision-circuits` (PyPI: `decision-circuits`, GitHub: `Barneyjm/decision-circuits`). That's the *consumer* SDK — a small DSL (`Circuit`, `Q`, gates) that sends typed questions to any System One server and turns the probabilities into decisions in code. It also owns the public site and API gateway (see "What's live" below). If you're asked to change the SDK, add an integration, or touch the site/API, go there instead — it has its own `onboarding` skill.

## The core idea, in one paragraph

A base LLM (Qwen3, Qwen3-VL, Qwen2-Audio, whatever) gets a LoRA adapter on its language model plus a small **pointer readout head**: a query vector from the hidden state at a "decide" token, a key vector from the hidden state at each option's closing delimiter, scaled dot product, softmax over however many options there are. No option cap, order-invariant, and the head doesn't care what encoder (text/vision/audio) produced the hidden state it reads. Training data is a **grid**: judgment operations (extract, classify, compare, count, consistency, temporal, negation, ordinal, rule, mention, order, ...) crossed with state formats (string, JSON, nested, list, thread, document for text; receipt/chart/table/form/scene/photo for vision; call/list/numbers/sounds/speech/digits for audio). Every label is computed by the code that generated the item — never by another model — except where the item wraps a real dataset with real human labels (see "Real data sources" below). ~8% of items in each grid are deliberately built to be undecidable, with a flat/soft-0.5 reference label, to test whether the model knows when it doesn't know.

## Repo layout

```
s1proto/            the package: schema.py, template.py (prompt rendering + pointer layout), scorer.py (LoRAScorer/MultimodalScorer/HFScorer), service.py (FastAPI POST /v1/systemone), media.py (heads + multimodal plumbing, shared by trainer/service/evaluators), circuits.py (compat shim -> decision_circuits.gates), data/ (grid.py, vision_grid.py, audio_grid.py generators)
scripts/            train_lora.py (the trainer, all modalities), eval_set.py / eval_jev.py / eval_nimble.py / eval_vision.py / eval_audio.py (evaluators), build_*.py (dataset builders), grid_report.py, vision_probe.py / audio_probe.py (raw-base Phase-1 probes), voxcpm_bridge.py (subprocess bridge to a separate VoxCPM venv), remote_setup.sh / remote_watch.sh / remote_events.sh / remote_collect.sh (RunPod GPU lifecycle — see below)
data/               grid_*.jsonl, wide_*.jsonl, hf_eval.jsonl (real CC-licensed data), water_calls.jsonl (the original article's 100 calls), cmdiy_eval.jsonl (546 real production questions), vision/grid/, audio/grid/ (generated media, gitignored), audio/sources/ vision/sources/ (downloaded real-data sources: LibriSpeech, Common Voice, FSDD, Open Images — gitignored, re-fetched by the generator docstrings)
runs/               trained model directories: adapter/, head.pt, config.json (gitignored — pull from Hugging Face or a pod)
docs/               cold-eval.md (every eval result table, append-only), grid.md, wide-mix.md (license audit per data source), decision-circuits.md, phase1-3.md (early exploration notes)
deploy/modal_app.py serverless hosting (see "What's live")
results/            every eval as JSON, plus pipelineNN.sh / launchNN.sh scripts that were used for each GPU pod run (numbered, historical — the pattern to copy for a new run)
```

## What's live right now (check before assuming — these change)

- **Hugging Face**, all public under `jbarney/`: `circuit-1.7b`, `circuit-8b`, `circuit-vl-4b`, `circuit-audio-7b`. Each repo's README is the model card with current numbers — trust that over anything in this skill.
- **decisioncircuits.com**: the public site (owned by the `decision-circuits` repo), with a family section, a live scoreboard, and a "think you're better than the model" quiz that calls the real hosted models.
- **api.decisioncircuits.com**: a Cloudflare Worker gateway (in `decision-circuits/gateway/`) that issues free API keys and proxies to Modal.
- **Modal** (`deploy/modal_app.py` in *this* repo, app name `circuit`, workspace `james-73074`): one scale-to-zero GPU class per model, `max_containers=1` each as a hard spend ceiling. Deploy with `uv run --group deploy modal deploy deploy/modal_app.py` (needs `S1_API_KEY` env set from `.env` and `modal token set` already run once).
- **RunPod**: used for one-off training pods, torn down after each run. **Check `runpod.get_pods()` before assuming none are running** — if one is up and idle, terminate it (see rule below).

## Hard rules, not suggestions

- **No idle compute, ever.** A GPU pod that's finished its job and hasn't self-destructed is money burning for nothing. Every pipeline script (`results/pipelineNN.sh` + `launchNN.sh`) ends with a self-destruct timer; `scripts/remote_watch.sh` babysits a pod, collects weights every N minutes, and terminates it when the pipeline reports done or the box goes unreachable. Copy that pattern for any new pod run. When you're done checking on a run, verify `runpod.get_pods()` is empty.
- **Always collect weights before terminating.** `scripts/remote_collect.sh <host> <port>` rsyncs `runs/` back. Never terminate a pod you haven't collected from.
- **Every real (non-code-generated) data source needs a license check** before it goes in training or eval data — see `docs/wide-mix.md` for the format (source, license, ok/not-commercial/unclear). CC0, CC-BY, Apache 2.0, MIT are fine; anything "custom"/"research-only"/unclear gets flagged, not silently included.
- **No teacher-model outputs in training data.** Every label is either code-computed or a real human annotation from the source dataset. That's a stated design principle on the model cards and the site — don't quietly break it by using an LLM to generate soft labels.
- **`uv run <script>` for the main deps; `uv run --group audio` for anything touching Kokoro/VoxCPM/audio_grid; `uv run --group deploy` for `modal`.** The base `pyproject.toml` deps don't include those — a bare `uv run python -m s1proto.data.audio_grid` will `ModuleNotFoundError` on `kokoro`.
- **`.env` (gitignored) holds `WANDB_API_KEY`, `RUNPOD_API_KEY`, `S1_API_KEY`.** Never put these in code, commits, or scripts committed to the repo.
- **`git config core.hooksPath .githooks` is set** — a pre-commit hook runs `ruff format --check` + `ruff check` on staged `.py` files and blocks the commit if either fails. If it seems oddly strict or is blocking something it shouldn't, fix the code, don't bypass the hook.

## How to check "what's the current state" quickly

1. `git log --oneline -20` in both this repo and `decision-circuits` — commit messages here are written to be read later.
2. `docs/cold-eval.md` — every model's results, in order added, including ones that were trained and *not* published (with the reason).
3. `cat runs/<model>/config.json` if the weights are local — has the `best` checkpoint metrics and full training args.
4. `runpod.get_pods()` and `modal container list` (via `uv run --group deploy modal container list`) — nothing should be running unless a job is actively in flight.
5. The model cards on Hugging Face are the source of truth for published numbers; `docs/cold-eval.md` also carries unpublished/rejected runs (e.g. a 14B that trained fine but didn't beat the 8B).

## Adding a new modality or base model — the pattern to follow

Every modality (text/vision/audio) follows the same shape, factored into `s1proto/media.py` (heads: `PointerHead`/`SlotHead`; model plumbing: `load_base`, `hidden_states`, `head_logits`, `chat_text`, `encode`; media loading: `load_image`/`load_audio`/`split_media_state`). To add one:

1. Write a `data/<modality>_grid.py` generator mirroring `grid.py`'s shape: a `CELLS` dict of `(operation, format) -> generator function`, each returning a labeled item with ~8% ambiguous soft-labeled items, a `generate()` function, and a `if __name__` CLI writing `train.jsonl`/`eval.jsonl`.
2. Add the modality to `scripts/train_lora.py`'s `--modality` choices and `s1proto/media.py`'s `load_base`/`hidden_states`/`encode` if the base needs new plumbing (e.g. a tokenizer without the `<|box_start|>` pointer-delimiter tokens needs an entry in `template.py`'s `POINTER_TOKEN_SETS`).
3. Write `scripts/eval_<modality>.py` mirroring `eval_vision.py`/`eval_audio.py`: raw-base letter-logit baseline plus `--lora` scoring through the trained head.
4. Train small on the laptop first (`--epochs 1 --batch 2`, a few hundred items) to catch shape bugs before spending on a GPU pod.

For a genuinely new *base model* (not just modality) — e.g. Gemma, Phi-4, OLMo instead of Qwen — the only required changes are in `scorer.py`/`media.py`'s `load_base` (which `AutoModelFor...` class) and confirming the tokenizer has (or gets assigned, via `use_pointer_tokens`) three single-token pointer delimiters. Everything else — the grid data, the trainer, the head — is base-agnostic by design.
