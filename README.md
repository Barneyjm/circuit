# circuit

Open-weights **System One** models and the harness that trains and
measures them. A System One model answers typed questions about a state
with calibrated probability distributions in one forward pass, no text
generation. These are the models behind
[decision-circuits](https://decisioncircuits.com); they speak TypeSafe's
`POST /v1/systemone` contract, so anything written for Jev runs against
them with a URL change.

| model | base | status |
|---|---|---|
| [circuit-1.7b](https://huggingface.co/jbarney/circuit-1.7b) | Qwen3-1.7B-Base | released |
| [circuit-8b](https://huggingface.co/jbarney/circuit-8b) | Qwen3-8B-Base | released |
| [circuit-vl-4b](https://huggingface.co/jbarney/circuit-vl-4b) | Qwen3-VL-4B-Instruct | released (images, video frames) |
| [circuit-audio-7b](https://huggingface.co/jbarney/circuit-audio-7b) | Qwen2-Audio-7B-Instruct | released (speech, sound) |

How they're built: LoRA on the language model plus a **pointer readout
head**. Each option is wrapped in delimiter tokens and the sequence ends
with a decide token; the head scores every option's closing delimiter
against the decide token and applies softmax. No option cap, no
positional bias, and the probabilities are trained with cross-entropy on
outcome labels, so calibration is learned. Training data is labeled by
code or by humans, never by another model; every source is permissive.

## Results

Same items, same labels, every model we can run. Accuracy / ECE.

| cold eval, 1,200 human-labeled items | MultiNLI | SMS spam | toxicity | CLINC 151-way |
|---|---|---|---|---|
| Jev (TypeSafe) | 88% / 0.04 | 96% / 0.05 | 82% / 0.06 | 90% / 0.05 |
| Bespoke-Nimble-9B | 84% / 0.09 | 91% / 0.06 | 86% / 0.08 | n/a (26-option cap) |
| kev-0.5b | 46% / 0.28 | 50% / 0.30 | 62% / 0.16 | 62% / 0.17 |
| circuit-1.7b | 81% / 0.09 | 98% / 0.02 | 90% / 0.16 | 86% / 0.06 |
| circuit-8b | 86% / 0.08 | 98% / 0.02 | 93% / 0.14 | 95% / 0.03 |

Out of distribution for everyone: the 100 water-utility calls from
[the article](https://towardsdatascience.com/attaining-llm-certainty-with-ai-decision-circuits/),
Jev 98%, Nimble 93%, circuit-8b 93%, circuit-1.7b 92%, kev 80%. Full tables and the
generalization grid (9 judgment operations x 6 state formats, labels
computed by code) in [docs/cold-eval.md](docs/cold-eval.md) and
[docs/grid.md](docs/grid.md).

Vision: on the rendered vision grid (receipts, charts, tables, forms,
scenes; 300 held-out items) circuit-vl-4b scores 98.3% / ECE 0.018 against
the raw base's 96.0% / 0.041 by letter logits. Seven minutes of training.

Audio: on the synthesized audio grid (support calls, spoken lists and
numbers, beeps and noise; 279 decidable held-out clips) circuit-audio-7b
scores 93.5% / ECE 0.072 against the raw base's 72.0% / 0.188. Trained on
a laptop in 35 minutes.

## Run a model

```bash
uv sync
S1_MODEL=lora:runs/circuit-1.7b uv run python -m s1proto     # serves POST /v1/systemone on :8901
```

Get the weights from the Hub into `runs/circuit-1.7b/` (`adapter/`,
`head.pt`, `config.json`). Then:

```bash
curl -s localhost:8901/v1/systemone -H 'Authorization: Bearer x' -H 'Content-Type: application/json' -d '{
  "state": "Card charged twice, refund NOW. My card ends in 4412.",
  "questions": {
    "pii":  {"type": "noul",   "instructions": "Does the message contain personal information?"},
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "technical": null, "other": null}}
  }}'
```

Or with [decision-circuits](https://pypi.org/project/decision-circuits/):
`SystemOne("http://localhost:8901/v1/systemone", api_key="x")`.

## Serve it serverless

`deploy/modal_app.py` runs the same service on [Modal](https://modal.com):
one scale-to-zero GPU container per model, weights pulled from the Hub
into a volume on first boot, idle containers stopped after two minutes.

```bash
uv run --group deploy modal setup                                        # once
S1_API_KEY=... uv run --group deploy modal deploy deploy/modal_app.py    # the key is required as the bearer token
```

That gives `https://<workspace>--circuit-1-7b.modal.run/v1/systemone` (L4)
and `...--circuit-8b.modal.run/v1/systemone` (L40S). Cold start is about
75 s for the 1.7B; a warm single-question call round-trips in 0.4 s.

## Train one

```bash
uv run python scripts/build_grid.py                       # 9 x 6 grid, code-labeled
uv run python scripts/build_wide_mix.py --commercial      # public tasks, permissive licenses only
uv run python scripts/build_hf_eval.py --split train      # CC-licensed real data
uv run python scripts/train_lora.py Qwen/Qwen3-1.7B-Base data/publish_train.jsonl \
    --out runs/circuit-1.7b --head pointer --epochs 2 --batch 4 --max-length 1024 --grad-checkpoint --wandb s1proto
uv run python scripts/eval_set.py lora:runs/circuit-1.7b data/hf_eval.jsonl --out results/hf_circuit-1.7b.json
uv run python scripts/grid_report.py results/grid_*.json
```

`--modality vision` trains the same head on a Qwen3-VL base with the
vision grid (`python -m s1proto.data.vision_grid`); `--modality audio` on
Qwen2-Audio with the audio grid (`uv run --group audio python -m s1proto.data.audio_grid`;
Kokoro voices plus LibriSpeech and spoken-digit recordings, see the module docstring). `--load-4bit` for
QLoRA on small cards. `scripts/remote_setup.sh`, `remote_watch.sh`, and
`remote_events.sh` run all of this on a rented GPU and pull the weights
back.

## Layout

```
s1proto/            the package: schema, template (letters and pointer layouts), scorers, service, data generators
scripts/            build_*, train_lora, eval_set, eval_jev, eval_nimble, eval_vision, audio_probe, vision_probe, grid_report
data/               grid, wide mix, CC-licensed real data, the article's calls, Cole's bench; rendered images are regenerated
docs/               phase1..3, human labels, cold eval, grid, wide-mix licenses, decision circuits
results/            every eval as JSON; pipeline scripts
```

The package is still called `s1proto` inside; the rename to `circuit`
is on the list.

## License

Apache 2.0. Model weights carry the base model's license (Apache 2.0 for
Qwen3). Data sources and their licenses are listed in
[docs/wide-mix.md](docs/wide-mix.md).
