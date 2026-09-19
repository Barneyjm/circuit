"""Serve the circuit models on Modal: scale-to-zero GPU containers, one URL per model.

    uv run modal setup                                   # once, logs the CLI in
    S1_API_KEY=... uv run modal deploy deploy/modal_app.py

Each model is a class: the container downloads the adapter, head, and base
weights into a persistent volume on first boot, builds the same FastAPI app
`python -m s1proto` serves locally, and exposes it at
https://<workspace>--circuit-<name>.modal.run/v1/systemone. Idle containers
stop after two minutes; a cold start costs the base-model load time.
"""

from __future__ import annotations

import os

import modal

app = modal.App("circuit")
weights = modal.Volume.from_name("circuit-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8",
        "transformers>=5.17,<6",
        "peft>=0.21",
        "accelerate>=1.15",
        "huggingface_hub",
        "fastapi[standard]>=0.141",
        "pydantic>=2.13",
        "numpy",
        "decision-circuits",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("s1proto")
)
# Deploy-time environment becomes the container's: set S1_API_KEY to require it on every call.
secret = modal.Secret.from_dict({k: v for k in ("S1_API_KEY",) if (v := os.environ.get(k))})

MODELS = {
    "circuit-1.7b": {"repo": "jbarney/circuit-1.7b", "gpu": "L4"},
    "circuit-8b": {"repo": "jbarney/circuit-8b", "gpu": "L40S"},
}


def build(name: str):
    """Fetch the run from the Hub into the volume (adapter/, head.pt, config.json), then the base it names."""
    from huggingface_hub import snapshot_download

    run_dir = f"/vol/runs/{name}"
    snapshot_download(MODELS[name]["repo"], local_dir=run_dir)
    weights.commit()
    from s1proto.scorer import LoRAScorer
    from s1proto.service import create_app

    scorer = LoRAScorer(run_dir=run_dir)  # downloads the base into HF_HOME on the volume the first time
    weights.commit()
    return create_app(scorer)


@app.cls(image=image, gpu=MODELS["circuit-1.7b"]["gpu"], volumes={"/vol": weights}, secrets=[secret], scaledown_window=120, timeout=600)
@modal.concurrent(max_inputs=8)
class Circuit17B:
    @modal.enter()
    def load(self) -> None:
        self.web = build("circuit-1.7b")

    @modal.asgi_app(label="circuit-1-7b")
    def api(self):
        return self.web


@app.cls(image=image, gpu=MODELS["circuit-8b"]["gpu"], volumes={"/vol": weights}, secrets=[secret], scaledown_window=120, timeout=900)
@modal.concurrent(max_inputs=8)
class Circuit8B:
    @modal.enter()
    def load(self) -> None:
        self.web = build("circuit-8b")

    @modal.asgi_app(label="circuit-8b")
    def api(self):
        return self.web
