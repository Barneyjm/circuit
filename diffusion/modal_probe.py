"""Is a bidirectional (masked-diffusion) base any better for open-taxonomy choice than a causal one?

    uv run --with modal modal run diffusion/modal_probe.py

Same starting weights both ways: Qwen3-0.6B, and Qwen3-0.6B converted to masked diffusion
(dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1, Apache-2.0). Each is frozen and gets a fresh pointer
head (train_lora --freeze-adapter), on the same rows, then the open-taxonomy and tagger evals.
The causal one encodes options side by side (--parallel-options), as the served models do; the
bidirectional one needs no such trick. One L4 per arm, in parallel; nothing is served.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ["v22_head_mix_train.jsonl", "open_tax_eval.jsonl", "open_tax_tagger_eval.jsonl"]
weights = modal.Volume.from_name("circuit-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8", "transformers>=5.17,<6", "peft>=0.21", "accelerate>=1.15", "huggingface_hub", "pydantic>=2.13", "numpy", "decision-circuits>=0.5.4"
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("s1proto")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)
for f in DATA:
    image = image.add_local_file(ROOT / "data" / f, f"/root/data/{f}")
app = modal.App("circuit-diffusion-probe", image=image)

ARMS = {
    "causal": ("Qwen/Qwen3-0.6B", ["--parallel-options"]),
    "diffusion": ("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", ["--masked"]),
}


@app.function(gpu="L4", volumes={"/vol": weights}, timeout=3 * 3600)
def arm(name: str) -> str:
    import subprocess

    model, extra = ARMS[name]
    out = f"/vol/probe/{name}"
    train = [
        "python", "-u", "scripts/train_lora.py", model, "data/v22_head_mix_train.jsonl", "--out", out, "--freeze-adapter", "--head", "pointer",
        "--epochs", "1", "--batch", "8", "--head-lr", "5e-4", "--max-length", "2048", "--val-frac", "0.03", "--eval-every", "800", *extra,
    ]  # fmt: skip
    t = subprocess.run(train, cwd="/root", capture_output=True, text=True, check=False)
    log = [line for line in (t.stdout + t.stderr).splitlines() if any(k in line for k in ("kept step", "val:", "done in", "Error", "Traceback"))]
    if t.returncode:
        return f"== {name}: training failed\n" + "\n".join(log[-20:]) + "\n" + (t.stdout + t.stderr)[-3000:]
    weights.commit()
    ev = subprocess.run(
        ["python", "scripts/eval_open_taxonomy.py", f"lora:{out}", "data/open_tax_eval.jsonl", "data/open_tax_tagger_eval.jsonl", "--batch", "8"],
        cwd="/root",
        capture_output=True,
        text=True,
        check=False,
    )
    table = [line for line in ev.stdout.splitlines() if line[:1].isalpha()]
    return f"== {name} ({model})\n" + "\n".join(log[-3:]) + "\n" + "\n".join(table) + ("" if ev.returncode == 0 else "\n" + ev.stderr[-2000:])


@app.local_entrypoint()
def main() -> None:
    for result in arm.map(list(ARMS)):
        print(result, flush=True)
