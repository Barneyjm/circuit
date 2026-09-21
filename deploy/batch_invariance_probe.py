"""Does batch-invariant arithmetic make our answers reproducible under batching?

    uv run --group deploy modal run deploy/batch_invariance_probe.py

Ephemeral: it does not touch the deployed app. Scores one prompt alone and in a
batch, with the batch-invariant kernels off and then on, and prints the numbers
at full precision along with what the mode costs in time.
"""

from __future__ import annotations

import modal

# Inlined rather than imported from modal_app: only s1proto is added to the
# container's source, so `deploy` does not exist inside it.
REPO = "jbarney/circuit-1.7b"
weights = modal.Volume.from_name("circuit-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # the kernels install from source
    .pip_install(
        "torch>=2.8",
        "transformers>=5.17,<6",
        "peft>=0.21",
        "accelerate>=1.15",
        "huggingface_hub",
        "pydantic>=2.13",
        "numpy",
        "triton",
        "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("s1proto")
)
app = modal.App("circuit-batch-invariance", image=image)


@app.function(gpu="L40S", volumes={"/vol": weights}, timeout=1800)
# S1_ATTN=flex_attention tests whether a batch-invariant attention path closes the
# remaining gap between prompts of different lengths.
def probe(name: str = "circuit-1.7b", attn: str = "sdpa") -> None:
    import os

    os.environ["S1_ATTN"] = attn
    import time

    from huggingface_hub import snapshot_download

    from s1proto.schema import NoulQuestion
    from s1proto.scorer import LoRAScorer
    from s1proto.template import render

    run_dir = f"/vol/{name}"
    snapshot_download(REPO, local_dir=run_dir)
    import os as _os

    attn = _os.environ.get("S1_ATTN", "sdpa")
    sc = LoRAScorer(run_dir=run_dir)
    sc.prefix_cache = False
    if attn != "sdpa":  # swap the attention backend on the loaded model
        sc.model.set_attn_implementation(attn)
    print(f"\n  attention backend: {attn}")

    q = NoulQuestion(type="noul", instructions="Does this need someone today?")
    target = render("the pipe burst on Elm street", q, layout="pointer")
    company = [render(t, q, layout="pointer") for t in ["hi", "x" * 400, "a meter reading that looks wrong", "y" * 900]]

    # Same token count, different content: the case that decides whether bucketing
    # prompts by length is enough, or whether only identical prompts are safe.
    tok = sc.tokenizer
    n_target = len(tok.encode(target.text))
    same_length = []
    for filler in ["alpha", "bravo", "charlie", "delta"]:
        text = target.text.replace("Elm street", f"{filler} street")
        ids = tok.encode(text)
        while len(ids) < n_target:
            text = text.replace("burst", "burst suddenly", 1)
            ids = tok.encode(text)
        if len(ids) == n_target:
            same_length.append(render(text.split("\n")[0], q, layout="pointer"))
    same_length = [p for p in same_length if len(tok.encode(p.text)) == n_target]

    def measure(label: str) -> None:
        sc.score([target])  # warm
        t0 = time.perf_counter()
        alone = sc.score([target])[0].probabilities[0]
        solo_ms = (time.perf_counter() - t0) * 1000
        same = sc.score([target] * 5)[0].probabilities[0]
        t0 = time.perf_counter()
        mixed = sc.score([target] + company)[0].probabilities[0]
        batch_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  {label}")
        print(f"    alone                      {alone!r}   ({solo_ms:.0f} ms)")
        print(f"    batch of 5, identical      {same!r}   diff {abs(alone - same):.2e}")
        print(f"    batch of 5, mixed lengths  {mixed!r}   diff {abs(alone - mixed):.2e}   ({batch_ms:.0f} ms)")
        if same_length:
            eq = sc.score([target] + same_length)[0].probabilities[0]
            print(f"    batch, same length new text{eq!r}   diff {abs(alone - eq):.2e}   ({len(same_length) + 1} prompts)")

    measure("standard kernels")
    from batch_invariant_ops import set_batch_invariant_mode

    with set_batch_invariant_mode():
        measure("batch-invariant kernels")
