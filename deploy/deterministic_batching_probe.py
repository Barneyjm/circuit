"""Does a batched answer reproduce when prompts are bucketed by length?

    uv run --group deploy modal run deploy/deterministic_batching_probe.py

Runs the Batcher itself, not the scorer directly, so what is measured is what
the service would do: uniform-length batching with batch-invariant kernels.
"""

from __future__ import annotations

import modal

REPO = "jbarney/circuit-1.7b"
weights = modal.Volume.from_name("circuit-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
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
app = modal.App("circuit-deterministic-batching", image=image)


@app.function(gpu="L40S", volumes={"/vol": weights}, timeout=1800)
def probe() -> None:
    import threading
    import time

    from huggingface_hub import snapshot_download

    from s1proto.batching import Batcher
    from s1proto.schema import NoulQuestion
    from s1proto.scorer import LoRAScorer
    from s1proto.template import render

    run_dir = "/vol/circuit-1.7b"
    snapshot_download(REPO, local_dir=run_dir)
    sc = LoRAScorer(run_dir=run_dir)
    sc.prefix_cache = False
    q = NoulQuestion(type="noul", instructions="Does this need someone today?")
    target = render("the pipe burst on Elm street", q, layout="pointer")

    # Traffic of assorted lengths, as a shared endpoint actually sees it.
    others = [
        render(t, q, layout="pointer")
        for t in ["hi", "the meter is stuck", "x" * 300, "water in the road since Tuesday morning", "y" * 800, "ok thanks", "z" * 120]
    ]

    def run(uniform: bool) -> None:
        b = Batcher(sc, max_batch=8, wait_ms=25, uniform=uniform)
        alone = b.score([target])[0].probabilities[0]
        got: list[float] = []

        def fire(p):
            got.append(b.score([p])[0].probabilities[0])

        threads = [threading.Thread(target=fire, args=(target,))] + [threading.Thread(target=fire, args=(o,)) for o in others]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        batched = got[0]
        print(f"\n  uniform={uniform}  batch_invariant={b.stats['batch_invariant']}")
        print(f"    alone                {alone!r}")
        print(f"    in mixed traffic     {batched!r}   diff {abs(alone - batched):.2e}")
        print(f"    8 requests in {wall * 1000:.0f} ms, {b.stats['batches']} passes, {b.stats['prompts_per_batch']} prompts per pass")

    run(uniform=True)
    run(uniform=False)
