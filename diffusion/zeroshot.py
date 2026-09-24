"""Zero-shot: can a base masked-diffusion model answer a choice question at all? No training.

    uv run --with modal modal run diffusion/zeroshot.py --rows <rows.jsonl>

Each row (family, state, question, ref) becomes one chat prompt with lettered options. The
diffusion model (Qwen3-0.6B converted to masked diffusion) answers in one pass: the reply is a
single <|mask|>, and the option letters' probabilities are read at it. Plain Qwen3-0.6B gets
the same prompt and is read at its next token. Per family: top-1 against the reference, how
often "other" is picked, and the probability that lands on a valid letter at all.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8", "transformers==4.57.0", "accelerate>=1.10", "huggingface_hub"
    )  # the version the diffusion model was saved with; its code reads layer attributes 5.x removed
    .env({"HF_HOME": "/vol/hf"})
    # The model file imports `dllm` only in its __main__ demo, but transformers checks every
    # import in remote code before loading it; an empty module satisfies the check.
    .run_commands("python -c \"import site, pathlib; d = pathlib.Path(site.getsitepackages()[0]) / 'dllm'; d.mkdir(); (d / '__init__.py').touch()\"")
)
weights = modal.Volume.from_name("circuit-weights", create_if_missing=True)
app = modal.App("circuit-diffusion-zeroshot", image=image)
MODELS = {"diffusion": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "causal": "Qwen/Qwen3-0.6B"}
OTHER = {"other", "none", "none_of_these", "not_listed"}


def prompt(row: dict) -> tuple[str, list[str]]:
    state = row["state"]
    text = "\n".join(state["transcript"]) if isinstance(state, dict) else str(state)
    keys = list(row["question"]["criteria"])
    letters = [chr(65 + i) for i in range(len(keys))]
    opts = "\n".join(f"{L}) {k}: {d or ''}".rstrip(": ") for L, k, d in zip(letters, keys, row["question"]["criteria"].values(), strict=True))
    body = f"{text}\n\nQuestion: {row['question']['instructions']}\n{opts}\n\nAnswer with the letter only."
    return body, keys


@app.function(gpu="L4", volumes={"/vol": weights}, timeout=1800)
def run(rows: list[dict]) -> dict:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    out = {}
    for name, mid in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        cls = AutoModelForMaskedLM if name == "diffusion" else AutoModelForCausalLM
        model = cls.from_pretrained(mid, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        res = []
        for r in rows:
            body, keys = prompt(r)
            chat = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if name == "diffusion":
                chat += tok.mask_token + "<|im_end|>"
            ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if not res:  # one warm-up pass, untimed: kernels compile and memory settles
                with torch.inference_mode():
                    model(input_ids=ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_ids=ids).logits[0]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            pos = (ids[0] == tok.mask_token_id).nonzero()[0, 0] if name == "diffusion" else ids.shape[1] - 1
            p = torch.softmax(logits[pos].float(), -1)
            letter_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(len(keys))]
            lp = p[letter_ids]
            gold = max(r["ref"], key=r["ref"].get)
            pick = keys[int(lp.argmax())]
            res.append(
                {
                    "family": r["family"],
                    "gold": gold,
                    "pick": pick,
                    "on_letters": float(lp.sum()),
                    "p_gold": float(lp[keys.index(gold)] / lp.sum()),
                    "ms": ms,
                    "tokens": ids.shape[1],
                }
            )
        out[name] = res
        del model
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main(rows: str) -> None:
    import json
    from collections import defaultdict

    data = [json.loads(line) for line in open(rows)]
    out = run.remote(data)
    for name, res in out.items():
        print(f"\n== {name} ({MODELS[name]})")
        groups = defaultdict(list)
        for r in res:
            for g in ("all", r["family"].split("_ref_")[0] if "_ref_" in r["family"] else r["family"], r["family"]):
                groups[g].append(r)
        for g, rs in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
            n = len(rs)
            top1 = sum(r["pick"] == r["gold"] for r in rs) / n
            other = sum(r["pick"] in OTHER for r in rs) / n
            on = sum(r["on_letters"] for r in rs) / n
            print(f"  {g:22} n {n:3}  top1 {top1:.2f}  other {other:.2f}  p_on_letters {on:.2f}  p_gold {sum(r['p_gold'] for r in rs) / n:.2f}")
        ms = sorted(r["ms"] for r in res)
        toks = sorted(r["tokens"] for r in res)
        print(
            f"  speed: one pass p50 {ms[len(ms) // 2]:.1f} ms, p90 {ms[int(0.9 * len(ms))]:.1f} ms, at a median {toks[len(toks) // 2]} tokens (L4, bf16, batch 1)"
        )
