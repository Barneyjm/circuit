# Circuits on masked-diffusion language models

A masked-diffusion LM (LLaDA, Dream, or a causal model converted, such as
[dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1),
Apache-2.0) is a bidirectional transformer: every token sees every other. A circuit never
generates, so the denoising loop is not used; one forward pass gives the hidden states the pointer
head reads, the same as a causal base. What changes:

- options need no side-by-side encoding (`--parallel-options`): there is no order to undo
- a state cannot be encoded once and shared across questions: each question is a full pass
- the models load with their own code, written for transformers 4.57, so this folder has its
  own environment (`pyproject.toml` overrides transformers; the parent stays on 5.x)

`train_lora.py --masked` and the scorer (`"masked": true` in a run's config) handle the loading.

| file | what |
|---|---|
| `zeroshot.py` | Modal, one L4, no training: answer choice questions from a `<\|mask\|>` slot, against the causal model it came from |
| `pipeline_d1.sh` | a pod job: a smoke check, then a head on the frozen diffusion model and on its causal source, and the open-taxonomy evals |
| `modal_probe.py` | the same head comparison on Modal |

First zero-shot result (137 questions, Qwen3-0.6B vs its diffusion conversion): top-1 0.52 vs
0.34, "other" picked 4% vs 27%, 34 ms a pass for both on an L4. The conversion was also
instruction-tuned (tulu-3, smoltalk), so not all of the gap is attention.

    cd diffusion && uv sync                    # this environment
    uv run --with modal modal run diffusion/zeroshot.py --rows rows.jsonl
    GPUS="NVIDIA RTX A4000,NVIDIA RTX A5000" MAX_HOURS=2 \
      STAGE="data/v22_head_mix_train.jsonl data/open_tax_eval.jsonl data/open_tax_tagger_eval.jsonl" \
      scripts/pod_run.sh circuit-0.6b-dlm-head,circuit-0.6b-causal-head \
      dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1,Qwen/Qwen3-0.6B ./diffusion/pipeline_d1.sh
