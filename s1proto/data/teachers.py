"""Reference-distribution teachers.

Following TypeSafe's eval method (average of two models' probabilities)
with the two we can reach from this machine:

- **Jev** (`jev-latest`) via the TypeSafe API. Returns calibrated
  distributions natively; it is also the model we are trying to
  approximate, so training against it is distillation.
- **Gemini** (Vertex, default reasoning) asked to output a JSON
  probability map. A generator asked for probabilities is not
  calibrated by construction, which is exactly why two teachers are
  averaged and a human subset is budgeted in the handoff.

Every teacher returns a dict option_key -> probability over the
question's canonical option keys (noul: "yes"/"no"; choice: the
criteria keys; score: "0".."N-1"), normalized to sum to 1.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


def option_keys(question: dict[str, Any]) -> list[str]:
    kind = question["type"]
    if kind == "noul":
        return ["yes", "no"]
    if kind in ("choice", "multi", "rank"):
        return list(question["criteria"].keys())
    if kind == "match":  # each item's row: the options, then "none"
        return [*question["criteria"], "none"]
    if kind == "locate":
        raise ValueError("locate options come from the state; use template.locate_candidates")
    return [str(i) for i in range(len(question["criteria"]))]


def item_keys(item: dict[str, Any]) -> list[str]:
    """What each position a pointer head scores means for a JSONL row: the options, or
    for locate the state's candidates and then "none". A locate ref names only the
    candidates it puts mass on; the rest are 0."""
    if item["question"]["type"] == "locate":
        from s1proto.template import locate_candidates

        return [p for p, _ in locate_candidates(item["state"])] + ["none"]
    return option_keys(item["question"])


def normalize(raw: dict[str, float], keys: list[str]) -> dict[str, float]:
    vals = [max(0.0, float(raw.get(k, 0.0))) for k in keys]
    z = sum(vals)
    if z <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / z for k, v in zip(keys, vals, strict=True)}


def average(dists: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return normalize({k: sum(d[k] for d in dists) / len(dists) for k in keys}, keys)


class JevTeacher:
    name = "jev"

    def __init__(self, api_key: str | None = None, model: str = "jev-latest", concurrency: int = 8, url: str = TYPESAFE_URL):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "x")
        self.model = model
        self.url = url
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=30.0, headers={"Authorization": f"Bearer {self.api_key}"})

    async def distribution(self, state: Any, question: dict[str, Any]) -> dict[str, float]:
        keys = option_keys(question)
        body = {"state": state, "model": self.model, "questions": {"q": question}}
        async with self.sem:
            for attempt in range(4):
                r = await self.client.post(self.url, json=body)
                if r.status_code in (429, 529):
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                r.raise_for_status()
                break
            else:
                r.raise_for_status()
        a = r.json()["answers"]["q"]
        if question["type"] == "noul":
            p = float(a["noul"])
            return {"yes": p, "no": 1.0 - p}
        return normalize(a["probabilities"], keys)

    async def aclose(self) -> None:
        await self.client.aclose()


class GeminiTeacher:
    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash", project: str = "fabled-imagery-137915", concurrency: int = 6):
        from google import genai
        from google.genai import types

        self.types = types
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location="global",
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0, max_delay=8.0, http_status_codes=[408, 429, 500, 502, 503, 504])
            ),
        )

    def _prompt(self, state: Any, question: dict[str, Any]) -> tuple[str, list[str]]:
        keys = option_keys(question)
        kind = question["type"]
        if kind == "noul":
            crit = question.get("criteria") or {}
            options_text = f'- "yes": {crit.get("true") or "the answer is yes"}\n- "no": {crit.get("false") or "the answer is no"}'
        elif kind == "choice":
            options_text = "\n".join(f'- "{k}": {v if v is not None else k}' for k, v in question["criteria"].items())
        else:
            options_text = "\n".join(f'- "{i}": {lvl}' for i, lvl in enumerate(question["criteria"]))
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
        prompt = (
            "You are a calibrated judge. Read the STATE, then answer the QUESTION by giving a probability for every option. "
            "Probabilities must reflect how often you would be right if you said that option, over many similar cases. "
            "Spread probability across options when the state is ambiguous; concentrate it when it is clear. "
            "Output only a JSON object mapping each option key to a probability; the values must sum to 1.\n\n"
            f"STATE:\n{state_text}\n\nQUESTION: {question['instructions']}\n\nOPTIONS:\n{options_text}\n\n"
            f"JSON keys must be exactly: {json.dumps(keys)}"
        )
        return prompt, keys

    async def distribution(self, state: Any, question: dict[str, Any]) -> dict[str, float]:
        prompt, keys = self._prompt(state, question)
        cfg = self.types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=4000, temperature=0.0)
        async with self.sem:
            resp = await self.client.aio.models.generate_content(model=self.model, contents=prompt, config=cfg)
        try:
            raw = json.loads(resp.text or "{}")
        except json.JSONDecodeError:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        return normalize(raw, keys)

    async def generate_json(self, prompt: str, max_output_tokens: int = 24000) -> Any:
        """Free-form JSON generation (used for state synthesis)."""
        cfg = self.types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=max_output_tokens, temperature=1.0)
        async with self.sem:
            resp = await self.client.aio.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return json.loads(resp.text or "null")
