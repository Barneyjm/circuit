"""Acceptance: TypeSafe's official Python SDK works against our endpoint
with only `base_url` changed. Boots a real uvicorn process on a free
port (the SDK ships its own vendored httpx, so an in-process ASGI
transport isn't an option)."""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    env = {**os.environ, "S1_MODEL": os.environ.get("S1_ACCEPTANCE_MODEL", "fake"), "PORT": str(port)}
    proc = subprocess.Popen([sys.executable, "-m", "s1proto"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if httpx.get(f"{url}/healthz", timeout=2).json().get("ok"):
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("server did not come up")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_official_sdk_end_to_end(server_url):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    with TypeSafeClient(api_key="not-checked", base_url=server_url, model="s1-proto") as client:
        response = client.system_one(
            state={"document": "I was charged twice. Please fix this ASAP."},
            questions={
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
                "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
            },
        )

    assert 0.0 <= response.nouls["billing"].noul <= 1.0
    tone = response.choices["tone"]
    assert tone.choice in {"calm", "frustrated", "angry"}
    assert abs(sum(tone.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= tone.confidence <= 1.0
    urg = response.scores["urgency"]
    assert 0.0 <= urg.score <= 2.0
    # The SDK normalizes legend keys to ints; compare by value order.
    assert [urg.legend[k] for k in sorted(urg.legend, key=int)] == ["can wait", "this week", "today"]
