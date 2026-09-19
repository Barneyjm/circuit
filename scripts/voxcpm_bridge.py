"""A tiny VoxCPM server run with VoxCPM's own venv, so the audio grid can clone
voices without merging two torch stacks into one environment.

    ~/Documents/code/VoxCPM/.venv/bin/python scripts/voxcpm_bridge.py

Protocol: one JSON object per stdin line {"text", "prompt_wav", "prompt_text",
"out", "steps"?, "seed"?}; one JSON line back {"out", "seconds"} or {"error"}.
The generator (s1proto.data.audio_grid) starts it once and feeds it clips.
"""

import json
import sys

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

SR = 48000
model = VoxCPM.from_pretrained(device="mps")
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    try:
        audio = np.asarray(
            model.generate(
                text=req["text"],
                cfg_value=2.0,
                prompt_wav_path=req["prompt_wav"],
                prompt_text=req["prompt_text"],
                inference_timesteps=int(req.get("steps", 12)),
                normalize=True,
                seed=int(req.get("seed", 7)),
            ),
            dtype=np.float32,
        )
        sf.write(req["out"], audio, SR)
        print(json.dumps({"out": req["out"], "seconds": round(len(audio) / SR, 2)}), flush=True)
    except Exception as e:  # keep serving; the caller decides what to do with a failed line
        print(json.dumps({"error": str(e)[:300]}), flush=True)
