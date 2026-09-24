"""Install a head trained with --freeze-adapter as a plug-in head of the run it was trained on.

    uv run python scripts/export_head.py runs/circuit-1.7b-v2.2-head runs/circuit-1.7b-v2.0 support

Checks that the head-only run's adapter is the base run's, tensor for tensor (a head reads the
hidden states of the adapter it was trained with and no other), then writes
`<base>/heads/<name>/head.pt` and `config.json` with the head's fitted temperatures, the
question types it answers, its validation numbers and the base adapter's sha256, which the
scorer checks at load. A request then names it as `<model>+<name>`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1proto.scorer import _file_sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trained", help="the head-only run (train_lora --init <base> --freeze-adapter)")
    ap.add_argument("base", help="the run whose adapter it was trained on")
    ap.add_argument("name", help="the head's name: letters, digits, - and _")
    args = ap.parse_args()
    trained, base = Path(args.trained), Path(args.base)
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise SystemExit(f"head name {args.name!r}: letters, digits, - and _ only")

    a, b = load_file(trained / "adapter" / "adapter_model.safetensors"), load_file(base / "adapter" / "adapter_model.safetensors")
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise SystemExit(f"{trained} was not trained on {base}'s adapter (train it with --init {base} --freeze-adapter)")
    cfg, base_cfg = json.loads((trained / "config.json").read_text()), json.loads((base / "config.json").read_text())
    for k in ("base", "hidden", "head", "head_dim", "pointer_tokens", "parallel_options"):
        if cfg.get(k) != base_cfg.get(k):
            raise SystemExit(f"config {k!r} differs: {cfg.get(k)!r} here, {base_cfg.get(k)!r} in {base}")

    out = base / "heads" / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trained / "head.pt", out / "head.pt")
    meta = {
        "name": args.name,
        "adapter_sha256": _file_sha(str(base / "adapter" / "adapter_model.safetensors")),
        "temperatures": cfg.get("temperatures") or {},
        "question_types": cfg.get("question_types"),
        "best": cfg.get("best"),
        "trained_from": str(trained),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"installed {out} ({(out / 'head.pt').stat().st_size / 1e6:.1f} MB); ask for it as <model>+{args.name}")


if __name__ == "__main__":
    main()
