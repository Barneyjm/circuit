"""Convert the ClassicMiniDIY `typesafe-bench` tasks + Jev baseline into our
JSONL format: one item per (task item, question), reference = Jev's
answer distribution.

The bench's own README says it plainly: the baseline is a model's
answers, not a truth set. Numbers against it are agreement with Jev.

    uv run python scripts/build_cmdiy_eval.py /path/to/typesafe-bench --out data/cmdiy_eval.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", default="data/cmdiy_eval.jsonl")
    ap.add_argument("--baseline", default="results/typesafe-jev-1.13.0.jsonl")
    args = ap.parse_args()

    baseline = {}
    for line in open(os.path.join(args.repo, args.baseline)):
        r = json.loads(line)
        baseline[r["id"]] = r

    out = []
    skipped = Counter()
    for path in sorted(glob.glob(os.path.join(args.repo, "tasks", "*.jsonl"))):
        for line in open(path):
            t = json.loads(line)
            b = baseline.get(t["id"])
            if not b or b.get("error"):
                skipped["no_baseline"] += 1
                continue
            for qid, q in t["questions"].items():
                a = b["answers"].get(qid)
                if not a:
                    skipped["no_answer"] += 1
                    continue
                if q["type"] == "choice" and len(q["criteria"]) < 2:
                    # e.g. search-miss-triage `corrected` with no code-found candidates: only `none` remains
                    skipped["single_option_choice"] += 1
                    continue
                if q["type"] == "noul":
                    ref = {"yes": float(a["noul"]), "no": 1.0 - float(a["noul"])}
                    keys = ["yes", "no"]
                elif q["type"] == "choice":
                    keys = list(q["criteria"].keys())
                    ref = {k: float(a["probabilities"].get(k, 0.0)) for k in keys}
                else:
                    keys = [str(i) for i in range(len(q["criteria"]))]
                    ref = {k: float(a["probabilities"].get(k, 0.0)) for k in keys}
                z = sum(ref.values()) or 1.0
                ref = {k: v / z for k, v in ref.items()}
                out.append(
                    {
                        "id": f"{t['task']}/{t['id']}/{qid}",
                        "family": f"{t['task']}.{qid}",
                        "kind": q["type"],
                        "heldout": True,
                        "state": t["state"],
                        "question": q,
                        "refs": {"jev": ref, "gemini": ref},
                        "ref": ref,
                        "source": "cmdiy",
                        "jev_ms": b.get("duration_ms"),
                        "jev_input_tokens": b.get("input_tokens"),
                    }
                )
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    fams = Counter(i["family"].split(".")[0] for i in out)
    kinds = Counter(i["kind"] for i in out)
    nopts = max(len(i["ref"]) for i in out)
    ms = [i["jev_ms"] for i in out if i.get("jev_ms")]
    print(f"{len(out)} question-items from {len(baseline)} bench items; skipped {dict(skipped)}")
    print("per task:", dict(fams))
    print("kinds:", dict(kinds), "| max options:", nopts, "| Jev median ms/item:", sorted(ms)[len(ms) // 2] if ms else None)


if __name__ == "__main__":
    main()
