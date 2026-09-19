"""Re-query both teachers for one family after its question text changed.
Keeps the states, replaces `question`, `refs`, `ref` in place.

    TYPESAFE_API_KEY=... uv run python scripts/relabel_family.py agent_next_action data/train.jsonl data/eval.jsonl
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1proto.data.families import BY_KEY
from s1proto.data.teachers import GeminiTeacher, JevTeacher, average, option_keys


async def main() -> None:
    fam = BY_KEY[sys.argv[1]]
    q = fam.question()
    keys = option_keys(q)
    jev, gem = JevTeacher(concurrency=12), GeminiTeacher(concurrency=12)
    for path in sys.argv[2:]:
        rows = [json.loads(line) for line in open(path)]
        targets = [r for r in rows if r["family"] == fam.key]

        async def one(r):
            dj, dg = await asyncio.gather(jev.distribution(r["state"], q), gem.distribution(r["state"], q))
            r["question"] = q
            r["refs"] = {"jev": dj, "gemini": dg}
            r["ref"] = average([dj, dg], keys)

        results = await asyncio.gather(*[one(r) for r in targets], return_exceptions=True)
        errs = sum(1 for x in results if isinstance(x, BaseException))
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: relabeled {len(targets) - errs}/{len(targets)} {fam.key} items")
    await jev.aclose()


if __name__ == "__main__":
    asyncio.run(main())
