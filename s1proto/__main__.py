"""`uv run python -m s1proto` — serve on $PORT (default 8901) with $S1_MODEL."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("s1proto.service:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8901")), workers=1)
