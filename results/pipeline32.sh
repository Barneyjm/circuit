#!/usr/bin/env bash
# Both media circuits on one 48 GB card at once (vision 4B and audio 7B fit together), each
# through pipeline31, both streamed to stdout with a prefix so the driver sees progress.
# pipeline31's own DONE lines are renamed so only the last line here ends the driver's wait.
cd /workspace/s1-proto
./results/pipeline31.sh vision 2>&1 | sed -u 's/^=== PIPELINE31 \(.*\) DONE/=== finished \1/; s/^/[vision] /' &
./results/pipeline31.sh audio 2>&1 | sed -u 's/^=== PIPELINE31 \(.*\) DONE/=== finished \1/; s/^/[audio] /' &
wait
echo "=== PIPELINE32 media DONE"
