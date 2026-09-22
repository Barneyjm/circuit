#!/usr/bin/env bash
# Both media circuits on one 48 GB card at once (vision 4B and audio 7B fit together), each
# through pipeline31. One DONE line at the end so the driver waits for both.
cd /workspace/s1-proto
./results/pipeline31.sh vision > results/pipeline31_vision.log 2>&1 &
./results/pipeline31.sh audio > results/pipeline31_audio.log 2>&1 &
wait
cat results/pipeline31_vision.log results/pipeline31_audio.log
echo "=== PIPELINE32 media DONE"
