#!/usr/bin/env bash
# Run pipeline25 then remove this pod 30 minutes later, whatever happened.
cd /workspace/s1-proto
./results/pipeline25.sh >> results/pipeline25.log 2>&1
echo "=== SELF-DESTRUCT in 30 min" >> results/pipeline25.log
sleep 1800
runpodctl remove pod "$RUNPOD_POD_ID"
