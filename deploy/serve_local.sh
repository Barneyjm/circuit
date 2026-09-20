#!/bin/zsh
# One model, one port, for the local tier the API gateway prefers before it rents a GPU.
#   deploy/serve_local.sh circuit-8b 8902
# Run under launchd (see deploy/com.decisioncircuits.plist.example) so it survives a reboot.
set -e
DIR=${0:a:h:h}
cd "$DIR"
set -a
[ -f .env ] && source .env
set +a
export S1_MODEL="lora:runs/$1"
export PORT="$2"
export S1_MAX_INFLIGHT=${S1_MAX_INFLIGHT:-3}      # running + waiting before this box declines
export S1_CONCURRENCY=${S1_CONCURRENCY:-1}        # forward passes at once; 1 on Apple silicon
export S1_QUEUE_WAIT_S=${S1_QUEUE_WAIT_S:-8}
exec uv run python -m s1proto
