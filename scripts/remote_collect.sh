#!/usr/bin/env bash
# Pull trained weights and results back from a GPU box.
#   scripts/remote_collect.sh <ssh-host> [ssh-port]
# Copies runs/<name>/{adapter,head.pt,config.json} (best checkpoints, not the
# `latest/` resume state) and results/*.json|*.log into this repo. Safe to run
# repeatedly; it only adds or updates.
set -euo pipefail
HOST="${1:?ssh host}"; PORT="${2:-22}"; KEY="${SSH_KEY:-$HOME/.ssh/runpod_s1}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SSHCMD="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
rsync -rltz --no-owner --no-group -e "$SSHCMD" --include='*/' --include='adapter/***' --include='head.pt' --include='config.json' --exclude='*' "root@$HOST:/workspace/s1-proto/runs/" "$SRC/runs/"
rsync -rltz --no-owner --no-group -e "$SSHCMD" --include='*.json' --include='*.log' --exclude='*' "root@$HOST:/workspace/s1-proto/results/" "$SRC/results/"
echo "collected: $(ls -d "$SRC"/runs/circuit-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
