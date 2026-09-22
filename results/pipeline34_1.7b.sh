#!/usr/bin/env bash
# pipeline34 on circuit-1.7b: the cheaper first test of the hard tier.
BASE=Qwen/Qwen3-1.7B-Base N=circuit-1.7b-hard exec bash "$(dirname "$0")/pipeline34.sh"
