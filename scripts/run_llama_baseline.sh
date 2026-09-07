#!/bin/bash
# Task 3 — llama.cpp baseline runner (thin wrapper over serve_llama.py)
set -e
cd /root/autodl-tmp/exp
PY=/root/autodl-tmp/conda_envs/vllm/bin/python
AGENTS="${1:-4}"
SESSIONS="${2:-12}"
$PY scripts/serve_llama.py --config configs/serving.yaml --agents "$AGENTS" --sessions "$SESSIONS"
