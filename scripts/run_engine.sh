#!/bin/bash
# Build + run the single-engine shared-KV runtime.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_BUILD="${LLAMA_BUILD:-/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp/build}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf}"
TRACE="${TRACE:-/root/autodl-tmp/exp/traces_unified/sessions_react.txt}"
AGENTS="${AGENTS:-3}"
PRE_STREAM="${PRE_STREAM:--1}"
DEC_STREAM="${DEC_STREAM:-1}"
N_CTX="${N_CTX:-49152}"
SRC="${ROOT}/src/runtime/agentserve_engine.cpp"

g++ -std=c++17 -O2 \
  -I"$LLAMA_BUILD/../include" -I"$LLAMA_BUILD/../ggml/include" -I/usr/local/cuda/include \
  -o "${ROOT}/build/agentserve_engine" "$SRC" \
  -L"$LLAMA_BUILD/bin" -L/usr/local/cuda/lib64 \
  -lllama -lggml -lggml-cuda -lggml-cpu -lggml-base -lcuda -lcudart \
  -Wl,-rpath,"$LLAMA_BUILD/bin"

echo "running engine: agents=$AGENTS ctx=$N_CTX"
env -u AGENTSERVE_PREFILL_PCT -u AGENTSERVE_DECODE_PCT "${ROOT}/build/agentserve_engine" "$TRACE" "$AGENTS" "$PRE_STREAM" "$DEC_STREAM" "$N_CTX"
