# Experiments

## Workload

We generate **3-state agent traces** (cold prefill / resume prefill / short decode) matching the
paper's Table I distribution, using a ToolBench-style agent loop:

- **States:** cold prefill (system + task prompt, 2.5–3.5k tokens), resume prefill (tool output
  appended, ~30–127 tokens), decode (very short, ~27–99 tokens).
- **Paradigms:** ReAct (frequent resume prefills + short decodes) and Plan-and-Execute (long cold
  prefills + medium decodes).
- **Concurrency:** N = 3 / 6 concurrent agent sessions.

Generate with `scripts/gen_traces.py --config configs/trace-gen.yaml`.

## Backends compared (all on the same GPU + trace)

| Backend | Method | Notes |
|---|---|---|
| llama.cpp (baseline) | `scripts/serve_llama.py` (llama-server, slot-pinned, per-token events) | reference baseline |
| vLLM | `scripts/serve_backend.py --backend vllm` | streamed TPOT unreliable (OpenAI stream buffering) |
| SGLang | `scripts/serve_backend.py --backend sglang` | same caveat |
| **AgentServe (this repo)** | `scripts/run_engine.sh` | single-engine shared-KV runtime |

## Metrics

- `TTFT_cold` — request arrival → first token after cold prefill.
- `TTFT_resume` — resume-prefill → first token after append.
- `TPOT` — per-token decode latency (benchmark, with `cudaDeviceSynchronize`).
- Throughput — decoded tokens / wall-clock.

Defined in `src/agentserve_repro/metrics.py` and `configs/metrics.yaml`.

## Experiment matrix

| # | Scenario | N | Status |
|---|---|---|---|
| 1 | 3-state trace generation (Table I) | — | ✅ |
| 2 | llama.cpp baseline (HoL) | 1/3/6 | ✅ |
| 3 | Green-context SM-scaling profile | — | ✅ |
| 4 | Green-context isolation microbench | — | ✅ |
| 5 | vLLM / SGLang backend comparison | 3 | ✅ |
| 6 | **Single-engine shared-KV engine** | 3/6 | ✅ |
| 7 | Engine + continuous batching | 3/6 | ✅ |
| 8 | Engine + prefix caching | 3/6 | ✅ |
| 9 | Diverse-task three-way | 6 | ✅ |

## How to reproduce the headline result

```bash
# 1. baseline
make serve-llama A=6 S=6

# 2. shared-KV engine (prefix-cached)
make engine A=6

# 3. compare + plot
make compare
```
