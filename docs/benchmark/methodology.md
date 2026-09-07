# Benchmark methodology

How the 4-way comparison is measured, and what "fair" means for the different backends.

## Workload

We generate **3-state agent traces** matching the paper's Table-I distribution using a
ToolBench-style agent loop:

- **States:** cold prefill (system + task prompt, 2.5–3.5 k tokens), resume prefill (tool output
  appended, ~30–127 tokens), decode (very short, ~27–99 tokens).
- **Paradigms:** ReAct (frequent resume prefills + short decodes) and Plan-and-Execute (long cold
  prefills + medium decodes).
- **Concurrency:** N = 3…6 concurrent agent sessions (12 sessions total).

A single **unified 12-task set** (`data/unified_tasks.json`) is shared by both paradigms; only the
prompt template and the token-count distribution differ. Generate with
`python scripts/gen_traces.py --config configs/trace-gen.yaml` (`--pool-file` reuses the same tasks).

## Backends (all on the same GPU + trace)

| Backend | Driver | Notes |
|---|---|---|
| llama.cpp (baseline) | `scripts/serve_llama.py` | llama-server, self-contained multi-phase prompt |
| vLLM | `scripts/serve_backend.py --backend vllm` | `--max-model-len 32768` |
| SGLang | `scripts/serve_backend.py --backend sglang` | `--max-total-tokens 49152` |
| **AgentServe (this repo)** | `scripts/run_engine.sh` | single-engine shared-KV runtime |

Every backend is driven by the **same** harness: the same 12 sessions, same `N`, same `tool_wait`, same
EOS behaviour, and a **self-contained multi-phase prompt** (full conversation + previous model output
+ the phase delta) so prefix caching sees a coherent prefix.

## Metrics

- `TTFT_cold` — request arrival → first token after cold prefill.
- `TTFT_resume` — resume-prefill → first token after the append.
- `TPOT` — per-token decode latency (measured with per-token streaming; LLM kernels synced via
  `cudaDeviceSynchronize` / per-stream sync so wall-clock is meaningful).
- Throughput — decoded tokens / wall-clock.

Defined in `src/agentserve_repro/metrics.py` and `configs/metrics.yaml`.

## Context: the two different meanings of "context"

`context` is **not** the same concept across backends, which is the #1 source of unfair comparisons:

| Backend | Parameter | Meaning |
|---|---|---|
| vLLM | `--max-model-len` | **per-sequence** max (must be ≤ the model's native `max_position_embeddings`, e.g. 32768 for Qwen2.5-3B). |
| SGLang | `--max-total-tokens` | **total KV-cache pool** budget (per-sequence max comes from the model config). |
| engine | `n_ctx` | **KV pool** shared by all N sessions (no per-sequence cap). |
| llama.cpp | `context_length` | total pool, divided among `--parallel N` slots (per-slot = ctx/N). |

**Fairness = (a) no prompt truncates** (per-sequence max ≥ longest prompt ≈ 4.5 k tokens) **and
(b) the pool can hold N concurrent sessions** (pool ≥ N × ~4.5 k ≈ 27 k at N=6). All four backends
satisfy both with the parameters above. See [`../notes/pitfalls.md`](../notes/pitfalls.md) for the
SGLang `--max-total-tokens 8192` bug that starved its pool.

## Measurement hygiene

- **Serial measurement:** one backend at a time; GPU freed between runs (`scripts/kill_gpu.py`).
- **Same workload across backends** — same sessions, `N`, `tool_wait`, EOS.
- **tool_wait** must be simulated by the engine too (per-session `wait_until` gate), otherwise its
  throughput is overstated.
- llama.cpp's `cache_prompt` **cannot** drive a multi-phase agent conversation, so we use a
  self-contained prompt instead.
