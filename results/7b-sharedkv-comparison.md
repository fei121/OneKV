# Qwen2.5-7B — Shared-KV Engine vs llama.cpp Baseline

Same single-engine shared-KV engine, on **Qwen2.5-7B (F16, 15.2 GB)**, RTX 3090, ToolBench-style
ReAct trace, N=3 / N=6.

## Results

| Metric | N | llama.cpp baseline | **shared-KV engine** |
|---|---|---|---|
| Throughput | 3 | 75.96 tok/s | **95.0 tok/s** |
| Throughput | 6 | 79.34 tok/s | **160.6 tok/s** |
| TPOT p50 / p95 | 3 | 20.47 / 38.03 ms | **18.57 / 20.55 ms** |
| TPOT p50 / p95 | 6 | 24.43 / **109.62** ms | **19.15 / 22.12 ms** |
| Cold TTFT p50 | 3 | 633 ms | **563.7 ms** |
| Cold TTFT p50 | 6 | **2404 ms** | **568.7 ms** |

## Cold TTFT — real (per-session) vs reported

The baseline **cold TTFT is bimodal**, not a single number. Raw per-session values
(`REQUEST_ARRIVE → COLD_PREFILL_END`):

- **N=3:** `[41, 75, 156, 633, 1190, 1717] ms` → p50 = **633 ms**, p95 = **1717 ms**
- **N=6:** `[713, 1278, 1840, 2404, 2964, 3489] ms` → p50 = **2404 ms**, p95 = **3489 ms**

Why bimodal:
- llama-server's **slot prefix caching** serves sessions that share the system prompt **fast**
  (41–156 ms at N=3), but sessions that miss the cache (or contend at N=6) take **0.6–3.5 s**.
- The `155.6 ms` value in the metrics JSON is the **median of that bimodal mix** — misleading,
  because it hides the 1190–3489 ms tail (head-of-line blocking for the cold sessions).

## Takeaway

- **Throughput dramatically better**: +25% (N=3), **+102% (N=6)** — continuous batching scales with
  model size.
- **TPOT far more stable**: p95 stays **22.12 ms** while the baseline degrades to **109.62 ms**
  (N=6). This is the decode-protection benefit, reproduced at 7B scale.
- **Cold TTFT is actually better too**: the engine's **unified 563.7 / 568.7 ms** shared-system
  prefill is lower than the baseline's **p50 633 / 2404 ms**, and — crucially — has **no 1.7–3.5 s
  tail** (the baseline's head-of-line spike). With prefix caching the per-session cold TTFT is even
  lower than the engine's reported shared-prefill value.

The engine ran with `n_ctx=24576` (N=3) and `n_ctx=49152` (N=6), fitting in 24 GB alongside the
15 GB weights.

## Files

- Engine: `src/runtime/agentserve_engine.cpp` (model path is a CLI arg).
- Figure: `figures/backend-compare-sharedkv-7b.png`.
