> **⚠️ RETRACTED** — The results/figures in this document were produced by an engine that ignored EOS, ran fewer sessions than the baselines, and skipped tool_wait; **and** the llama.cpp baseline harness uses `cache_prompt`, which cannot drive a multi-phase agent conversation (degenerates to a 1-token response after the first resume). These comparisons are **invalid** and are being re-worked. The engine itself is verified faithful on the phases the reference works (cold + first resume). See [`docs/known-limitations.md`](../docs/known-limitations.md).

# Qwen2.5-3B backend comparison (N=3, 12 sessions)

| backend(3B) | TTFT cold p50(ms) | p95(ms) | TPOT p95(ms) | throughput(tok/s) |
|---|---|---|---|---|
| llama.cpp (baseline) | 94.7 | 759.3 | 33.8 | 103.4 |
| vLLM | 217.2 | 609.1 | 0.0 | 137.9 |
| SGLang | 227.3 | 8070.2 | 0.0 | 67.7 |
| AgentServe (single-engine green) | 109.7 | 1416.0 | 39.1 | 89.7 |

