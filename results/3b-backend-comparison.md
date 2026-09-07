# Qwen2.5-3B backend comparison (N=3, 12 sessions)

| backend(3B) | TTFT cold p50(ms) | p95(ms) | TPOT p95(ms) | throughput(tok/s) |
|---|---|---|---|---|
| llama.cpp (baseline) | 94.7 | 759.3 | 33.8 | 103.4 |
| vLLM | 217.2 | 609.1 | 0.0 | 137.9 |
| SGLang | 227.3 | 8070.2 | 0.0 | 67.7 |
| AgentServe (single-engine green) | 109.7 | 1416.0 | 39.1 | 89.7 |

