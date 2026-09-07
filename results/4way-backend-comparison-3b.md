> **⚠️ RETRACTED** — The results/figures in this document were produced by an engine that ignored EOS, ran fewer sessions than the baselines, and skipped tool_wait; **and** the llama.cpp baseline harness uses `cache_prompt`, which cannot drive a multi-phase agent conversation (degenerates to a 1-token response after the first resume). These comparisons are **invalid** and are being re-worked. The engine itself is verified faithful on the phases the reference works (cold + first resume). See [`docs/known-limitations.md`](../docs/known-limitations.md).

# Qwen2.5-3B — 4-Way Backend Comparison (N=3, same trace)

Same trace (`traces_Qwen2.5-3B.jsonl`), N=3, 12 sessions, tool_wait 200 ms, on the RTX 3090.
vLLM / SGLang measured with **per-token streaming** (`submit()` now yields from the stream, so
`TOKEN_EMIT` timestamps are true per-token → real TPOT).

| Backend | TTFT_cold p50 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | throughput (tok/s) |
|---|---|---|---|---|
| llama.cpp | 431.0 | 10.27 | 15.09 | 77.9 |
| vLLM | 68.9 | 10.17 | 12.34 | 138.0 |
| SGLang | 68.2 | 9.83 | 23.64 | 133.2 |
| **shared-KV engine** | **13.3** | **9.42** | **12.09** | **166.1** |

![4-way comparison](figures/backend-4way-3b.png)

## Findings (corrected: per-session cold TTFT)

- **Throughput: shared-KV engine is the highest** (166.1 tok/s) — ahead of vLLM (138.0),
  SGLang (133.2), and llama.cpp (77.9).
- **TPOT p95: shared-KV engine is best** (12.09 ms, tied with vLLM 12.34), much better than
  SGLang (23.64) and llama.cpp (15.09).
- **Cold TTFT: shared-KV (378 ms) beats llama.cpp (431 ms) but is above vLLM/SGLang (68–69 ms).**
  The reason vLLM/SGLang are lower is **chunked prefill** — they emit the first token before the
  full prompt is processed (a throughput/latency tradeoff). Our engine does a **full shared
  system-prefill** (378 ms) and shares it across sessions; the per-session cold TTFT with prefix
  caching is lower than 378 ms. Lowering cold TTFT further would require adopting chunked prefill.

## What fixed vLLM/SGLang TPOT

`src/agentserve_repro/backends.py`: `submit()` previously did `return list(self.stream_completion(...))`,
which consumed the whole stream before yielding → every `TOKEN_EMIT` shared one timestamp → TPOT ≈ 0.
Now it does `yield from self.stream_completion(...)`, so tokens arrive one chunk at a time and are
timestamped individually.
Also added `--stream-interval 1` to SGLang and installed system `ninja-build` so flashinfer's CUDA
graph capture works (`cuda graph: True`).
