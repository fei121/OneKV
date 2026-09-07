> **⚠️ RETRACTED** — The results/figures in this document were produced by an engine that ignored EOS, ran fewer sessions than the baselines, and skipped tool_wait; **and** the llama.cpp baseline harness uses `cache_prompt`, which cannot drive a multi-phase agent conversation (degenerates to a 1-token response after the first resume). These comparisons are **invalid** and are being re-worked. The engine itself is verified faithful on the phases the reference works (cold + first resume). See [`docs/known-limitations.md`](../docs/known-limitations.md).

# Qwen2.5-3B — Concurrency Scaling (Engine vs llama.cpp)

> **Note:** this is the earlier 2-backend (engine vs llama.cpp) snapshot for N=3/4/5/6.
> The full **4-way** N-scale (N=3…10, engine vs llama.cpp / vLLM / SGLang) supersedes it;
> see [`4way-nscale-3b.md`](4way-nscale-3b.md).

Same trace, N=3/4/5/6 concurrent agents, RTX 3090.

| N | **shared-KV engine** throughput | **llama.cpp** throughput | **engine** TPOT p95 | **llama** TPOT p95 |
|---|---|---|---|---|
| 3 | 166.1 | 77.9 | 12.09 | 15.09 |
| 4 | 206.7 | 111.0 | 12.56 | 56.00 |
| 5 | 241.2 | 96.4 | 13.27 | 69.41 |
| 6 | **270.9** | 113.7 | **13.32** | 72.65 |

![engine vs llama N-scaling](figures/engine-vs-llama-nscale-3b.png)

## Findings

- **Throughput scales with N and stays far above baseline**: engine 166→271 tok/s (N=3→6);
  llama.cpp 78→114 (fluctuating).
- **TPOT is fundamentally stable for the engine** (12.09→13.32 ms) while **llama.cpp explodes**
  (15.09→72.65 ms) under concurrency. This is the AgentServe decode-protection benefit, reproduced
  cleanly at N=3/4/5/6.
- Engine output correct (total_decode matches trace) at every N; no OOM (n_ctx=49152).

## How the baseline was measured

`serve_llama.py` (llama-server, slot-pinned) run per N with `--event-log` / `--results` pointing to
N-specific files. (Note: the earlier failure was because `serve_llama.py` does not accept `--tag`;
it was fixed by using `--event-log`/`--results`.)
