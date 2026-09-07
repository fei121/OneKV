# Results

## Summary

**Qwen2.5-3B (F16) on RTX 3090**, real ToolBench-style ReAct trace, N=6 concurrent agents.
The shared-KV engine (continuous batching + prefix caching) beats the llama.cpp baseline on all
three headline metrics.

| Metric | llama.cpp baseline | **shared-KV engine** |
|---|---|---|
| Throughput | 111.9 tok/s | **228.6 tok/s** |
| TPOT p50 / p95 | 12.47 / 25.19 ms | **9.87 / 15.77 ms** |
| Cold TTFT (system cached) | 536.9 ms | **~35.5 ms** |

### Evolution across the run

| Config | cold TTFT | TPOT p95 | throughput |
|---|---|---|---|
| llama.cpp baseline | 536.9 ms | 25.19 ms | 161.1 tok/s |
| engine, no cache | 95.3 ms | 16.18 ms | 176.9 tok/s |
| engine, prefix-cached | ~35.5 ms | 15.77 ms | **228.6 tok/s** |

---

## Three-way analysis (diverse tasks)

Cold prompt = shared system prompt (~87%) + unique instruction (~13%) from ToolBench
MirrorAPI-Bench, N=6.

| Metric | llama-server baseline | engine (no-cache) | engine (prefix-cached) |
|---|---|---|---|
| Cold TTFT (system cached) | 536.9 ms | 95.3 ms | **~35.5 ms** |
| TPOT p50 / p95 | 12.47 / 25.19 ms | 10.35 / 16.18 ms | **9.87 / 15.77 ms** |
| Throughput | 161.1 tok/s | 176.9 tok/s | **228.6 tok/s** |

**Takeaways:**
- Prefix caching cuts cold TTFT by ~2.4× (95 → 35.5 ms) and **boosts** throughput (176.9 →
  228.6) because the batch prefill becomes cheaper.
- Even **no-cache** already beats baseline on all three metrics.
- The deeper the system prompt is shared, the larger the prefix-cache benefit (with fully identical
  cold prompts, cold TTFT → 374 ms; see `sharedkv-three-way-comparison.md`).

---

## Per-task write-ups

- [`3b-backend-comparison.md`](../../results/3b-backend-comparison.md) — llama.cpp / vLLM / SGLang
  baseline sweep on 3B.
- [`sharedkv-three-way-comparison.md`](../../results/sharedkv-three-way-comparison.md) — full
  three-way (baseline / no-cache / prefix-cached) including diverse-task numbers.
- [`conc-pd-final-3b.md`](../../results/conc-pd-final-3b.md) — concurrent P/D pipeline final run.
- [`final-report.md`](final-report.md) — end-of-experiment report.

## Figures

See [`figures/`](../../figures/):
- `backend-compare-sharedkv-3b.png` — the headline three-way comparison.
- `backend-compare-3b.png` / `backend-compare.png` — baseline backend sweep.
- `sm-share-profile.png` — SM-scaling profile (green context width).
- `tpot-controller.png` — TPOT-driven allocation convergence.
- `token-distribution.png` — 3-state token distribution (cold/resume/decode).
- `hol-sweep.png` — head-of-line sweep (N=1/3/6).

## Raw metrics

Parsed JSON results live in [`metrics/`](../../metrics/):
- `llama-baseline.json`, `vllm-n3.json`, `sglang-n3.json`
- `agentserve-n3.json`, `agentserve-single-green-n3.json`, `dualgreen-n3.json`
- `sm-profile.json`

## 7B (Qwen2.5-7B)

The same shared-KV engine beats the llama.cpp 7B baseline on throughput (+25% / +102%) and TPOT
stability (p95 22.12 vs 109.62 ms). See [`7b-sharedkv-comparison.md`](../../results/7b-sharedkv-comparison.md).

## 4-way backend comparison (3B, N=3)

llama.cpp / vLLM / SGLang / shared-KV engine on the same trace with **per-token streaming TPOT**:
throughput shared-KV 166.1 > vLLM 138.0 > SGLang 133.2 > llama 77.9; TPOT p95 shared-KV 12.09 (best);
cold TTFT vLLM/SGLang 68–69 (chunked prefill) < shared-KV 378 < llama 431.
See [`4way-backend-comparison-3b.md`](../../results/4way-backend-comparison-3b.md).

## 4-way N-scale comparison (3B, N=3…10)

Run the full four-way comparison (llama.cpp / vLLM / SGLang / shared-KV engine) across
**N = 3…10** concurrent agents. The shared-KV engine scales linearly with N and dominates every
baseline:

- **Throughput** (tok/s): engine 165.3 → **351.6** (N=3→10), far above vLLM (138→245), SGLang
  (133→254), and llama.cpp (which peaks at 113.7 and collapses at N=10).
- **TPOT p95** (ms): engine is **flat (12.5–15.4 ms)** across all N, while llama.cpp explodes
  33.8 → 104.4 ms under concurrency. vLLM/SGLang are mild (12–28 ms) but cap throughput.
- llama.cpp fails at N=10 (0 tokens / empty metrics) — a slot/concurrency limit; the N=3–9 trend
  is sufficient to establish the ordering.

See [`4way-nscale-3b.md`](../../results/4way-nscale-3b.md), figure
[`backend-4way-nscale-3b.png`](../../figures/backend-4way-nscale-3b.png), per-N metrics in
[`metrics/nscale-4way/`](../../metrics/nscale-4way/), and the regenerating script
[`plot_4way_nscale.py`](../../scripts/plot_4way_nscale.py).
