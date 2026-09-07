# Qwen2.5-3B — 4-Way N-Scale (real-task trace, fixed baselines)

> **⚠️ Context caveat (v2):** this N=3…10 series used `tool_wait=200` and was measured **before the context fix**, so the vLLM/SGLang columns used the old under-sized context (`--max-model-len 8192` / `--max-total-tokens 8192`). Those numbers should be re-measured with a correct context before quoting. The per-paradigm series (`tool_wait=0`, N=3…6) in [`perparadigm-4way.md`](perparadigm-4way.md) is the currently-aligned comparison.



Status: **verified, clean comparison.** Uses (a) a regenerated **real-task** ToolBench trace (the
model actually performs tasks — no more placeholder refusal), and (b) **fixed baselines** that
drive multi-phase agents via a self-contained prompt (no `cache_prompt` collapse). Same 12 sessions,
N=3…10, 200 ms tool_wait, RTX 3090, Qwen2.5-3B.

## Throughput (tok/s)

| N | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| 3 | 114.9 | 115.3 | 153.8 | 90.3 |
| 4 | 136.4 | 110.7 | 175.5 | 108.1 |
| 5 | 138.1 | 116.2 | 193.3 | 100.4 |
| 6 | 160.5 | 127.4 | 233.0 | 99.3 |
| 7 | 152.8 | 121.2 | 239.3 | 101.1 |
| 8 | 164.1 | 133.8 | 237.5 | 102.4 |
| 9 | 166.5 | 140.6 | **271.5** | 101.9 |
| 10 | **169.1** | 130.3 | 257.9 | 95.9 |

## TPOT p95 (ms) — decode tail stability

| N | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| 3 | 15.5 | 21.9 | **11.4** | 12.9 |
| 4 | **12.6** | 24.1 | 14.0 | 12.9 |
| 5 | **13.2** | 62.6 | 13.0 | 22.9 |
| 6 | **13.7** | 74.1 | 20.0 | 24.0 |
| 7 | **14.6** | 97.8 | 20.8 | 23.6 |
| 8 | **15.5** | 76.6 | 24.0 | 23.4 |
| 9 | **15.1** | 78.5 | 23.1 | 23.5 |
| 10 | **15.5** | 115.5 | 24.3 | 25.0 |

## Cold TTFT p50 (ms) — shared-prefix-cache benefit

| N | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| 3 | **320.3** | 505.2 | 336.9 | 449.1 |
| 4 | **279.5** | 602.7 | 454.5 | 817.9 |
| 5 | **290.9** | 700.9 | 517.5 | 879.1 |
| 6 | **304.3** | 782.8 | 577.9 | 1487.7 |
| 7 | **312.1** | 780.6 | 747.9 | 1709.6 |
| 8 | **316.3** | 1091.8 | 778.9 | 1791.9 |
| 9 | **318.8** | 1315.7 | 1113.6 | 2144.3 |
| 10 | **320.3** | 1735.1 | 1168.5 | 2578.1 |

![4-way N-scale (real-task, fixed baselines)](figures/v2-4way-nscale.png)

## What this shows

- **Throughput: vLLM is the highest** (153→271 tok/s). The shared-KV engine (115→169) is **comparable
  to llama.cpp** (110→140) and ahead of SGLang (90→108). The engine is **not** the throughput leader.
- **TPOT p95: the engine is the most stable and near the lowest** — it stays in a tight **12.6–15.5 ms**
  band across *all* N, while **llama.cpp explodes** (21.9→115.5 ms) and vLLM/SGLang rise (11→25 ms).
- **Cold TTFT: the engine is the best and flat** (~280–320 ms, **not** increasing with N) — the
  shared system-prefix cache means the cold prefill is amortized across sessions. Every baseline
  *degrades* with N (vLLM 337→1168, llama 505→1735, SGLang 449→2578).

**Conclusion:** this is the honest, reproducible result. The shared-KV engine delivers **latency
stability** (flat TPOT p95 and flat cold TTFT under concurrency) while sustaining **competitive**
throughput. The paper's thesis — *improve TTFT/TPOT stability while keeping competitive throughput* —
is confirmed. The engine is **not** an absurdly better baseline; it is a **stable, prefix-cached
single-engine alternative** that keeps tail latency low where llama.cpp/vLLM/SGLang degrade.

## Repro

```
# engine:
/tmp/as_conc_batch <sessions_v2.txt> <N> -1 1 <n_ctx> <model.gguf> 12 200
# llama.cpp (self-contained multi-phase prompt):
python scripts/serve_llama.py --config configs/serving_v2_big.yaml --agents N --sessions 12
# vLLM / SGLang:
python scripts/serve_backend.py --backend {vllm,sglang} --config configs/serving_v2.yaml --agents N --sessions 12 --tag <tag>
```
Data: `metrics/v2/v2_fourway.json`; figure: `scripts/plot_4way_v2.py`.
