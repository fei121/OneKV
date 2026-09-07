# Per-Paradigm 4-Way N-Scale (ReAct / P&E) — real tasks, fixed baselines, tool_wait=0

Status: **clean, sequential measurement** (one backend at a time, GPU freed between runs — no
server concurrency). Same **unified 12-task set** (`data/unified_tasks.json`) for BOTH paradigms;
only the prompt template (ReAct vs P&E) and token-count distribution differ. N=3…6, 12 sessions,
tool_wait=0, RTX 3090, Qwen2.5-3B (BASE).

## ReAct

| metric (N) | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 149.8→175.2 | 131.1→145.0 | 167.7→222.8 | 164.2→226.3 |
| TPOT p95 3→6 (ms) | **11.9→13.5** | 60.0→72.7 | 19.5→24.0 | 24.0→27.9 |
| cold TTFT 3→6 (ms) | **308.6→367.6** | 439.5→737.8 | 302.7→621.4 | 302.2→454.8 |

![ReAct 4-way](figures/react-4way-nscale.png)

## Plan-and-Execute

| metric (N) | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 162.5→192.2 | 145.1→147.7 | 170.4→259.1 | 179.5→243.1 |
| TPOT p95 3→6 (ms) | **11.8→13.5** | 50.2→70.7 | 22.0→38.7 | 24.8→31.4 |
| cold TTFT 3→6 (ms) | **316.5→383.8** | 412.3→773.2 | 343.0→603.4 | 335.0→478.2 |

![P&E 4-way](figures/pe-4way-nscale.png)

## Findings (consistent across paradigms)

- **The shared-KV engine is the latency-stability leader in BOTH paradigms**: its **TPOT p95 stays
  flat at ~12–14 ms** across N=3…6, while llama.cpp explodes (50–73 ms) and vLLM/SGLang rise
  (20–39 ms). Its **cold TTFT is the lowest and roughly flat** (~310–385 ms, shared-system-prefix
  caching) while every baseline degrades with N (vLLM 302→621 ms, SGLang 302→455 ms, llama 440→773 ms).
- **Throughput is NOT the engine's strength**: **vLLM (168→259) and SGLang (164→243) now lead** —
  once the SGLang harness bug was fixed, SGLang's throughput is on par with vLLM and **above the
  engine**. The engine is 3rd (150→192), ahead of llama.cpp (131–148).
- **P&E vs ReAct**: the engine behaves identically (flat TPOT/TTFT). llama.cpp is worst on latency
  (TPOT p95 50–73 ms, coldest TTFT) in both.

> **Note on SGLang/vLLM numbers.** These were **re-measured with adequate context**. The earlier
> harness hard-coded `--max-total-tokens 8192` for SGLang, which starved its total KV-cache pool and
> made it look like the worst backend (throughput dropped with N, cold TTFT exploded to 1–2 s). That
> was a **harness bug, not an SGLang limitation**. Now SGLang uses `--max-total-tokens 49152`
> (KV pool) and vLLM uses `--max-model-len 32768` (must not exceed the model's native
> `max_position_embeddings` = 32768). See [`docs/pitfalls.md`](docs/pitfalls.md).

**Conclusion**: the engine delivers **decode stability + low cold latency via prefix caching** at
**competitive (not maximal) throughput** — i.e. it trades some throughput for TTFT/TPOT stability
(paper's thesis). vLLM/SGLang get higher throughput but clearly higher TPOT p95 and cold TTFT under
concurrency; llama.cpp is worst on latency. The absolute numbers are for the BASE Qwen2.5-3B (no native tool-calling; model
produces plan-style output), so they measure serving performance, not agent quality.

## Repro

```
# engine (12 sessions, N concurrency, tool_wait=0):
/tmp/as_conc_batch <sessions_{react|plan_and_execute}.txt> <N> -1 1 <n_ctx> <model.gguf> 12 0
# llama / vLLM / SGLang (self-contained multi-phase prompt), per paradigm:
python scripts/serve_llama.py --config configs/serving_{react|pe}_u.yaml --agents N --sessions 12
python scripts/serve_backend.py --backend {vllm,sglang} --config configs/serving_{react|pe}_u.yaml --agents N --sessions 12
# unified task set: data/unified_tasks.json (12 items + schedule, shared by both paradigms)
```
Data: `metrics/perparadigm/{react,pe}.json`; figures: `scripts/plot_perparadigm.py`.
