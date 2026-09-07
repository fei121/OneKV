# Per-Paradigm 4-Way N-Scale (ReAct / P&E) — real tasks, fixed baselines, tool_wait=0

Status: **clean, sequential measurement** (one backend at a time, GPU freed between runs — no
server concurrency). Same **unified 12-task set** (`data/unified_tasks.json`) for BOTH paradigms;
only the prompt template (ReAct vs P&E) and token-count distribution differ. N=3…6, 12 sessions,
tool_wait=0, RTX 3090, Qwen2.5-3B (BASE).

## ReAct

| metric (N) | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 149.8→175.2 | 131.1→145.0 | 165.4→224.7 | 104.1→82.4 |
| TPOT p95 3→6 (ms) | **11.9→13.5** | 60.0→72.7 | 19.8→35.1 | 20.6→21.1 |
| cold TTFT 3→6 (ms) | **308.6→367.6** | 439.5→737.8 | 338.1→575.9 | 525.6→1660.9 |

![ReAct 4-way](figures/react-4way-nscale.png)

## Plan-and-Execute

| metric (N) | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 162.5→192.2 | 145.1→147.7 | 169.5→255.0 | 105.3→84.2 |
| TPOT p95 3→6 (ms) | **11.8→13.5** | 50.2→70.7 | 23.3→33.0 | 11.6→11.2 |
| cold TTFT 3→6 (ms) | **316.5→383.8** | 412.3→773.2 | 443.8→779.6 | 779.6→1975.0 |

![P&E 4-way](figures/pe-4way-nscale.png)

## Findings (consistent across paradigms)

- **The shared-KV engine is the latency-stability leader in BOTH paradigms**: its **TPOT p95 stays
  flat at ~12–14 ms** across N=3…6 (llama.cpp explodes to 60–73 ms; vLLM/SGLang rise to 20–35 ms),
  and its **cold TTFT is the lowest and flat** (~300–380 ms, shared-system-prefix caching) while every
  baseline degrades with N (SGLang cold TTFT explodes to 1660–1975 ms).
- **Throughput is NOT the engine's strength**: vLLM is the highest in both paradigms (165→255 tok/s);
  the engine is consistently 2nd (150→192), ahead of llama.cpp and well ahead of SGLang.
- **P&E vs ReAct**: the engine behaves the same (flat TPOT/TTFT). SGLang has unusually **low TPOT p95
  in P&E** (~11 ms) but the **worst cold TTFT** (explodes to 1975 ms) and the lowest throughput — a
  strong latency-vs-throughput tradeoff.

**Conclusion**: this is the honest, per-paradigm serving comparison. The engine delivers **decode
stability + low cold latency via prefix caching**, at **competitive (not maximal) throughput** — the
paper's thesis. The absolute numbers are for the BASE Qwen2.5-3B (no native tool-calling; model
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
