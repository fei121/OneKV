# Per-Paradigm 4-Way N-Scale — Qwen2.5-7B (real tasks, fixed baselines, tool_wait=0)

Status: **clean, sequential** (one backend at a time, GPU freed between runs). Same unified 12-task
set as the 3B run (`data/unified_tasks.json`; 3B/7B share the Qwen tokenizer, so task text is
identical). N=3…6, 12 sessions, tool_wait=0, RTX 3090, **Qwen2.5-7B (BASE)**. vLLM/SGLang re-measured
with adequate context (vLLM `--max-model-len 32768`, SGLang `--max-total-tokens 49152`).

## ReAct (7B)

| metric | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 84.6→103.3 | 81.9→92.4 | 88.8→123.6 | 90.3→125.2 |
| TPOT p95 3→6 (ms) | **20.8→22.3** | 65.0→125.8 | 37.9→46.0 | 31.3→44.6 |
| cold TTFT 3→6 (ms) | **524.2→625.0** | 800.6→1179.0 | 591.1→1214.5 | 590.5→958.6 |

![ReAct 7B](figures/react7-4way-nscale.png)

## P&E (7B)

| metric | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 93.2→115.8 | 92.1→108.1 | 94.8→131.3 | 97.2→134.5 |
| TPOT p95 3→6 (ms) | **20.4→22.1** | 71.5→111.1 | 21.6→110.0 | 25.6→48.5 |
| cold TTFT 3→6 (ms) | **540.9→646.0** | 803.2→1252.2 | 648.5→1036.0 | 782.8→945.2 |

![P&E 7B](figures/pe7-4way-nscale.png)

## Findings (7B, consistent with 3B)

- **The shared-KV engine is the latency-stability leader in BOTH paradigms**: TPOT p95 flat at
  **20–22 ms** while llama.cpp explodes (65–126 ms) and vLLM/SGLang rise (31–110 ms). Cold TTFT is
  **lowest & flat** (~520–650 ms, prefix caching) while baselines degrade (SGLang 590→959 ms, vLLM
  591→1215 ms, llama 800→1252 ms).
- **Throughput: vLLM ≈ SGLang highest** (89→135), the engine is 3rd (85→116), ahead of llama.cpp
  (82–108).
- **The engine's advantage grows with model size**: on 7B the TPOT-p95 gap vs llama.cpp widens
  (22 vs 126 ms at N=6) and the cold-TTFT prefix-cache advantage is more pronounced.

**Conclusion**: same as 3B — the engine delivers **decode stability + low cold latency (prefix
cache)** at **competitive (not maximal) throughput**, confirmed on Qwen2.5-7B. Numbers are on the
BASE 7B (no native tool-calling), so they measure serving performance, not agent quality.

## Repro

```
# engine (12 sessions, N concurrency, tool_wait=0):
/tmp/as_conc_batch <sessions_{react|plan_and_execute}.txt> <N> -1 1 65536 \
  /root/autodl-tmp/models/Qwen2.5-7B-f16.gguf 12 0
# llama / vLLM / SGLang (self-contained multi-phase prompt), per paradigm:
python scripts/serve_llama.py --config configs/serving_{react|pe}_7b.yaml --agents N --sessions 12
python scripts/serve_backend.py --backend {vllm,sglang} --model-path /root/models/Qwen2.5-7B \
  --config configs/serving_{react|pe}_7b.yaml --agents N --sessions 12
```
Data: `metrics/perparadigm/{react,pe}7.json`; figures: `scripts/plot_perparadigm.py`.
