# Per-Paradigm 4-Way N-Scale — Qwen2.5-7B

> **⚠️ Context caveat (7B vLLM/SGLang):** the vLLM/SGLang columns in this file were measured with the **old under-sized context** (`--max-model-len 8192` / `--max-total-tokens 8192`), which starved SGLang's KV pool. Those numbers are **for reference only** and should be **re-measured** with `--max-model-len 32768` (vLLM) / `--max-total-tokens 49152` (SGLang) before being quoted. The engine and llama.cpp columns are valid. See [`../docs/notes/pitfalls.md`](../docs/notes/pitfalls.md).

 (real tasks, fixed baselines, tool_wait=0)

Status: **clean, sequential** (one backend at a time, GPU freed between runs). Same unified 12-task
set as the 3B run (`data/unified_tasks.json`; 3B/7B share the Qwen tokenizer, so task text is
identical). N=3…6, 12 sessions, tool_wait=0, RTX 3090, **Qwen2.5-7B (BASE)**.

## ReAct (7B)

| metric | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 84.6→103.3 | 81.9→92.4 | 88.5→125.0 | 55.2→41.3 |
| TPOT p95 3→6 (ms) | **20.8→22.3** | 65.0→125.8 | 27.9→45.0 | 30.2→24.8 |
| cold TTFT 3→6 (ms) | **524.2→625.0** | 800.6→1179.0 | 617.6→1055.2 | 960.2→3331.0 |

![ReAct 7B](../figures/react7-4way-nscale.png)

## P&E (7B)

| metric | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput 3→6 (tok/s) | 93.2→115.8 | 92.1→108.1 | 93.2→132.7 | 55.3→47.5 |
| TPOT p95 3→6 (ms) | **20.4→22.1** | 71.5→111.1 | 21.9→83.1 | 21.0→21.1 |
| cold TTFT 3→6 (ms) | **540.9→646.0** | 803.2→1252.2 | 756.9→1041.9 | 1079.5→4250.8 |

![P&E 7B](../figures/pe7-4way-nscale.png)

## Findings (7B, consistent with 3B)

- **Engine is the latency-stability leader in both paradigms**: TPOT p95 flat at **20–22 ms** (7B is
  ~1.7× the 3B value) while llama.cpp explodes to 65–126 ms; cold TTFT is **lowest & flat**
  (~520–650 ms, prefix caching) while SGLang explodes to **3331–4251 ms**.
- **Throughput: vLLM highest** (89–133), the engine is 2nd (85–116), ahead of llama.cpp and well
  above SGLang (41–55).
- **The engine's advantage grows with model size**: on 7B, the TPOT-p95 gap vs llama.cpp widens
  (22 vs 126 ms at N=6) and the cold-TTFT prefix-cache advantage is even more pronounced vs SGLang
  (646 vs 4251 ms) because the 7B prefill is heavier.

**Conclusion**: same as 3B — the engine delivers **decode stability + low cold latency (prefix
cache)** at **competitive (not maximal) throughput**, now confirmed on Qwen2.5-7B. Numbers are on the
BASE 7B (no native tool-calling), so they measure serving performance, not agent quality.

## Repro

Same as the 3B run but with the 7B model:
```
/tmp/as_conc_batch <sessions_*.txt> <N> -1 1 32768 /root/autodl-tmp/models/Qwen2.5-7B-f16.gguf 12 0
python scripts/serve_llama.py --config configs/serving_{react|pe}_7b.yaml ...
python scripts/serve_backend.py --backend {vllm,sglang} --model-path /root/models/Qwen2.5-7B --config configs/serving_{react|pe}_7b.yaml ...
```
Data: `metrics/perparadigm/{react,pe}7.json`; figures: `scripts/plot_perparadigm.py`.
