# Results

> **⚠️ RETRACTED (old data)** — earlier `results/*.md` and `figures/*.png` reported the engine
> "crushing" llama.cpp/vLLM/SGLang. Those were produced by an engine that **ignored EOS**, ran fewer
> sessions, skipped tool_wait, and by a llama.cpp baseline whose `cache_prompt` cannot drive a
> multi-phase agent (it collapsed to 1-token replies after the first resume). **Those comparisons are
> invalid** and have been removed. See [`known-limitations.md`](../known-limitations.md).

## Current, valid findings

The clean, reproducible comparisons use the **real-task** ToolBench trace (model actually performs
tasks), **fixed** baselines (self-contained multi-phase prompt), a **unified 12-task set** shared by
ReAct and P&E, **N=3…6**, **tool_wait=0**, and a **serial** measurement (one backend at a time, GPU
freed between runs).

### Per-paradigm 4-way (unified 12 tasks, tool_wait=0)

- [`perparadigm-4way.md`](../../results/perparadigm-4way.md) — Qwen2.5-**3B**, ReAct + P&E.
- [`perparadigm-4way-7b.md`](../../results/perparadigm-4way-7b.md) — Qwen2.5-**7B**, ReAct + P&E.
- [`v2-4way-nscale.md`](../../results/v2-4way-nscale.md) — combined 4-way N=3…10 (3B).

**Bottom line (consistent across ReAct/P&E and 3B/7B):**
- The **shared-KV engine is the latency-stability leader**: TPOT p95 stays **flat** (~12–14 ms on 3B,
  ~20–22 ms on 7B) while llama.cpp explodes (60–126 ms) and vLLM/SGLang rise; cold TTFT is **lowest &
  flat** (prefix-cache amortization) while baselines degrade with N (SGLang explodes to 1k–4k ms).
- **Throughput is not the engine's strength**: vLLM is highest; the engine is consistently 2nd
  (ahead of llama.cpp and well above SGLang).

### Verdict

The engine delivers **decode stability + low cold latency via prefix caching** at **competitive (not
maximal) throughput** — the paper's thesis. Numbers are on the **BASE** Qwen2.5-3B/7B (no native
tool-calling, plan-style output), so they measure **serving performance**, not agent quality.

## Docs

- [`known-limitations.md`](../known-limitations.md) — methodology pitfalls + the honest status.
- [`pitfalls.md`](../pitfalls.md) — lessons learned (model paths, service startup, engine bugs,
  baseline harness, benchmark methodology, shell gotchas).
- [`figures/`](../../figures/) — final per-paradigm + combined 4-way figures.

## Repro

```
# unified task set (shared by ReAct & P&E):
data/unified_tasks.json
# engine (12 sessions, N concurrency, tool_wait=0):
/tmp/as_conc_batch <sessions_{react|plan_and_execute}.txt> <N> -1 1 <n_ctx> <model.gguf> 12 0
# llama / vLLM / SGLang:
python scripts/serve_llama.py --config configs/serving_{react|pe}_{u|7b}.yaml --agents N --sessions 12
python scripts/serve_backend.py --backend {vllm,sglang} --model-path /root/models/Qwen2.5-{3B|7B} --config configs/serving_{react|pe}_{u|7b}.yaml --agents N --sessions 12
# figures:
python scripts/plot_perparadigm.py   # per-paradigm (3B + 7B)
python scripts/plot_4way_v2.py      # combined
```
