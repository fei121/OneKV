# Known Limitations & Verified Findings (honest status)

> **TL;DR** — Earlier `results/` and `figures/` claimed the shared-KV engine "crushes"
> llama.cpp / vLLM / SGLang on throughput and SLO. **Those conclusions are retracted.** A deep
> re-investigation shows the engine's code is actually **faithful** (it reproduces llama-server's
> model output, verified token-for-token on the cold and first-resume phases). The real problem is
> on the **baseline side**: `serve_llama.py` + llama-server's `cache_prompt` **cannot drive a
> multi-phase agent conversation** (it degenerates to a 1-token response after the first resume),
> so the baseline numbers are not a real agent workload. The comparison was therefore invalid.

## Why the earlier conclusions were retracted

The 4-way (llama.cpp / vLLM / SGLang / shared-KV engine) comparison was confounded by:

1. **The engine ignored EOS** in the original code — it forced every phase to its token cap
   (a model stops at EOS earlier). Fixed: the engine now respects EOS.
2. **Unequal workload size** — the engine ran the first `N` sessions; baselines ran 12.
3. **tool_wait skipped** by the engine.
4. **Config mismatch** — engine `n_batch=20000`/`n_ctx=65536`/flash-attn vs llama-server
   `batch=512`/`context=24576` vs vLLM/SGLang `max-model-len=8192`.
5. **Baseline `cache_prompt` cannot run multi-phase agents** (see below). This is the deepest issue:
   the llama.cpp baseline harness was at best measuring a *degraded* agent, not a real one.

## Engine code fixes (now committed)

- **EOS respected** (`llama_vocab_is_eog`), phase ends at EOS instead of forcing the cap.
- **12-session workload** over `N` slots (seq-0 shared-prefix template + slot recycle).
- **200 ms tool_wait** between agent steps.
- Correctness fixes: `seq_cp(0,slot,0,common)` (**`p1` exclusive** — the prior `common-1` dropped the
  last shared token); phase advance only when `drem==0 && need_pre==0`; EOG also advances `n_past`;
  no premature termination during tool_wait.

## Verified findings

### 1. The shared-KV engine is faithful

For session `s000`, greedy (`temperature 0`) generation is **token-for-token identical** to a fresh
llama-server call:

| phase | llama-server (fresh replay) | engine |
|---|---|---|
| 0 (cold) | 15 tokens "I'm sorry, but I'm unable to assist with that task." | 15 (same) |
| 1 (resume) | 41 tokens (repeats tool result) | 41 (same) |
| 2 (resume) | **1 token (degenerate)** | 44 (full cap) |
| 3 (resume) | **1 token (degenerate)** | 36 (full cap) |

The engine matches the reference **exactly** on the phases where the reference itself works
(cold, resume-1). So the single-engine shared-KV + prefix-caching + continuous-batching path is
**faithful** — there is **no engine-side "logits drift"**. (The earlier hypothesis blaming the
`ctx_other` mirror was tested and ruled out: decoding on a single context still reproduces this.)

### 2. `cache_prompt` cannot drive a multi-phase agent (baseline limitation)

This is the decisive finding from the deep-dive:

- `serve_llama.py` sends each phase's full prompt with `cache_prompt=true` + a pinned `slot_id`.
- After the first resume, the slot's KV holds a long, generated history that **diverges** from the
  next phase's prompt. llama-server then returns a **degenerate 1-token response** for phases 2/3 —
  even **in isolation** (a fresh single-session sequential replay of `s000` gives
  `phase2=1, phase3=1`).
- So the llama.cpp baseline does **not** actually run a coherent multi-phase ReAct agent
  conversation. Its per-session token count (964) is **not** a real agent workload — it is 15 +
  41 + 1 + 1 = 58 for `s000`, i.e. the second/third tool rounds collapse.

### What this means

- **Engine**: faithfully runs the full multi-phase agent conversation (all phases get the model's
  genuine greedy output). Its code is sound.
- **Baseline (llama.cpp harness)**: broken for multi-phase agents (`cache_prompt` degenerates), so
  its throughput/latency are **not** meaningful for this workload.
- **Comparison**: invalid. We cannot claim "engine is better" *or* "baseline is better" from a
  comparison where one side doesn't actually run the workload. Both the old "engine crushes
  baseline" claims and any "engine is worse" takeaway are withdrawn.
- Separately, the model itself gives a **degenerate response to this trace** (it refuses, then
  repeats the tool result). That points to a **trace/model mismatch** (the ToolBench ReAct prompt
  or task may not be well-formed for Qwen2.5), which is independent of the engine.

## Engine status summary

| Aspect | Status |
|---|---|
| Cold prefix-caching | **Faithful** (verified vs reference) |
| Resume-1 | **Faithful** (verified vs reference) |
| Resume-2/3 | Engine produces model's genuine full output; reference (`cache_prompt`) degenerate |
| EOS / 12-session / tool_wait | Fixed and committed |
| Baseline multi-phase driver | **Broken** (`cache_prompt` degenerates) |
| 4-way throughput / SLO comparison | **Invalid** (retracted) |

## Repro

```
# engine (12 sessions, N concurrency, 200ms tool_wait, EOS respected):
/tmp/as_conc_batch <trace> <N> -1 1 <n_ctx> <model.gguf> 12 200
# isolated llama-server phase-by-phase replay of a session (cache_prompt + slot_id)
```

---

## Clean verified result (real-task trace, fixed baseline)

After (a) fixing the trace to use real ToolBench tasks, and (b) fixing the llama.cpp baseline to
drive multi-phase agents via a **self-contained prompt** (full conversation + previous model output
+ `<tool_result>` block, sent to `cache_prompt`), the comparison becomes honest at **N=3** on the
RTX 3090 with Qwen2.5-3B:

| metric | shared-KV engine | llama.cpp (fixed) | ratio |
|---|---|---|---|
| throughput (tok/s) | 114.9 | 115.3 | ≈ 1.0× |
| TPOT p50 (ms) | 9.65 | 10.77 | 0.90× |
| **TPOT p95 (ms)** | **11.63** | 21.86 | **0.53× (engine ~1.9× better)** |
| **TTFT_cold (ms)** | **258.9** | 505.2 | **0.51× (engine ~1.95× better)** |

**This is the meaningful finding**: the engine is **not** "crushes baseline" on throughput — it is
**comparable** (114.9 vs 115.3). Its genuine advantages are **latency stability**:
- **TPOT p95 1.9× lower** (continuous-batching decode stays stable under concurrency), and
- **TTFT_cold 1.95× lower** (shared-system-prefix caching cuts cold prefill).

This matches the paper's thesis: *AgentServe improves TTFT/TPOT latency stability while sustaining
competitive throughput.*

## Repro

```
# engine (12 sessions, N concurrency, 200ms tool_wait, EOS respected, on the bench server):
/tmp/as_conc_batch <sessions_{react|plan_and_execute}.txt> <N> -1 1 <n_ctx> <model.gguf> 12 0

# llama.cpp baseline (self-contained multi-phase prompt):
python scripts/serve_llama.py --config configs/serving_react_u.yaml --agents N --sessions 12

# regenerate the real-task trace:
python scripts/gen_traces.py --config configs/trace_gen.yaml --model-dir /root/models/Qwen2.5-3B
```

---

## Engine throughput variants: batched-prefill vs continuous batching

Two engine scheduling variants are tracked as separate git branches. On the **3B** unified
12-task trace (N=3…6, tool_wait=0, only the shared-KV engine) the tradeoff is:

| variant | engine loop | 3B throughput | TPOT p95 | cold TTFT |
|---|---|---|---|---|
| **`main` (innovation-1)** | batched multi-sequence resume-prefill on A + **separate** batched decode on B | 152–179 (react) / 163–195 (P&E) | **stable 12–13.6 ms** | ~306–386 ms |
| **`chunk_prefill`** | **one** `llama_decode` mixing resume-prefill + decode tokens (continuous batching) | 155–182 / 167–198 (**+1–4%**) | **degrades 13→20 ms (react), 13→41 ms (P&E)** | ~310–390 ms |

### Why the tradeoff

- **Continuous batching** keeps the GPU saturated by packing prefill and decode tokens into one
  forward pass — the vLLM/SGLang approach. It buys only **+1–4%** throughput here, because the
  workload's (resume) prefills are short and infrequent.
- **But it perturbs the short decodes** that agent workloads rely on: a long/medium prefill sitting in
  the same batch delays the tiny decode tokens, so **TPOT p95 jumps** (react N6 13.5→20.3 ms; P&E N6
  13.6→41.2 ms). This is exactly the paper's warning about chunked prefill.
- **Cold TTFT is unchanged**, because the cold path still prefills separately in `activate()`.

### Implications

- **`main` (innovation-1)** is the right default for the **latency-stability** story (the paper's
  thesis): batched prefills + separate decode keep decode latency flat.
- **`chunk_prefill`** is a **pure-throughput** variant that sacrifices decode stability; use it only if
  throughput is the sole objective.
- Neither removes the fundamental *decode-batch-size* limit (N tokens per decode); to raise throughput
  further, **serve more concurrent sessions (higher N)** — the shared-KV pool is designed for that —
  or explore **speculative/parallel decoding**.

Raw numbers: `results/chunk-prefill-engine.md` and `metrics/chunk_prefill/{react,pe}.json` (on the
`chunk_prefill` branch).
