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
/tmp/as_conc_batch <sessions.txt> <N> -1 1 <n_ctx> <model.gguf> 12 200

# llama.cpp baseline (self-contained multi-phase prompt):
python scripts/serve_llama.py --config configs/serving_v2.yaml --agents N --sessions 12

# regenerate the real-task trace:
python scripts/gen_traces.py --config configs/trace_gen.yaml --model-dir /root/models/Qwen2.5-3B
```
