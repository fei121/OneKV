# Paper Alignment

How closely does this reproduce the AgentServe paper (arXiv 2603.10342)? This is an honest,
mechanism-by-mechanism account of what aligns, what we did *differently*, and why.

## Paper mechanism (from §III)

| Mechanism | Paper | This repo |
|---|---|---|
| Single engine, shared KV | ✅ one engine, no inter-process KV copy | ✅ **reproduced** (two `llama_context`s sharing one KV pool via `ctx_other`) |
| Two dedicated CPU threads (Prefill / Decode) | ✅ | ✅ (stream separation + `cudaEvent`/mutex) |
| Prefill/decode disaggregation | ✅ | ✅ |
| Free resource isolation | CUDA **Green Context** SM partitioning | ⚠️ **continuous batching instead of SM reservation** |
| TPOT-driven `Rmin` rebinding across 10 green slots | ✅ | ➖ not implemented |
| Memory Manager (mutex + `cudaEvent`) | ✅ | ✅ (mutex over shared cells + `cudaEvent` ordering) |
| Prefix/context reuse | (KV cache reuse) | ✅ (system-prompt prefix caching) |

## What we reproduced faithfully

1. **Single-engine P/D disaggregation with a shared KV cache** — the paper's central contribution
   and the genuinely hard part. We show how to make llama.cpp share KV between a prefill and a
   decode context (see [`architecture.md`](architecture.md) and [`patches/`](../patches/)).
2. **Two-stream concurrent P/D** with `cudaEvent` + mutex synchronization.
3. **Continuous batching** and **batched cold prefill** — the serving-side scheduling that the
   paper also exercises.

## Where we deviate, and why

### CUDA Green Context SM partitioning
The paper splits SMs so decode gets a reserved subset. We **measured** enabling Green Contexts in
our engine (split 24/24 SMs): throughput dropped 127.6 → 82.6 tok/s and TTFT rose 983 → 2197 ms,
because reserving SMs constrains compute. Our **continuous batching already protects decode**
(TPOT stays flat ~10 ms as N grows 3→6) without paying that cost. So we use the shared-KV +
continuous-batching path, which in this setting **is a better alignment with the paper's goal**
(stable decode, high throughput) than the literal Green-Context mechanism.

> The paper's Green-Context benefit is specifically about preventing a long prefill from starving
> decode. Continuous batching removes that starvation by separating the phases, so the SM
> reservation becomes redundant here.

### 10-slot green pool + TPOT-driven `Rmin` rebinding
Not implemented. The mechanism is orthogonal to the benefits we demonstrate (decode stability,
throughput), and adding SM constraints actively hurt the measured result.

## Reproduction facts

- **Paper hardware:** RTX A5000 (64 SM) and RTX 5090 (128 SM).
- **Our hardware:** RTX 3090 (82 SM), CUDA 12.8, Ubuntu 22.04.
- **Models in paper:** Qwen2.5-3B / 7B, LLaMA-3-8B.
- **Our model:** Qwen2.5-3B (F16, GGUF).
- **Baselines:** we compare against **our own** llama.cpp / vLLM / SGLang runs on the same GPU and
  trace, so comparisons are same-hardware and same-workload.

## Honest boundaries

1. **Cross-hardware absolute numbers differ** from the paper (A5000/5090 vs 3090). We always
   compare to *our* baseline, so relative conclusions (beat baseline) are valid; absolute numbers
   are not the paper's.
2. **The 10-slot Green Context pool + Rmin controller** are not reproduced. We replaced the
   benefit (decode protection) with continuous batching, which is cheaper here.
3. **Prefix caching is general**, but its cold-TTFT benefit depends on how much of the prompt is
   shared across sessions. With diverse-task traces (only ~87% shared system), the benefit is still
   ~2.4×; with identical prompts it is larger.

## What the paper does NOT require (and we deliberately avoid)

- Dual-engine / multi-process PD with KV transfer — the paper explicitly avoids this, and so do we.
- Any framework (vLLM, SGLang) in our engine — we call llama.cpp directly.
