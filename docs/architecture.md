# Architecture

The system reproduces the *single-engine* part of AgentServe: prefill/decode disaggregation
inside one engine, with a shared KV cache, driven by continuous batching and prefix caching.

## Components

```
                     ┌──────────────────────────────────────────────┐
                     │               AgentServe engine               │
                     │                                              │
   agent requests ──▶│  Scheduler (phase.classify -> QD/QP queues)    │
                     │     cold_prefill / resume_prefill / decode    │
                     │                                              │
                     │  ┌─────────┐            ┌─────────┐           │
                     │  │ Prefill │  streams   │ Decode  │           │
                     │  │  engine │ ◀────────▶ │  engine │           │
                     │  │  (ctx A)│  cudaEvent │  (ctx B)│           │
                     │  └────┬────┘   +mutex   └────┬────┘           │
                     │       │                       │                │
                     │       └────── SHARED KV ──────┘                │
                     │        (v_cells_impl + K/V tensors)            │
                     └──────────────────────────────────────────────┘
```

## 1. Shared KV across two contexts (the crux)

`llama_context` is not re-entrant, and historically its KV cache is private. We patch llama.cpp so
a **decode context `B` shares the KV pool of a prefill context `A`**:

- `llama_kv_cache` already carries `llama_kv_cache * other`, a shared `v_cells_impl`
  (`shared_ptr`), and a `layer_share_cb`.
- **`llama-model.cpp`** (Qwen dense branch): pass `params.mem_other` to the KV cache and a
  `share` callback (`il -> il`), so `B`'s K/V tensors point at `A`'s.
- **`llama-context.cpp`**: propagate `params.ctx_other` so non-special architectures can set it.
- **`llama-kv-cache.cpp`**:
  - `apply_ubatch`: remove `if (other) return;` so the decode context can **write** its decode
    cells into the shared pool (otherwise the mirror only reads).
  - `seq_pos_min/max`: read the shared cells directly instead of delegating to `other`.
- **`ggml-cuda/common.cuh`**: `as_sidx()` keys cuBLAS handles / workspaces / pools by the active
  green-stream index, fixing per-stream resource binding across two green streams.

With these, `B` reads and appends to `A`'s KV **without copying**, which is the paper's "avoiding
inter-engine KV transfers" goal.

See [`patches/`](../patches/) for the full diffs.

## 2. P/D concurrency

- Prefill runs on a prefill CUDA stream; decode on a decode stream.
- A `cudaEvent` per session orders GPU work so decode waits until the session's prefill completed.
- A mutex protects the shared cell bookkeeping (find-slot / apply) between the two contexts.

## 3. Continuous batching

Instead of decoding one session at a time, the decode thread packs **one token per ready session**
into a single `llama_decode(B, batch)`. This amortizes kernel launch and keeps SMs busy. Batched
prefill likewise packs all cold prompts into one `llama_decode(A, batch)`.

## 4. Prefix caching

Agent sessions share a long system prompt. We tokenize each cold prompt, find the common token
prefix, prefill it **once**, then share those KV cells across sequences with
`llama_memory_seq_cp(0 → i)`; each session then prefills only its unique instruction. This drops
cold TTFT to roughly the instruction-prefill time.

---

## Engine entry point

[`src/runtime/agentserve_engine.cpp`](../src/runtime/agentserve_engine.cpp) drives the full flow:

1. Load model; create `A` (prefill) and `B` (decode, `ctx_other=A`).
2. Batch cold prefill (with optional prefix cache) on `A`.
3. Loop: batched decode on `B`; on phase end, schedule resume prefill on `A`.
4. Record `TTFT_cold`, `TTFT_resume`, `TPOT` per session/phase to an events JSONL.

Measurement uses `cudaDeviceSynchronize` after `llama_decode` because llama.cpp's decode is
asynchronous — without a sync, wall-clock times are misleading.

## Phase / token model

Trace format per session: `sid|cold_b64|d0|app1_b64|d1|app2_b64|d2|...`

- `cold_b64` — the cold (system + task) prompt.
- `dK` — number of decode tokens in the K-th phase.
- `appK_b64` — K-th resume (tool output) appended to the prompt.

Phase index `i`:
- prefill input: `i==0 ? parts[1] : parts[1+2*i]`
- decode count: `parts[2+2*i]`
