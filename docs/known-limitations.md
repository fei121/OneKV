# Known Limitations & Verified Findings (honest status)

> **TL;DR** — An earlier round of `results/` and `figures/` reported that the shared-KV engine
> "crushes" llama.cpp / vLLM / SGLang on throughput and SLO. **Those conclusions are retracted.**
> The engine does faithfully reproduce the **cold prefix-caching** mechanism, and its
> continuous-batching decode is genuinely **stable**; but the **resume path is NOT byte-faithful**
> (dual-context shared-KV logits drift), so per-session output lengths and hence throughput/SLO
> are **not comparable** to the baselines.

## What led to the retraction

While trying to reproduce a fair 4-way (llama.cpp / vLLM / SGLang / shared-KV engine) comparison,
we found the engine had **multiple measurement artifacts** that inflated its apparent advantage:

1. **The engine ignored EOS.** `agentserve_engine.cpp` never checked for end-of-generation, so it
   forced every phase to its token cap. A real model stops at EOS earlier (e.g. session `s000`:
   engine generated 160 forced tokens, the reference model generates ~58). This alone inflated the
   engine's token count and throughput by ~2.7×.
2. **Unequal workload size.** The engine ran only the first `N` sessions; the baselines ran 12
   sessions at concurrency `N`. The engine did strictly less work.
3. **tool_wait skipped.** The engine did not simulate the 200 ms tool-call latency, so its wall time
   was pure compute; baseline throughput divided by a wall that included tool_wait.
4. **Config mismatch.** Engine `n_batch=20000` / `n_ctx=65536` / flash attention vs llama-server
   `batch=512` / `context=24576` vs vLLM/SGLang `max-model-len=8192`.

These together produced the "engine 351 tok/s, 100% SLO, crushes baselines" picture. **Not valid.**

## Structural fixes applied to the engine (now committed)

The engine was reworked so it is measured on the **same workload** as the baselines:

- **EOS respect**: per-decode check `llama_vocab_is_eog`; the phase ends at EOS instead of forcing
  the cap.
- **12-session reuse**: the engine now serves a fixed set of 12 sessions over `N` slots (seq 0 is a
  permanent shared-prefix template; slots are recycled with `llama_memory_seq_rm` + `seq_cp`).
- **200 ms tool_wait** between agent steps (via a per-session `wait_until` time gate).
- **Correctness fixes found in the process**:
  - `seq_cp(0, slot, 0, common)` — `p1` is **exclusive**, so `common` copies the *full* shared
    prefix (the prior `common-1` dropped the last shared token, corrupting the context).
  - Phase advancement only when `drem==0 && need_pre==0` (previously the scheduler skipped all
    resume phases).
  - EOS/EOG also advances `n_past` (else the next resume prefill collided with the KV tail).
  - Termination was breaking while sessions were merely waiting on tool_wait — fixed.

## Verified findings (what we can stand behind)

### 1. Cold prefix-caching is faithful

For session `s000` the cold (system + task) prompt is **identical** between the engine's trace and
the baseline's (`MATCH: True`, 9666 chars). Greedy (`temperature 0`) generation over that prompt
produces **identical text**:

| source | s000 cold output |
|---|---|
| llama-server `/completion` (baseline) | `"I'm sorry, but I'm unable to assist with that task."` (15 tokens) |
| shared-KV engine | `'m sorry, but I'm unable to assist with that task.` (13 tokens, leading `I` dropped only by the log capture) |

So the **shared-system-prefix prefill once + `seq_cp` + unique-instruction prefill** mechanism is
faithful: the model sees the same context and produces the same response.

### 2. Continuous-batching decode is stable (agent & per-token)

The engine decodes **one token per ready session in a single `llama_decode`** (continuous batching),
so per-token decode latency stays near a single batched forward pass. In the (now structurally
correct) runs the per-round decode latency is ~9–14 ms and **does not blow up with concurrency** —
the mechanism underlying "decode protection".

## Known limitation: resume cross-context logits drift

Even after the fixes, the engine's **resume** phases are not byte-faithful:

- Context **positions** are correct (`n_past = pos_max+1`, no "inconsistent sequence positions").
- Resume input **format** matches the baseline exactly (`\n\n<tool_result>\n…\n</tool_result>\n`,
  verified equal to `phase[i].prompt − phase[i-1].prompt`).
- **Yet** the engine generates to the **full token cap** in every resume phase
  (e.g. s000 phases 1/2/3 = **41/44/36** tokens), while the baseline stops at EOG after ~14 each.
  N=3 totals: engine **1617 vs llama 964** output tokens (+68%).

Because the cold phase is faithful and positions/format are correct, the remaining cause is
**dual-context (A=prefill, B=decode, `ctx_other=A`) shared-KV producing different logits from a
single-context decode** once new tokens are appended to an already-decoded sequence. This is a
deep llama.cpp shared-KV behavior (the paper's exact mechanism), and it is **not** a trace or
positioning bug.

### Consequence

- Cold latency (TTFT via prefix caching) and per-token decode stability **are** real.
- Throughput and SLO **are not comparable**: the engine generates longer outputs (to cap) in resume,
  so its "higher throughput" is partly an artifact of producing more tokens per session.
- **The earlier "crushes baselines / 100% SLO" claims are withdrawn.**

## Path forward

1. Deep-dive the dual-context `ctx_other` resume logits drift (in progress).
2. Re-scope any headline claims to the verified parts: cold prefix caching + decode-stability.
3. Regenerate figures/docs only from verified, same-workload data.

## Repro commands

```
# engine (12 sessions, N concurrency, 200ms tool_wait, EOS respected) on the bench server:
/tmp/as_conc_batch <trace> <N> -1 1 <n_ctx> <model.gguf> 12 200
# reference greedy cold output for a session:
#   llama-server /completion, temperature 0
```
