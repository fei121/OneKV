# Pitfalls & Service-Startup Notes (lessons learned)

A running log of issues hit while building/benchmarking the AgentServe shared-KV reproduction on the
RTX 3090 bench server. Each entry: **symptom → root cause → fix**. Add to this as you hit new ones.

> Server: `sshpass -p '<pass>' ssh -p 22482 root@connect.nmb2.seetacloud.com`.
> Key paths: models `/root/models/Qwen2.5-3B` (HF), `/root/autodl-tmp/models/*.gguf`; exp root
> `/root/autodl-tmp/exp`.

---

## 1. Model path & tokenizer

- **Symptom:** `AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-3B')` raises
  `HFValidationError: Repo id must be in the form 'repo_name'...` or `OSError: Can't load the
  configuration of '/root/autodl-tmp/models/Qwen2.5-3B'`.
  - **Root cause:** the HF model dir is **`/root/models/Qwen2.5-3B`**, not
    `/root/autodl-tmp/models/Qwen2.5-3B`. The old path was removed.
  - **Fix:** use `/root/models/Qwen2.5-3B` (vLLM/SGLang/tokenizer). Add `local_files_only=True` to
    `AutoTokenizer.from_pretrained`, else it tries to fetch from HF and fails.
- **Symptom:** model does not emit `<tool_call>`; it outputs a natural-language plan regardless of
  prompt (ReAct text or native `tools`).
  - **Root cause:** `/root/models/Qwen2.5-3B` is the **BASE** model (`eos_token_id=151643` =
    `<|endoftext|>`), **not Instruct/Chat**. Base Qwen is not tool-calling tuned.
  - **Fix:** use `Qwen2.5-3B-Instruct` (HF + GGUF) for true function calling; or accept plan-style output.

---

## 2. Service startup (vLLM / SGLang / llama-server)

- **Symptom:** `RuntimeError: vllm server not ready` (backends.py `VllmBackend.start`).
  - **Root cause A:** GPU already held by a leftover server (e.g. a diagnostic `llama-server` on
    8083, or a `VLLM::EngineCore`). vLLM fails to allocate.
  - **Fix:** free the GPU, then re-run. Find the holder:
    `nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader`, then
    `kill -9 <pid>`.
  - **Root cause B:** wrong `model_path` (see §1) → vLLM crashes on config load immediately.
- **Symptom:** vLLM/SGLang startup logs are invisible (backends.py used `subprocess.DEVNULL`).
  - **Fix:** point `Popen(..., stdout=open('/tmp/<backend>_backend.log','w'), stderr=STDOUT)` and
    raise the `/health` retry count.
- **Symptom:** SGLang fails at flashinfer CUDA-graph capture (JIT can't find `ninja`).
  - **Fix:** `apt install -y ninja-build` (system) so flashinfer can JIT.
- **Symptom:** `pkill -f llama-server` / `pgrep -f llama-server` kills the SSH command itself →
  exit 255 with no output.
  - **Root cause:** `-f` matches the *current* bash command line because the command string contains
    "llama-server" (e.g. a redirect path). Self-match.
  - **Fix:** do not `pkill -f` a pattern that appears in your own command. Use a python script that
    parses `ps -eo pid,args` and kills by matching a distinctive arg, or kill by GPU pid.
- **Symptom:** `llama-server` N=10 returns 0 tokens / empty metrics.
  - **Root cause:** `context_length=24576` with `--parallel 10` → **2457 tokens/slot** < the
    ~2700-token cold prompt → every slot overflows.
  - **Fix:** raise `context_length` (e.g. 49152/65536) so `ctx/N` ≥ cold prompt.
- **Symptom:** llama.cpp baseline `cache_prompt` collapses multi-phase agents (phase 2/3 → 1 token).
  - **Root cause:** the phase prompts don't contain the model's own previous output (it's only in the
    slot KV); after a long generation the slot KV diverges from the next prompt, so llama-server
    returns a degenerate 1-token reply. Happens even in isolation.
  - **Fix:** build a **self-contained** prompt: `prev_full_prompt + prev_model_output + <tool_result>`.

---

## 3. Engine (`agentserve_engine.cpp`) bugs

- **Symptom:** engine throughput absurdly high (e.g. 351 tok/s).
  - **Root cause:** the engine **ignored EOS** and forced every phase to the token cap (a term ends at
    EOS). Fix: check `llama_vocab_is_eog(v, t)` after argmax.
- **Symptom:** "inconsistent sequence positions" / context corruption.
  - **Root cause:** `llama_memory_seq_cp(mem, 0, dst, 0, common-1)` — `p1` is **exclusive**, so it
    copies only `common-1` cells and **drops the last shared token**.
  - **Fix:** `seq_cp(mem, 0, dst, 0, common)` to copy the full shared prefix.
- **Symptom:** sessions skip all resume phases (only cold output, no `TTFT_resume` events).
  - **Root cause:** in the scheduler loop, phase advancement was `if(drem==0)`, which re-fires while
    the session is waiting on a tool_wait / resume prefill → phase counter jumps to the end.
  - **Fix:** only advance when `drem==0 && need_pre==0`.
- **Symptom:** resume prefill collides with the KV tail (`Y == X`, needed `Y = X+1`) after EOS.
  - **Fix:** on EOG, also `n_past++` (the token was fed; advance past it).
- **Symptom:** engine exits early while all sessions are waiting on tool_wait.
  - **Root cause:** termination `if(!any_progress && !any_pending) break;` broke while waiting.
  - **Fix:** break only when **no active && no pending**; otherwise sleep ~1ms and loop.
- **Symptom:** compile errors using the memory API.
  - `llama_memory_seq_rm` requires `(mem, seq, p0, p1)` — clear all with `(mem, seq, 0, -1)`.
  - `llama_memory_seq_pos_min/max` take `llama_memory_t`, so do `llama_get_memory(ctx)` first.
  - `llama_token_to_piece` needs 6 args: `(vocab, token, buf, len, lstrip, special)`.

---

## 4. Baseline harness (`serve_llama.py` / `serve_backend.py`)

- **Symptom:** phase prompts omit the model's previous output → incoherent multi-phase.
  - **Fix:** rebuild each phase as `full_prompt + prev_model_output + (phase[i].prompt − phase[i-1].prompt)`,
    send to `cache_prompt`; this keeps prefix caching AND a coherent conversation.
- **Symptom:** vLLM/SGLang TPOT ≈ 0.
  - **Root cause:** `submit()` did `return list(self.stream_completion(...))` → all tokens collected
    at once, so all `TOKEN_EMIT` share one timestamp.
  - **Fix:** `submit()` must `yield from self.stream_completion(...)`.
- **Symptom:** `serve_backend.py` uses a stale model path (old `/root/autodl-tmp/models`).
- **Symptom:** SGLang looks like the worst backend (throughput DROPS as N grows, cold TTFT
  explodes to 1–2 s), yet vLLM is fine.
  - **Root cause:** `SglangBackend.start()` hard-coded `--max-total-tokens 8192`. In SGLang this is the
    **total KV-cache pool budget**, not a per-sequence limit. With N concurrent multi-phase sessions
    (~3 k-token cold prompts) the pool saturates (`token usage` → 0.95), so only 1–2 requests run and
    the rest queue (`#running-req: 1–2`, `#queue-req: 4–5`, `#pending-token ~14 k`).
  - **Fix:** give SGLang a pool big enough for N sessions, e.g. `--max-total-tokens 49152`.
- **Symptom:** vLLM refuses to start with `--max-model-len 49152`.
  - **Root cause:** `--max-model-len` is a **per-sequence** cap and must not exceed the model's native
    `max_position_embeddings` (Qwen2.5-3B = 32768). vLLM refuses with
    `VLLM_ALLOW_LONG_MAX_MODEL_LEN` unless explicitly overridden (which risks RoPE NaN/OOB).
  - **Fix:** use `--max-model-len 32768` (native max) for Qwen2.5.

---

## 5. Benchmark methodology

- Always drive the **same workload** across backends: same sessions, same `N`, same `tool_wait`, same
  EOS behavior. (The original engine ran fewer sessions and skipped tool_wait → not comparable.)
- **tool_wait**: the engine must simulate the 200 ms tool wait (per-session `wait_until` gate), or its
  wall time is compute-only and throughput is overstated vs baselines that include tool_wait.
- **Context must be adequate & semantically matched across backends**: engine `n_ctx` and SGLang
  `--max-total-tokens` are **pool** budgets (use 49152 for 3B); vLLM `--max-model-len` and llama
  `context_length` are **per-sequence/per-context** (use 32768 / 65536). Under-sized pools (SGLang
  8192) starve concurrency and wreck the comparison. See `src/agentserve_repro/backends.py`.
- `metrics.py`: `throughput_excl_tool_wait_tokens_per_s` is **broken for concurrent** workloads — it
  *sums* per-session tool_wait (which, under concurrency, exceeds wall → denominator ~0 → huge/999…).
  Use the union-of-idle-intervals approach, or report `throughput_tokens_per_s` with the caveat.
- **Isolated profile for SLO τ**: llama-server `cache_prompt` reuses the shared system prefix across
  *serial* sessions, so an isolated N=1 TTFT reads ~25 ms (cached), not the true uncached ~372 ms.
  Use the **uncached** cold value for τ.

---

## 6. Trace generation (`trace.py`)

- **Symptom:** model refuses ("I'm sorry, I'm unable to assist...") and repeats random tool outputs.
  - **Root cause:** the cold-prompt user message was a **generic placeholder**
    (`USER_TASK = "Retrieve and synthesize the requested information..."`), so the model had no real task.
    Resume appends were also random concatenations, unrelated to the task.
  - **Fix:** parse the real ToolBench request with `_extract_request` on the `instruction`'s
    `Request:` block; pair task↔schema↔output; use the real output as the resume append.
- **Symptom:** data format of ToolBench `instruction`:
  `API doc:\n{...}\n\nRequest:\n{'tool_name':..., 'api_name':..., 'tool_input':'{...}'...}`.
  The actual task is the `Request:` block; the schema is the `API doc:` block.
- **Flatten jsonl → `sessions.txt`:** `sid|cold_b64|d0|app1_b64|d1|...`. `app*` is the **bare** tool
  append (the engine wraps it in `<tool_result>`); `d*` is the decode-token target.

---

## 7. Command-line / shell gotchas

- Avoid `cmd | tail` for long runs — the pipe can break (exit 255). Redirect to a file and `tail` it.
- Her-here-docs with embedded quotes / f-strings break under `zsh`/`bash -c`. Write a `.py` file and
  run it, or escape carefully.
- Long server runs: `nohup ... > log 2>&1 &` then poll; or use a persistent PTY session.
- SSH + `kill $(pgrep -f X)`/`pkill -f X` **self-matches** when your command text contains `X`. Kill by
  GPU pid (`nvidia-smi --query-compute-apps`) or via a python ps-parser instead.
