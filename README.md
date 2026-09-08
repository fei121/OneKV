# OneKV

**Single-engine shared-KV serving for agentic AI on a consumer GPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![GPU](https://img.shields.io/badge/Hardware-RTX%203090-9cf)
![Lang](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20Python-informational)

OneKV is a **single-GPU inference serving engine** for agentic workloads (ReAct / Plan-and-Execute).
Its core idea is to **disaggregate prefill and decode *inside one engine* while sharing a single KV
cache** — no cross-process KV transfer — so decode latency stays flat even while a long prefill runs.

It runs on a consumer GPU (RTX 3090) with Qwen2.5-3B / 7B, and is benchmarked side-by-side against
**llama.cpp / vLLM / SGLang** on the same hardware and the same real-task trace.

---

## Highlights

- **Single-engine P/D disaggregation + shared KV** — two `llama_context`s (A = prefill, B = decode)
  share one KV pool (`ctx_other = A`); no KV copy between engines.
- **Continuous batching + prefix caching** — keeps **TPOT p95 flat** and **cold TTFT low & flat** under
  concurrency (the two pain points of agent serving).
- **Consumer-GPU ready** — RTX 3090, Qwen2.5-3B / 7B, ~24 GB.
- **Honest, reproducible 4-way benchmark** — same harness, same trace, same `N`, serial measurement.
- **Same-hardware comparison** — conclusions are *relative* to our own baselines, not the paper's.

> **Scope:** this measures **serving performance**, not agent *quality*. The model is Qwen2.5-**BASE**
> (no native tool-calling), so outputs are plan-style text.

---

## Performance

### Qwen2.5-3B (4-way, N=3→6, tool_wait=0)

| Paradigm | Metric | OneKV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|---|
| **ReAct** | throughput (tok/s) | 149.8→175.2 | 131.1→145.0 | **167.7→222.8** | **164.2→226.3** |
| | TPOT p95 (ms) | **11.9→13.5** | 60.0→72.7 | 19.5→24.0 | 24.0→27.9 |
| | cold TTFT p50 (ms) | **308.6→367.6** | 439.5→737.8 | 302.7→621.4 | 302.2→454.8 |
| **P&E** | throughput (tok/s) | 162.5→192.2 | 145.1→147.7 | **170.4→259.1** | **179.5→243.1** |
| | TPOT p95 (ms) | **11.8→13.5** | 50.2→70.7 | 22.0→38.7 | 24.8→31.4 |
| | cold TTFT p50 (ms) | **316.5→383.8** | 412.3→773.2 | 343.0→603.4 | 335.0→478.2 |

![ReAct 3B](figures/react-4way-nscale.png)
![P&E 3B](figures/pe-4way-nscale.png)

### Qwen2.5-7B (4-way, N=3→6, tool_wait=0)

| Paradigm | Metric | OneKV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|---|
| **ReAct** | throughput (tok/s) | 84.6→103.3 | 81.9→92.4 | 88.8→123.6 | 90.3→125.2 |
| | TPOT p95 (ms) | **20.8→22.3** | 65.0→125.8 | 37.9→46.0 | 31.3→44.6 |
| | cold TTFT p50 (ms) | **524.2→625.0** | 800.6→1179.0 | 591.1→1214.5 | 590.5→958.6 |
| **P&E** | throughput (tok/s) | 93.2→115.8 | 92.1→108.1 | 94.8→131.3 | 97.2→134.5 |
| | TPOT p95 (ms) | **20.4→22.1** | 71.5→111.1 | 21.6→110.0 | 25.6→48.5 |
| | cold TTFT p50 (ms) | **540.9→646.0** | 803.2→1252.2 | 648.5→1036.0 | 782.8→945.2 |

![ReAct 7B](figures/react7-4way-nscale.png)
![P&E 7B](figures/pe7-4way-nscale.png)

### Bottom line

The OneKV engine is the **latency-stability leader**:

- **TPOT p95 stays flat** (3B ~12–14 ms, 7B ~20–22 ms) while llama.cpp explodes (50–126 ms) and
  vLLM/SGLang rise (20–110 ms).
- **Cold TTFT is lowest & roughly flat** (prefix-cache amortization; 3B ~310–385 ms, 7B ~520–650 ms)
  while every baseline degrades with N.
- **Throughput is not its strong suit**: `vLLM ≈ SGLang > OneKV > llama.cpp`. It trades a little
  throughput for *stable TTFT/TPOT* — the core goal for agent serving.

> **Fairness note.** vLLM/SGLang are benchmarked with **adequate context** (vLLM `--max-model-len
> 32768`, SGLang `--max-total-tokens 49152`). An earlier `--max-total-tokens 8192` starved SGLang's KV
> pool and made it look worst — that was a harness bug, not an SGLang limitation. A context-window
> sweep ([`docs/benchmark/methodology.md`](docs/benchmark/methodology.md) +
> [`figures/context-windows-n6.png`](figures/context-windows-n6.png)) shows every backend is on a
> plateau once context is adequate, so the gaps are real, not a config artifact.

---

## Architecture

```
             ┌──────────────────────────────────────────────┐
             │           one model · one KV cache          │
             │        (llama_kv_cache, ctx_other = A)      │
             │                                              │
             │   seq 0 = shared-prefix template             │
             │        └─ copied to each session (seq_cp)   │
             │   seq 1..N = live sessions                   │
             └──────────────────────────────────────────────┘
                   ▲                    ▲
        write K/V │            read K/V │ + append new token
       ┌──────────┴─────────┐  ┌────────┴─────────┐
       │ context A (prefill) │  │ context B (decode) │
       │ llama_decode(A,...) │  │ llama_decode(B,...) │
       │ CUDA stream: pre     │  │ CUDA stream: dec     │
       └──────────────────────┘  └──────────────────────┘
```

- **A (prefill)** batches each session's *cold* prompt (shared system prefix prefilled once on `seq 0`
  and `llama_memory_seq_cp`'d to each session) and *resume* prompts (tool results) into multi-sequence
  `llama_decode(A)` calls.
- **B (decode)** does continuous batching — one token per ready session per `llama_decode(B)` — reading
  the KV that A wrote and appending each generated token.
- The `g_kv` mutex only serializes host-side cell bookkeeping; the two kernels run on separate CUDA
  streams.

### Making two contexts share one KV (the hard part)

A single llama.cpp `llama_context` is not re-entrant and its KV cache is bound to it. OneKV patches
llama.cpp (see [`patches/`](patches/)) so two contexts share one pool:

| File | Change |
|---|---|
| `llama-model.cpp` | Qwen branch passes `mem_other` + a share callback → decode K/V points at prefill K/V. |
| `llama-context.cpp` | Propagate `params.ctx_other` → `cparams.ctx_other`. |
| `llama-kv-cache.cpp` | `apply_ubatch` lets the mirror write the shared cells; `seq_pos_min/max` read shared cells. |
| `ggml-cuda-common.cuh` | `as_sidx()` keys cuBLAS handles/workspaces/pools by the active stream index. |

Engine source: [`src/runtime/onekv_engine.cpp`](src/runtime/onekv_engine.cpp).

---

## Install

### 1. Hardware / OS
NVIDIA GPU (tested **RTX 3090**, 24 GB), **CUDA 12.8**, Ubuntu 22.04.

### 2. Models
All four artifacts (Qwen2.5-3B/7B × GGUF/HF) and their server paths + SHA-256 are in
[`configs/models/models.yaml`](configs/models/models.yaml).

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
apt update && apt install -y aria2
export HF_ENDPOINT=https://hf-mirror.com
export HFD_DOWNLOADER="aria2c -x 16 -s 16 -k 1M"
hfd Qwen/Qwen2.5-3B-GGUF --include "*qwen2.5-3b-f16*.gguf"   # GGUF (engine + llama.cpp)
hfd Qwen/Qwen2.5-3B                                          # HF safetensors (vLLM/SGLang + traces)
```

### 3. Build llama.cpp with the patches
See [`docs/setup/llama-cpp-patch.md`](docs/setup/llama-cpp-patch.md) and
[`docs/setup/environment.md`](docs/setup/environment.md) for the exact source/build flags/pinned env.

```bash
cmake -DHF_ENABLED=OFF -DBUILD_UI=OFF -B build .
cmake --build build --target llama-cli llama-server
# copy the 4 patched files from patches/ over the corresponding source, then rebuild
```

### 4. Python package
```bash
pip install -e .   # requests, pyyaml, matplotlib, numpy, pytest
```

---

## Usage

```bash
# generate the unified 12-task trace (shared by ReAct & P&E)
make trace

# run the single-engine shared-KV runtime (A = # concurrent agents)
make engine A=3

# baselines (self-contained multi-phase prompt)
make serve-llama A=3 S=12
make serve-vllm  A=3 S=12
make serve-sglang A=3 S=12

# serial benchmark sweep (ReAct + P&E, N=3…6, one backend at a time)
make sweep

# plots
python scripts/plot_perparadigm.py      # per-paradigm 4-way (3B + 7B)
python scripts/plot_context_windows.py  # context-window sensitivity
```

---

## Reproducibility

The full server environment (hardware, CUDA, llama.cpp version + build flags, model SHA-256, conda
versions) is pinned in [`docs/setup/environment.md`](docs/setup/environment.md). Every backend is driven
by the same harness, the same unified 12-task set, the same `N` and `tool_wait`, and measured **serially**
(one backend at a time, GPU freed between runs).

---

## Documentation

| Doc | Content |
|---|---|
| [`docs/design/architecture.md`](docs/design/architecture.md) | System + engine + patch design. |
| [`docs/design/paper-alignment.md`](docs/design/paper-alignment.md) | Design decisions & differences vs. prior work. |
| [`docs/setup/environment.md`](docs/setup/environment.md) | Pinned environment for bit-for-bit reproduction. |
| [`docs/setup/llama-cpp-patch.md`](docs/setup/llama-cpp-patch.md) | The four shared-KV patches + build/verify. |
| [`docs/benchmark/methodology.md`](docs/benchmark/methodology.md) | Workload, backends, metrics + context-fairness rules. |
| [`docs/benchmark/results.md`](docs/benchmark/results.md) | Results, figures, how to reproduce. |
| [`docs/notes/known-limitations.md`](docs/notes/known-limitations.md) | Honest boundaries + methodology pitfalls. |
| [`docs/notes/pitfalls.md`](docs/notes/pitfalls.md) | Service-startup / benchmark lessons learned. |
| [`results/`](results/) | Clean per-paradigm 4-way tables. |
| [`REPORT.md`](REPORT.md) | Full written report (design, engine, results). |

---

## Honest limitations

1. **Absolute numbers are not cross-hardware comparable** — RTX 3090 (82 SM). We always compare against
   *our own* baselines on the same GPU and trace.
2. **The 10-slot green-context pool + TPOT-driven rebinding are not implemented.** We achieve the
   decode-protection benefit with **continuous batching**, which we measured to be cheaper (reserving
   SMs actually hurt on the 3090). See [`docs/design/paper-alignment.md`](docs/design/paper-alignment.md).
3. **BASE model, no native tool-calling** → this measures serving performance, not agent quality.
4. **Cross-model caveat:** both 3B and 7B are benchmarked with the same methodology; see the result
   tables above.

---

## License

MIT — see [`LICENSE`](LICENSE).
