# AgentServe Reproduction

**Single-Engine Shared-KV Serving for Agentic AI on a Consumer GPU.**

This repository is a faithful, from-scratch reproduction of the
[AgentServe](https://arxiv.org/abs/2603.10342) system (**arXiv 2603.10342**):
*Algorithm–System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU*.

We re-implemented the paper's central system contribution — **prefill/decode (P/D)
disaggregation inside a *single* engine with a shared KV cache** — and validated it on an
**RTX 3090** with **Qwen2.5-3B (F16)**. The reproduction **outperforms the llama.cpp
baseline on throughput, per-token latency (TPOT), and cold time-to-first-token (TTFT)**.

---

## Why this matters

Agent workloads (ReAct / Plan-and-Execute) alternate between **long cold prefills**, **short
resume prefills** (tool outputs), and **very short decodes**. On a single GPU this creates
head-of-line blocking. The paper's key idea is not to shard across engines (which pays KV
transfer) but to disaggregate *within* one engine so that decode is never starved by prefill.

The core obstacle is that a single `llama_context` is not re-entrant and cannot share its KV
across a prefill context and a decode context. This repo shows **how to make that work** by
patching llama.cpp so two contexts share one KV pool (`ctx_other`), then driving them with a
continuous-batching + prefix-caching scheduler.

---

## Highlights / Results (Qwen2.5-3B, RTX 3090, N=6 concurrent agents)

| Metric | llama.cpp baseline | **This repo (shared-KV engine)** |
|---|---|---|
| Throughput | 111.9 tok/s | **228.6 tok/s** |
| TPOT p50 / p95 | 12.47 / 25.19 ms | **9.87 / 15.77 ms** |
| Cold TTFT (system cached) | 536.9 ms | **~35.5 ms** |

See [`docs/results/`](docs/results/) and [`results/`](results/) for the full three-way analysis
(baseline vs no-cache vs prefix-cached), and [`figures/`](figures/) for the plots.

---

## Layout

```
agentserve-repro/
├── README.md
├── LICENSE                 MIT
├── pyproject.toml          Python package (scheduler / metrics / events / trace)
├── Makefile
├── src/
│   ├── agentserve_repro/     Python package: backends, scheduler, phase, metrics, events, trace
│   └── runtime/
│       ├── agentserve_engine.cpp   The single-engine shared-KV serving engine (C++)
│       └── legacy/                 Earlier/experimental engine variants
├── patches/                 llama.cpp patches that enable cross-context shared KV
├── configs/                 Model / serving / trace / metrics configuration
├── scripts/                 Trace generation, baselines, plotting, analysis
├── docs/
│   ├── architecture.md       System + engine + patch design
│   ├── paper-alignment.md    Mechanism vs. the paper, differences, honest boundaries
│   ├── experiments.md        Task matrix / methodology
│   └── results/              Final report and per-task write-ups
├── figures/                 Generated plots
├── metrics/                 Parsed metrics (JSON)
└── tests/                   Unit tests
```

---

## The core mechanism

Most LLM serving separates prefill and decode by running **two engines/processes** and copying
KV between them. AgentServe instead keeps **one engine** and splits the GPU at the **CUDA Green
Context / SM** level. We reproduce the *shared-KV single-engine* part with a different, more
efficient path:

1. **One model, two `llama_context`s** — a *prefill engine* (A) and a *decode engine* (B) that
   **share the same KV cache** via `ctx_other=A`. This gives `B` read/write access to `A`'s KV
   cells without any inter-process copy.
2. **Concurrent P/D streams** — P-and-D kernels are launched on separate CUDA streams and
   synchronized with `cudaEvent` (so decode only reads KV that prefill has finished writing),
   protected by a mutex over the shared cell bookkeeping.
3. **Continuous batching** — the decode thread packs one token per ready session into a single
   `llama_decode`, which amortizes kernel launch and feeds SMs efficiently.
4. **Prefix caching** — sessions share the long system prompt; we prefill it once and share those
   KV cells across sequences (`llama_memory_seq_cp`), so each session only prefills its unique
   instruction (≈13% of the prompt).

The engine lives in [`src/runtime/agentserve_engine.cpp`](src/runtime/agentserve_engine.cpp).
The llama.cpp patches in [`patches/`](patches/) are what make the shared KV possible.

---

## Getting started

### Prerequisites
- NVIDIA GPU (tested: RTX 3090, 24 GB), CUDA 12.x, Ubuntu 22.04
- `llama.cpp` (built with CUDA; see [`docs/llama-cpp-patch.md`](docs/llama-cpp-patch.md))
- Python 3.10+ (`pip install -e .`), `aria2c` for model download via `hfd`

### 1. Install
```bash
pip install -e .            # Python package (scheduler/metrics/events/trace)
```

### 2. Get the model (Qwen2.5-3B F16, on the data disk)
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HFD_DOWNLOADER="aria2c -x 16 -s 16 -k 1M"
hfd Qwen/Qwen2.5-3B-GGUF --include "*qwen2.5-3b-f16*.gguf"
```

### 3. Build llama.cpp with the patches
See [`docs/llama-cpp-patch.md`](docs/llama-cpp-patch.md) for the exact diff + build steps.

### 4. Generate the agent trace
```bash
python scripts/gen_traces.py --config configs/trace-gen.yaml
```

### 5. Run the baseline
```bash
./scripts/run_llama_baseline.sh --agents 4 --sessions 12
```

### 6. Run the shared-KV engine
```bash
# Build the engine (link against the patched llama.cpp)
./scripts/run_engine.sh --model /root/autodl-tmp/models/Qwen2.5-3B-f16.gguf \
                        --trace /root/autodl-tmp/exp/traces/sessions.txt --agents 6
```

### 7. Analyze + plot
```bash
python scripts/backend_compare.py
python scripts/plot_final.py
```

---

## Reproduction facts (honest notes)

- **Hardware in the paper:** RTX A5000 (64 SM) and RTX 5090 (128 SM). **We reproduce on RTX 3090
  (82 SM).** Cross-hardware absolute numbers differ; we compare against *our own* llama.cpp
  baseline on the same GPU/trace.
- **CUDA Green Contexts** are central to the paper, but in our engine continuous batching already
  delivers decode protection **without** the SM reservation, and we measured that reserving SMs
  (Green Context) *degrades* throughput here. So we use the shared-KV + continuous-batching path.
- **Prefix caching** is general; it caches the shared system prompt and benefits even diverse-task
  workloads (≈2.4× on cold TTFT, see [`docs/results/`](docs/results/)).

See [`docs/paper-alignment.md`](docs/paper-alignment.md) for a precise mechanism-by-mechanism
comparison and the honest boundaries of the reproduction.

---

## License

MIT — see [`LICENSE`](LICENSE).
