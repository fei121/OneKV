# Environment & Reproducibility

> This file records **exactly** the environment the experiments were run in (the remote RTX 3090
> box), so the results in `results/` / `figures/` can be reproduced bit-for-bit. Measured on
> **2026-09-07**; values captured from the running machine.

---

## 1. Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA **GeForce RTX 3090**, 24 GB (24576 MiB), Compute Capability **8.6**, **82 SMs** |
| Driver | NVIDIA 580.142 (CUDA 12.8) |
| CPU | Intel **Xeon Gold 6330** @ 2.00 GHz, 112 logical cores |
| OS | Ubuntu 22.04 (kernel `5.15.0-135-generic`), x86_64 |
| Host | AutoDL container (`autodl-container-…`) |

> The paper used RTX A5000 (64 SM) and RTX 5090 (128 SM). **Absolute numbers are NOT comparable
> across hardware**; all conclusions here are *relative* (engine vs. our own baselines on the same GPU).

## 2. CUDA / toolchain

| Item | Value |
|---|---|
| CUDA toolkit | **12.8** (`nvcc` release 12.8, V12.8.93) at `/usr/local/cuda-12.8` |
| g++ | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04) |

## 3. llama.cpp

| Item | Value |
|---|---|
| Version | **0.4.0-dev** (`LLAMA_VERSION_MAJOR=0`, `MINOR=4`, `PATCH=0`; `-dev` because not a release tag) |
| Library | `libllama.so.0.4.0` |
| Source | `/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp` (⚠️ **not a git repo** → no commit hash; version read from `CMakeLists.txt`) |
| Build dir | `…/third_party/llama.cpp/build` |

### CMake / CUDA build options (from `build/CMakeCache.txt`)

| Option | Value |
|---|---|
| `CMAKE_BUILD_TYPE` | `Release` |
| `CMAKE_CUDA_COMPILER` | `/usr/local/cuda-12.8/bin/nvcc` |
| `GGML_CUDA` | `ON` |
| `GGML_CUDA_FA` | `ON` (flash attention) |
| `GGML_CUDA_GRAPHS` | `ON` (CUDA graph capture) |
| `GGML_CUDA_F16` | `ON` |
| `GGML_CUDA_COMPRESSION_MODE` | `size` |
| `GGML_CUDA_NCCL` | `ON` |
| `GGML_CUDA_FORCE_CUBLAS` | `OFF` |
| `GGML_CUDA_FORCE_MMQ` | `OFF` |
| `GGML_CUDA_FA_ALL_QUANTS` | `OFF` |
| `GGML_CUDA_NO_PEER_COPY` | `OFF` |
| `GGML_CUDA_NO_VMM` | `OFF` |
| `GGML_NATIVE` | `ON` |
| `GGML_OPENMP` | `ON` |
| `LLAMA_BUILD_TESTS` | `ON` |

> Configure with `cmake -DHF_ENABLED=OFF -DBUILD_UI=OFF .` to skip the WebUI download, then
> `cmake --build build --target llama-cli llama-server` (see [`llama-cpp-patch.md`](llama-cpp-patch.md)).

## 4. AgentServe patches (shared-KV enabler)

The four modified llama.cpp source files are committed in [`patches/`](../../patches/). They are
**full patched copies**, not unified diffs — to reproduce, start from a **llama.cpp 0.4.0-dev**
checkout and copy each file over (or `diff -u upstream patched` per file).

| Patch file | Overwrites | Purpose |
|---|---|---|
| `patches/llama-model.cpp.patch` | `src/llama-model.cpp` | Qwen dense branch passes `mem_other` + a share callback so a decode context's K/V tensor points at the prefill context's. |
| `patches/llama-context.cpp.patch` | `src/llama-context.cpp` | Propagate `params.ctx_other` → `cparams.ctx_other` (was hard-coded `nullptr`). |
| `patches/llama-kv-cache.cpp.patch` | `src/llama-kv-cache.cpp` | `apply_ubatch`: let the mirror context write into the shared pool; `seq_pos_min/max`: read the shared cells. |
| `patches/ggml-cuda-common.cuh.patch` | `ggml/src/ggml-cuda/common.cuh` | `as_sidx()` keys cuBLAS handles/workspaces/pools by the active (green) stream index. |

> **Note on the overlap experiment** (branch `cold-prefill-overlap`): it additionally patched
> `ggml-cuda.cu` to expose the CUDA stream handle. That change was **reverted**; `main` uses only
> the four patches above.

## 5. Models

### GGUF (used by the C++ engine + llama.cpp baseline)

| Model | Path | Size | SHA-256 |
|---|---|---|---|
| Qwen2.5-3B-F16 | `/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf` | 6,178,316,800 B (~5.8 GiB) | `5bfca925442fbd1bc936801c10284f72a768d11dd4515bdbcd6055ac4c734054` |
| Qwen2.5-7B-F16 | `/root/autodl-tmp/models/Qwen2.5-7B-f16.gguf` | 15,237,852,960 B (~14.2 GiB) | `97ab3b8550c5123cbc328e197b00268360d7c7fd6fcc0294078768d41fda68b6` |

### HF safetensors (used by vLLM / SGLang backends + trace generation)

| Model | Path | Shards |
|---|---|---|
| Qwen2.5-3B (BASE) | `/root/models/Qwen2.5-3B` | `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` |
| Qwen2.5-7B (BASE) | `/root/models/Qwen2.5-7B` | `model-00001..00004-of-00004.safetensors` |

> **Model paths are also recorded in [`configs/models/models.yaml`](../../configs/models/models.yaml)** (the
> single source of truth for all four artifacts: Qwen2.5-3B/7B × GGUF/HF, with SHA-256).

### Download tooling

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
apt update && apt install -y aria2
export HF_ENDPOINT=https://hf-mirror.com
export HFD_DOWNLOADER="aria2c -x 16 -s 16 -k 1M"
hfd Qwen/Qwen2.5-3B-GGUF --include "*qwen2.5-3b-f16*.gguf"
```

> The models are **BASE** (not Instruct/Chat): no native tool-calling, so outputs are plan-style
> text. This means we measure **serving performance**, not agent *quality*.

## 6. Python / conda

conda envs live under `/root/autodl-tmp/conda_envs/`.

| Env | Python | Version pins |
|---|---|---|
| `vllm` | 3.12.14 | vllm **0.16.0**, torch **2.9.1+cu128**, transformers 4.57.6 |
| `sglang` | 3.12.14 | sglang **0.5.19**, torch **2.13.0+cu130** |

Project Python package (`pip install -e .`): `requests`, `pyyaml`, `matplotlib`, `numpy`, `pytest`
(see [`requirements.txt`](../../requirements.txt) / [`pyproject.toml`](../../pyproject.toml)).

## 7. Engine build (C++)

```bash
LLAMA_BUILD=/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp/build
g++ -std=c++17 -O2 \
  -I"$LLAMA_BUILD/../include" -I"$LLAMA_BUILD/../ggml/include" -I/usr/local/cuda/include \
  -o build/agentserve_engine src/runtime/agentserve_engine.cpp \
  -L"$LLAMA_BUILD/bin" -L/usr/local/cuda/lib64 \
  -lllama -lggml -lggml-cuda -lggml-cpu -lggml-base -lcuda -lcudart \
  -Wl,-rpath,"$LLAMA_BUILD/bin"
```

The engine links the **patched** `libllama.so` / `libggml-cuda.so` and drives two contexts
(`A`=prefill, `B`=decode, `B.ctx_other=A`) sharing one KV pool. The clean per-paradigm binary on
the server is `/tmp/as_conc_batch` (innovation-1).

> **Env for clean runs:** `AGENTSERVE_PREFILL_PCT` / `AGENTSERVE_DECODE_PCT` are **unset**
> (`env -u … `), which disables green-context stream routing.

## 8. Workload / traces

- **Unified 12-task set** shared by both paradigms: `data/unified_tasks.json`
  (server: `/root/autodl-tmp/exp/traces_unified/`).
- Trace files: `sessions_react.txt`, `sessions_plan_and_execute.txt` (+ `traces_7b/` for 7B).
- Table-I token distribution (per paradigm) in `distribution_summary.json`.
- Generation: `python scripts/gen_traces.py --config configs/trace-gen.yaml`
  (model dir `/root/autodl-tmp/models/Qwen2.5-3B`).

## 9. Reproduce the headline comparison

Same GPU, same trace, serial measurement (one backend at a time, GPU freed between runs),
`tool_wait=0`, 12 sessions, N=3…6.

```bash
# engine (innovation-1, prefix-cached):
/tmp/as_conc_batch <sessions_{react|plan_and_execute}.txt> <N> -1 1 <n_ctx> \
  /root/autodl-tmp/models/Qwen2.5-3B-f16.gguf 12 0
# llama.cpp baseline:
python scripts/serve_llama.py --config configs/serving_{react|pe}_u.yaml --agents N --sessions 12
# vLLM / SGLang:
python scripts/serve_backend.py --backend {vllm,sglang} --model-path /root/models/Qwen2.5-3B \
  --config configs/serving_{react|pe}_u.yaml --agents N --sessions 12
```

- Metrics: `metrics/perparadigm/{react,pe,react7,pe7}.json`.
- Figures: `scripts/plot_perparadigm.py`.

---

*Recorded 2026-09-07 from the RTX 3090 AutoDL box. llama.cpp has no git history on the box, so its
exact commit is unavailable; the version is pinned by `CMakeLists.txt` (0.4.0-dev) + the four
patches in [`patches/`](../../patches/).*
