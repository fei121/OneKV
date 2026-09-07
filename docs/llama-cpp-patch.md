# llama.cpp patches (single-engine shared-KV)

These four patches make a single llama.cpp `llama_context` share its KV cache with a second context,
which is the enabler for **single-engine prefill/decode disaggregation** in
[`src/runtime/agentserve_engine.cpp`](../src/runtime/agentserve_engine.cpp).

The patched files are committed as **full patched copies** in [`../patches/`](../patches/).

| Patch file | Overwrites | Purpose |
|---|---|---|
| `patches/llama-model.cpp.patch` | `src/llama-model.cpp` | Qwen dense branch passes `mem_other` + a `share` callback so a decode context's K/V tensors point at the prefill context's. |
| `patches/llama-context.cpp.patch` | `src/llama-context.cpp` | Propagate `params.ctx_other` → `cparams.ctx_other` (was hard-coded `nullptr` for non-special architectures). |
| `patches/llama-kv-cache.cpp.patch` | `src/llama-kv-cache.cpp` | `apply_ubatch`: let the mirror context write cells into the shared pool; `seq_pos_min/max`: read the shared cells directly. |
| `patches/ggml-cuda-common.cuh.patch` | `ggml/src/ggml-cuda/common.cuh` | `as_sidx()` keys cuBLAS handles/workspaces/pools by the active stream index (fixes two-stream CUDA resource conflicts). |

## Apply

Start from a **llama.cpp 0.4.0-dev** checkout (see [`environment.md`](environment.md)), then copy each
patched file over the upstream one (or `diff -u upstream patched` to review):

```bash
cd /path/to/llama.cpp
for f in llama-model llama-context llama-kv-cache; do
  cp /path/to/agentserve-repro/patches/$f.cpp.patch          src/$f.cpp
done
cp /path/to/agentserve-repro/patches/ggml-cuda-common.cuh.patch  ggml/src/ggml-cuda/common.cuh

# rebuild
cmake -DHF_ENABLED=OFF -DBUILD_UI=OFF -B build .
cmake --build build --target llama-cli llama-server
```

## Verify

The build must be configured with **CUDA + flash attention** (`GGML_CUDA=ON`, `GGML_CUDA_FA=ON`,
see [`environment.md`](environment.md)). A minimal sanity check: create context `A`, prefill a prompt,
create context `B` with `ctx_other = A`, and decode on `B` — it must produce **identical tokens** to
decoding on `A` alone (`iso8b`-style check).

## Not used: CUDA Green Context (SM partitioning)

The paper's Green Context (reserving a subset of SMs for decode) was **tried** via an
`AGENTSERVE_GREEN_PCT` stream (in `ggml_backend_cuda_context::stream`). On the RTX 3090 it **degraded**
throughput (127.6 → 82.6 tok/s; TTFT 983 → 2197 ms), so it is **not enabled** in the shipped engine.
See [`paper-alignment.md`](paper-alignment.md).
