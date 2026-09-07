# llama.cpp patches

These patches make a single llama.cpp `llama_context` share its KV cache with a second context,
which is the enabler for single-engine P/D disaggregation.

| Patch | File | Purpose |
|---|---|---|
| `llama-model.cpp.patch` | `src/llama-model.cpp` | Qwen dense branch passes `mem_other` + a `share` callback so a decode context's K/V tensors point at the prefill context's. |
| `llama-context.cpp.patch` | `src/llama-context.cpp` | Propagate `params.ctx_other` to `cparams.ctx_other` (was hard-coded `nullptr` for non-special archs). |
| `llama-kv-cache.cpp.patch` | `src/llama-kv-cache.cpp` | `apply_ubatch`: allow the mirror to write cells into the shared pool; `seq_pos_min/max`: read shared cells directly. |
| `ggml-cuda-common.cuh.patch` | `ggml/src/ggml-cuda/common.cuh` | `as_sidx()` keys cuBLAS handles/workspaces/pools by the active green-stream index. |

## Apply

The files in this directory are **full patched copies** of the llama.cpp source files. To apply to a
clean checkout, diff them against the upstream file, e.g.:

```bash
cd /path/to/llama.cpp
diff -u src/llama-model.cpp.orig src/llama-model.cpp.patch   # review
cp patches/llama-model.cpp.patch src/llama-model.cpp
# ... repeat, then rebuild llava + ggml-cuda
```

## Verify

After patching + rebuilding, a minimal sanity check: create context `A`, prefill a prompt, create
context `B` with `ctx_other=A`, and decode on `B` — it must produce the same tokens as decoding on
`A` alone (`iso8b` check).
