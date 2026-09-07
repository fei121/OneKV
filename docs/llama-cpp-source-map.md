# llama.cpp 源码地图（当前 checkout，0.4.0-dev，已重构分文件）

扫描对象：`/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp`
说明：本阶段**仅建立地图，不改业务逻辑**。版本为重构后的结构（`src/` 下按职责分文件），与旧教程不同。

| 目标 | 实际文件:行 | 类/函数 | 调用链 |
|---|---|---|---|
| Request entry | `tools/server/server-http.cpp`（HTTP/路由）→ `tools/server/server-context.cpp:4241` | `server_routes::handle_completions_impl` + `server_http` | HTTP 请求 → 路由派发(`server-context.cpp:4882…5031`) → `handle_completions_impl` → 入 `server_queue` |
| Tokenization | `src/llama-vocab.cpp:4415` | `llama_tokenize` | `common/common.cpp:1879` 调 `llama_tokenize`；server 端在 `handle_completions_impl` 里 `tokenize` |
| Batch creation | `src/llama-batch.cpp:931,945` | `llama_batch_get_one` / `llama_batch_init` / `llama_batch_add` | server 组装 token → `llama_batch` → `llama_decode`；`src/llama-context.cpp:3581` 初始化 batch |
| Model context | `src/llama-context.cpp:3765` | `llama_new_context_with_model` / `llama_context` | server 初始化时创建；上下文含 KV cache + 计算图 |
| KV cache | `src/llama-kv-cache-dsa.cpp`、`llama-kv-cache-dsv4.cpp`、`llama-kv-cache-dsa-iswa.cpp` | `llama_kv_cache_dsa::clear/seq_cp/seq_keep/seq_add` 等 | server slot 管理 → `llama_kv_cache_seq_*`；`llama-context.cpp` 持有 KV cache |
| Decode call | `src/llama-context.cpp:4244` | `int32_t llama_decode(...)` | `llama_decode` → 构建 `ggml_cgraph` → `ggml_backend_graph_compute` → CUDA backend |
| CUDA backend | `ggml/src/ggml-cuda/ggml-cuda.cu`（`ggml_backend_cuda_context`,`ggml_backend_cuda_reg_*` ~5600） | `ggml_backend_cuda` | backend 注册 → `ggml_cuda_compute_forward` → 各 `ggml_cuda_op_*` |
| Stream creation | `ggml/src/ggml-cuda/ggml-cuda.cu`（`cudaStreamPerThread`）、`allreduce.cu:440` | `cudaStreamCreateWithFlags` | backend/pool 初始化创建 CUDA stream；主计算用 per-thread stream |
| Kernel launch | `ggml/src/ggml-cuda/*.cu`（如 `binbcast.cu:437` `ggml_cuda_op_add`） | `<<<grid,block,0,stream>>>` / `cudaLaunchKernel` | `ggml_cuda_op_*(ctx,dst)` → launch 对应 kernel |
| Synchronization | `ggml/src/ggml-cuda/ggml-cuda.cu:784,792,800,810` | `cudaStreamSynchronize(cudaStreamPerThread)`、`cudaEventSynchronize` | 每个 op 结束后同步；graph compute 后 sync |
| Server scheduler | `tools/server/server-context.cpp:2777` | `server_context::update_slots` | `queue_tasks.on_update_slots` → `update_slots()` → 逐 slot `pre_decode/decode/post_decode` |
| Worker/thread | `server-http.cpp:465,312`；`server-context.cpp:1408-1414` | `std::thread(listen_after_bind)`；`n_threads_http`；`server_queue queue_tasks` | HTTP 线程池接请求；调度循环在 uev/uv 事件循环里 `on_new_task`→`process_single_task`(server-context.cpp:2359)，`on_update_slots`→`update_slots` |

## 关键结论（影响后续 AgentServe 集成）
1. **调度与解码是同步的**：`update_slots()`（server-context.cpp:2777）在同一线程里逐 slot 做 `llama_decode`，`llama_decode` 内部同步执行 CUDA kernel。要拆成「prefill 线程 + decode 线程」+ Green Context，需要在 `update_slots` 前后/内部把不同 slot/阶段的 decode 分派到不同 CUDA stream/context，并做事件同步。
2. **KV cache 按 slot 管理**：`llama_kv_cache`（`llama-context.cpp` 持有）+ `server_slot` 各自维护 `n_past`，天然区分 cold(新 slot) / resume(已有 n_past) / decode(单 token)。AgentServe 的「prefill/decode 共享 KV 缓存、不跨进程拷贝」正好落在这里。
3. **CUDA stream 只用了 `cudaStreamPerThread`**：要给 decode 预留 SM（Green Context），需在其上创建受控上下文/流的 kernel 分派；当前 ggml-cuda 强制用 per-thread stream（ggml-cuda.cu:784 等），是集成的改动点。
4. **没有 SM 分区**：当前 llama.cpp 无 Green Context 概念；需在 CUDA backend 层（`ggml_backend_cuda_context`）创建 green context 并把 kernel 流绑进去。
5. **请求路由**：`handle_completions_impl` 是所有 completion 请求的唯一入口，适合作为「phase 分类（cold/resume/decode）」的埋点。

## 后续集成切入点（非改动，仅记录）
- 双 worker：`update_slots`（预填/解码分类后）→ 拆成两个 CPU 提交路径，各用一条 CUDA stream。
- Green Context：在 `ggml_backend_cuda_context` 初始化时用 `cuDeviceGetDevResource`/`cuGreenCtxCreate` 建 10%–100% 上下文，kernel 流绑到对应 green context。
- 事件打点：利用本项目 `scripts/events.py` 的事件模型，在上述 `COLD_PREFILL_START/END`、`DECODE_STEP_*` 处埋点。
