# AgentServe 复现实验报告

> 复现目标：**arXiv 2603.10342《AgentServe: Algorithm–System Co-Design for Efficient Agentic AI
> Serving on a Consumer-Grade GPU》** 的核心系统贡献——**单引擎内 prefill/decode 分离 + 共享 KV**，
> 并在 RTX 3090 上以 Qwen2.5-3B / 7B 实测，与 llama.cpp / vLLM / SGLang 基线公平对比。

---

## 0. 一句话结论

**我们把论文"最难、也是真正贡献点"的"单引擎内跨 context 共享 KV（避免引擎间拷贝）"从"做不到"
做到了实机跑通并验证，并在 Qwen2.5-3B 和 7B 上让共享KV引擎在吞吐 / TPOT / 冷TTFT 三项上
反超 llama.cpp 基线。**

---

## 1. 目标与背景

Agent（ReAct / Plan-and-Execute）负载 = 长冷预填 + 短 resume 预填（工具输出）+ 极短 decode。
单 GPU 上混合执行会产生 **head-of-line blocking**（长预填饿死 decode）。

论文的核心思想：**不做双引擎 + KV 拷贝**，而是**单引擎内 P/D 分离 + 共享 KV**，让 decode 不被
prefill 抢占。要达成这一步，关键障碍是：
- `llama_context` **不可重入**（无法两线程并发 decode）。
- 单 context 无法让"预填 context"和"解码 context"**共享同一份 KV**。

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090（24GB / 82 SM / CC 8.6） |
| CUDA | 12.8 |
| OS | Ubuntu 22.04 |
| 模型 | Qwen2.5-3B-F16(6.2G)、Qwen2.5-7B-F16(15.2G)（GGUF） |
| 引擎底座 | llama.cpp（CUDA 编译） |
| 基线 | llama.cpp(llama-server) / vLLM / SGLang |

## 3. 核心难点攻关：单引擎共享 KV

这是本次实验最重要的攻坚。过程经历了多个根因，逐步修通：

### 3.1 发现 llama.cpp 已有共享机制但没接上
`llama_kv_cache` 源码里本来就有：
- `llama_kv_cache * other`
- 共享的 `v_cells_impl`（`shared_ptr`）
- `layer_share_cb`（跨 cache 共享 K/V 张量）
- 维护者留的 TODO：`[TAG_KV_CACHE_SHARE_CELLS]`

但 **Qwen 的默认 dense 分支传了 `nullptr`**，等于没用上。

### 3.2 四个补丁（实际改动）
| 文件 | 改什么 |
|---|---|
| `llama-model.cpp` | Qwen dense 分支把 `mem_other` 从 `nullptr` 改为 `params.mem_other`，并传 `share_cb(il→il)`，让解码 context 的 K/V 张量指向预填 context 的。 |
| `llama-context.cpp` | `cparams.ctx_other = params.ctx_other`（原来非特殊架构硬编码 `nullptr`，传不进来）。 |
| `llama-kv-cache.cpp` | `apply_ubatch` 去掉 `if(other) return;`（否则镜像 context 只读不写，第二次解码就失败/位置不连续）；`seq_pos_min/max` 改为直接读共享 cell（不再委托 other）。 |
| `ggml-cuda/common.cuh` | 加 `as_sidx()`，让 cuBLAS handle/workspace/pool 按当前绿流 index 绑定，修复两绿流切换的 CUDA error。 |

**验证**：A 预填 → B（`ctx_other=A`）解码，输出序列与原生单 context **完全一致**（`iso8b` MATCH=YES）。

### 3.3 攻坚中踩到并解决的坑
1. **镜像 context 第二次解码失败**：`apply_ubatch` 的 `if(other) return` 导致 B 不写共享 cell → `find_slot` 空 / 位置不连续 → 去掉。
2. **`seq_pos_min/max` 委托 other 导致位置陈旧**：B 读数一致但 decode 位置对不上 → 改为直接读共享 cell。
3. **KV 容量溢出**：N=6 会话 × ~5500 token ≈ 33000 > n_ctx=32768 → `used=8192` 满 → 把 `n_ctx` 提到 49152。
4. **跨 seq 前缀共享（seq_cp）的 off-by-one**：`llama_kv_cache::seq_cp` 的 `p1` 是**排他端点**（`pos<p1`），`p1=common-1` 漏最后一个共享 cell → `seq_pos_max=626`。修法：instruction 起始和 `n_past` 都取 **`llama_memory_seq_pos_max(seq)+1`**，保证 `Y=X+1`。

## 4. 实现：单引擎共享KV serving 引擎

`src/runtime/agentserve_engine.cpp`：

1. 一个模型 + 两个 `llama_context`（A=预填、B=解码），B `ctx_other=A` 共享 KV。
2. **双流并发 P/D**：prefill→prefill 流、decode→decode 流，`cudaEvent` 保证 decode 读到预填完成的 KV，mutex 保护共享 cell bookkeeping。
3. **连续批处理**：解码线程每轮把所有待解码会话的 1 个 token 合并成一次 `llama_decode(B)`。
4. **batched 冷预填**：把 N 个会话的冷 prompt 打包成一次 `llama_decode(A)`。
5. **前缀缓存**：tokenize 各冷 prompt、找公共前缀(共享 system)、预填一次、`llama_memory_seq_cp(0→i)` 共享给各会话，每会话只预填自己的 instruction。
6. 测量用 `cudaDeviceSynchronize`（llama_decode 是异步的，不 sync 会测出假 0.4ms）。

## 5. 实验与结果

### 5.1 Qwen2.5-3B（N=6，多样任务三方）

| 指标 | llama-server 基线 | 引擎(无缓存) | 引擎(前缀缓存) |
|---|---|---|---|
| 冷 TTFT p50 | 536.9 ms | 95.3 ms | **35.5 ms** |
| TPOT p50 / p95 | 12.47 / 25.19 ms | 10.35 / 16.18 ms | **9.87 / 15.77 ms** |
| 吞吐 | 161.1 tok/s | 176.9 tok/s | **228.6 tok/s** |

### 5.2 Qwen2.5-7B（N=6）

| 指标 | llama.cpp 基线 | **共享KV引擎** |
|---|---|---|
| 吞吐 | 79.34 tok/s | **160.6 tok/s** (+102%) |
| TPOT p50 / p95 | 24.43 / **109.62 ms** | **19.15 / 22.12 ms** |
| 冷 TTFT p50 | **2404 ms** | **568.7 ms** |

### 5.3 vLLM / SGLang
| backend | TTFT p50 | 吞吐 | TPOT |
|---|---|---|---|
| vLLM | 217.2 ms | 137.9 tok/s | 0.0(不可信) |
| SGLang | 227.3 ms | 67.7 tok/s | 0.0(不可信) |

**vLLM/SGLang TPOT 不可信**：OpenAI 兼容流式接口有**服务端缓冲**（攒一批才发），客户端测的
逐 token 间隔要么≈0 要么巨大，失真。故只比吞吐/TTFT。

## 6. 关键发现（诚实记录）

1. **绿上下文 SM 分区在我们这反而降性能**：启用绿分区后 N=3 吞吐 127.6→82.6、TTFT 983→2197ms，
   因为 SM 子集约束算力。我们的**连续批处理已经保护了 decode**（TPOT 稳定），不需要绿分区，
   故单靠连续批处理就更优。
2. **前缀缓存是通用优化**：会话冷 prompt 共享 system 前缀（MirrorAPI 里约 ~87% 共享），
   缓存它每会话只预填 instruction，冷 TTFT 降 ~2.4×；冷 prompt 完全相同时收益更大。
3. **基线冷 TTFT 是双峰分布（口径问题）**：llama-server 靠 slot 前缀缓存，部分会话 41–156ms
   （命中）、部分 633–3489ms（冷+串行）；metrics 报的 p50 是双峰混合中位数，会**误导**。
   用原始时间戳对齐全口径后，我的引擎冷 TTFT 反而更低且更均匀。
4. **llama_decode 是异步的**：不 `cudaDeviceSynchronize` 会测出 0.4ms 的假 TPOT。

## 7. 交付物

- **引擎**：`src/runtime/agentserve_engine.cpp`（3B/7B 通用，model 路径 CLI 参数）；`legacy/` 存早期变体。
- **补丁**：`patches/`（4 个 llama.cpp 补丁）。
- **Python 包**：`src/agentserve_repro/`（backends/scheduler/phase/metrics/events/trace）。
- **结果**：`results/`（3B 基线对比、三方、7B 对比）、`metrics/`（JSON）。
- **图**：`figures/`（3B/7B/跨模型总览）。
- **文档**：`docs/`（architecture、paper-alignment、experiments、llama-cpp 说明）。
- 项目已 `git init`（90 文件）。

## 8. 诚实边界

1. 论文硬件是 A5000/5090；我们 RTX 3090。绝对数值不同，**所有对比都是我们自己同 GPU/同 trace 跑的基线**，相对结论有效。
2. **10 个绿上下文槽 + TPOT 驱动 Rmin 重绑**未逐条实现——我们用连续批处理达成"decode 保护"这一收益，且实测绿分区反而降性能。
3. vLLM/SGLang TPOT 因流式缓冲不可用，只比吞吐/TTFT。
4. 跨模型对比图表里 3B 用多样任务 N=6、7B 用 N=6；如需与 vLLM/SGLang N=3 严格对齐，需在相同 N/trace 下重跑。
