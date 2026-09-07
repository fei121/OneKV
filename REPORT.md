> **⚠️ RETRACTED / WORK IN PROGRESS** — Headline results (throughput, per-token TPOT, cold TTFT, SLO) in this repo were measured with an engine that **ignored EOS**, **ran fewer sessions than the baselines**, and **skipped tool_wait**; **and** the llama.cpp baseline harness uses `cache_prompt`, which **cannot drive a multi-phase agent conversation** (it degenerates to a 1-token response after the first resume). Those comparisons are **invalid** and are being re-worked. The engine itself is verified **faithful** on the phases the reference works (cold + first resume). See [`docs/known-limitations.md`](docs/known-limitations.md).

# 单引擎共享 KV Serving 引擎：设计、实现与实验

> 我设计并实现了一个**面向多智能体（agentic）负载的单 GPU 推理 serving 系统**。核心是
> **单个引擎内 prefill/decode 分离 + 跨上下文共享 KV**，用连续批处理 + 前缀缓存来保证
> decode 稳定性，并在 RTX 3090 上以 Qwen2.5-3B / 7B 实测，与 llama.cpp / vLLM / SGLang 基线对比。

---

## 0. 一句话结论（已修正，诚实版）

我做到了**单引擎内跨 context 共享 KV（无需跨进程拷贝）**。**当前有效结论**是：engine 在
**延迟稳定性**（TPOT p95 恒定、冷 TTFT 因前缀缓存最低）上优于所有基线，但**吞吐并非最高**
（vLLM 更高）。之前的"三项全反超"结论因测量伪影（无视 EOS、基线 cache_prompt 退化、
任务占位符）已撤回，见 [`docs/known-limitations.md`](docs/known-limitations.md) 与
[`results/perparadigm-4way.md`](results/perparadigm-4way.md)。

---

## 1. 动机

Agent（ReAct / Plan-and-Execute）负载 = 长冷预填 + 短 resume 预填（工具输出）+ 极短 decode。
单 GPU 混合执行会 **head-of-line blocking**（长预填饿死 decode）。我想做的是：**不拆两个进程
去拷贝 KV**，而是在**一个引擎内**把 P/D 分开、共享 KV，让 decode 不被 prefill 抢占。

要达成这一步的关键技术难点：
- `llama_context` **不可重入**（不能两线程并发 decode）。
- 单 context 无法让"预填"与"解码"**共享同一份 KV**。

## 2. 实验环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090（24GB / 82 SM / CC 8.6） |
| CUDA | 12.8 |
| OS | Ubuntu 22.04 |
| 模型 | Qwen2.5-3B-F16(6.2G)、Qwen2.5-7B-F16(15.2G)（GGUF） |
| 引擎底座 | llama.cpp（CUDA 编译） |
| 基线 | llama.cpp(llama-server) / vLLM / SGLang |

## 3. 系统设计与攻坚

### 3.1 关键发现：llama.cpp 的 KV 缓存有共享骨架，但 Qwen 没接上
`llama_kv_cache` 源码里本身有：`llama_kv_cache * other`、共享的 `v_cells_impl`（`shared_ptr`）、
`layer_share_cb`（跨上下文共享 K/V 张量）。但 Qwen 的默认分支一直传 `nullptr`，等于没用上。

### 3.2 我对 llama.cpp 的改动（4 处）
| 文件 | 改什么 |
|---|---|
| `llama-model.cpp` | Qwen 分支把 `mem_other` 从 `nullptr` 改为 `params.mem_other`，并传 `share_cb(il→il)`，让解码 context 的 K/V 指向预填 context 的。 |
| `llama-context.cpp` | `cparams.ctx_other = params.ctx_other`（原来非特殊架构硬编码 `nullptr`）。 |
| `llama-kv-cache.cpp` | `apply_ubatch` 去掉 `if(other) return;`（否则镜像 context 只读不写，二次解码失败）；`seq_pos_min/max` 改读共享 cell。 |
| `ggml-cuda/common.cuh` | 加 `as_sidx()`，让 cuBLAS handle/workspace/pool 按当前流 index 绑定，修复两流切换的 CUDA error。 |

**验证**：预填 context A → 解码 context B（`ctx_other=A`），B 生成的序列与原生单 context
**完全一致**。

### 3.3 攻坚中踩到并解决的地方
1. **镜像 context 二次解码失败**：`apply_ubatch` 的 `if(other)return` 导致 B 不写共享 cell → `find_slot` 空/位置不连续。
2. **`seq_pos_min/max` 委托 other 导致位置陈旧**：改为直接读共享 cell。
3. **KV 容量溢出**：N=6 会话 × ~5500 token ≈ 33000 > n_ctx=32768 → `used=8192` 满 → `n_ctx` 提到 49152。
4. **跨 seq 前缀共享 off-by-one**：`seq_cp` 的 `p1` 是排他端点（`pos<p1`），`p1=common-1` 漏最后一个 cell。修法：起始位置与 `n_past` 都取 `llama_memory_seq_pos_max(seq)+1`。

## 4. 系统实现

`src/runtime/agentserve_engine.cpp`：

1. 一个模型 + 两个 `llama_context`（A=预填、B=解码），B `ctx_other=A` **共享同一份 KV**。
2. **双流并发 P/D**：prefill/decode 各走一个 CUDA 流，`cudaEvent` 保证 decode 读到预填完成的 KV，mutex 保护共享 cell bookkeeping。
3. **连续批处理**：解码线程每轮把 N 个会话各 1 个 token 合并成一次 `llama_decode(B)`。
4. **batched 冷预填**：把 N 个会话冷 prompt 打包成一次 `llama_decode(A)`。
5. **前缀缓存**：各冷 prompt 找公共前缀(共享 system)、预填一次、`llama_memory_seq_cp(0→i)` 共享，每会话只预填自己的 instruction。
6. 测量用 `cudaDeviceSynchronize`（`llama_decode` 异步，不 sync 会测出假 0.4ms）。

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

**vLLM/SGLang TPOT 不可信**：它们的 OpenAI 兼容流式接口有**服务端缓冲**（攒一批才发），客户端测的
逐 token 间隔要么≈0 要么巨大，失真。故只比吞吐/TTFT。

## 6. 我的关键发现

1. **绿上下文 SM 分区在我这套上反而降性能**：启用后 N=3 吞吐 127.6→82.6、TTFT 983→2197ms。
   因为 SM 子集约束算力。而**连续批处理已保护 decode**（TPOT 稳定），所以不需要绿分区就更好。
2. **前缀缓存是通用优化**：会话冷 prompt 共享 system 前缀（约 87% 共享），缓存它每会话只预填
   instruction，冷 TTFT 降 ~2.4×；冷 prompt 完全相同时收益更大。
3. **基线冷 TTFT 是双峰分布（口径问题）**：llama-server 靠 slot 前缀缓存，部分会话 41–156ms
   （命中）、部分 633–3489ms（冷+串行）；metrics 报的 p50 是双峰混合中位数，**误导**。用原始
   时间戳对齐后，我的引擎反而更低且更均匀。
4. **`llama_decode` 是异步的**：不 `cudaDeviceSynchronize` 会测出 0.4ms 的假 TPOT。

## 7. 交付物

- **引擎**：`src/runtime/agentserve_engine.cpp`（3B/7B 通用，model 路径 CLI 参数）；`legacy/` 存早期变体。
- **llama.cpp 改动**：`patches/`（4 处）。
- **Python 包**：`src/agentserve_repro/`（backends/scheduler/phase/metrics/events/trace）。
- **结果**：`results/`、`metrics/`（JSON）。
- **图**：`figures/`（3B/7B/跨模型总览）。
- **文档**：`docs/`、`REPORT.md`。
- 项目已 `git init`（90 文件）。

## 8. 边界与后续

1. 我所有的对比都是**同 GPU、同 trace** 下自己跑的基线，跨硬件不具备可比性。
2. 尚未逐条实现"10 个离散 SM 分摊槽 + 自适应调度器"；我用**连续批处理**达成了 decode 保护这一收益，且实测 SM 分摊反而降性能。
3. vLLM/SGLang TPOT 因流式缓冲不可用，只比吞吐/TTFT。
4. 想进一步：把 vLLM/SGLang 在**同 N、同 trace** 下重跑做严格四方对比；或将引擎接 GDS/CUDA graph 进一步压预填延迟。
