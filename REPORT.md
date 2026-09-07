# 单引擎共享 KV Serving 引擎：设计、实现与实验

> **范围说明**：本报告只评估 **serving 性能**（吞吐、TPOT、冷 TTFT）。所有指标均在同一
> RTX 3090、同一真实任务 trace、同一统一 12-task 集、固定基线、串行测量下测得，因此各后端
> 之间的**相对比较**成立；绝对数值只对当前硬件/模型有效，不与论文的 A5000/5090 数值直接可比。

我设计并实现了一个**面向多智能体（agentic）负载的单 GPU 推理 serving 系统**。核心是
**单个引擎内 prefill/decode 分离 + 跨上下文共享 KV**，用连续批处理 + 前缀缓存保证 decode
稳定性，并在 RTX 3090 上以 Qwen2.5-3B / 7B 实测，与 llama.cpp / vLLM / SGLang 基线做四方对比。

---

## 0. 一句话结论（诚实版）

我做到了**单引擎内跨 context 共享 KV（无需跨进程拷贝）**。**有效结论**是：engine 是**延迟稳定性
领先者**——TPOT p95 全程平稳（3B ~12–14ms、7B ~20–22ms），冷 TTFT 因前缀缓存最低且几乎不随
并发 N 增长；**但吞吐并非最高**（**vLLM ≈ SGLang > engine > llama.cpp**，SGLang 在修正上下文后与 vLLM 相当）。
这一结论**复现了论文论点**：*在不牺牲吞吐的前提下提升 TTFT/TPOT 稳定性*。

> 早前"三项全反超（吞吐/TPOT/TTFT 都碾压）"的结论因测量伪影（engine 无视 EOS、基线
> `cache_prompt` 无法驱动多阶段 agent、任务占位符）已被撤回。见
> [`docs/notes/known-limitations.md`](docs/notes/known-limitations.md) 与干净结果
> [`results/perparadigm-4way.md`](results/perparadigm-4way.md)。

---

## 1. 动机

Agent（ReAct / Plan-and-Execute）负载 = 长冷预填 + 短 resume 预填（工具输出）+ 极短 decode。
单 GPU 混合执行会产生 **head-of-line blocking**（长预填饿死 decode）。论文的目标不是拆两个进程
去拷贝 KV，而是在**一个引擎内**把 P/D 分开并共享 KV，让 decode 不被 prefill 抢占。

关键技术难点：

- `llama_context` **不可重入**（不能两线程并发 decode）。
- 单 context 无法让"预填"与"解码"**共享同一份 KV**。

## 2. 实验环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090（24GB / 82 SM / CC 8.6） |
| CUDA | 12.8 |
| OS | Ubuntu 22.04 |
| 模型 | Qwen2.5-3B-F16(6.2G)、Qwen2.5-7B-F16(15.2G)（GGUF，BASE） |
| 引擎底座 | llama.cpp（CUDA 编译） |
| 基线 | llama.cpp(llama-server) / vLLM / SGLang |

**统一 workload**：`data/unified_tasks.json`（12 条真实 ToolBench 任务），ReAct 与 P&E 共用同一
任务集，仅 prompt 模板与 token 分布不同。12 sessions，并发 N=3…6，tool_wait=0。

## 3. 系统设计与攻坚

### 3.1 关键发现：llama.cpp 的 KV 缓存有共享骨架，但 Qwen 没接上

`llama_kv_cache` 源码里本身有 `llama_kv_cache * other`、共享的 `v_cells_impl`（`shared_ptr`）、
`layer_share_cb`（跨上下文共享 K/V 张量）。但 Qwen 的默认分支一直传 `nullptr`，等于没用上。

### 3.2 对 llama.cpp 的改动（4 处）

| 文件 | 改什么 |
|---|---|
| `llama-model.cpp` | Qwen 分支把 `mem_other` 从 `nullptr` 改为 `params.mem_other`，并传 `share_cb(il→il)`，让解码 context 的 K/V 指向预填 context 的。 |
| `llama-context.cpp` | `cparams.ctx_other = params.ctx_other`（原来非特殊架构硬编码 `nullptr`）。 |
| `llama-kv-cache.cpp` | `apply_ubatch` 去掉 `if(other) return;`（否则镜像 context 只读不写，二次解码失败）；`seq_pos_min/max` 改读共享 cell。 |
| `ggml-cuda/common.cuh` | 加 `as_sidx()`，让 cuBLAS handle/workspace/pool 按当前流 index 绑定，修复两流切换的 CUDA error。 |

**验证**：预填 context A → 解码 context B（`ctx_other=A`），B 生成的序列与原生单 context
**完全一致**（token-for-token，已对齐 llama-server 冷 + 首次 resume 阶段）。

### 3.3 攻坚中解决的问题

1. **镜像 context 二次解码失败**：`apply_ubatch` 的 `if(other) return` 导致 B 不写共享 cell → `find_slot` 空/位置不连续。
2. **`seq_pos_min/max` 委托 other 导致位置陈旧**：改为直接读共享 cell。
3. **KV 容量溢出**：N=6 会话 × ~5500 token ≈ 33000 > n_ctx=32768 → `n_ctx` 提高到 49152。
4. **跨 seq 前缀共享 off-by-one**：`seq_cp` 的 `p1` 是排他端点（`pos<p1`），`p1=common-1` 漏最后一个 cell。修法：起始位置与 `n_past` 都取 `llama_memory_seq_pos_max(seq)+1`。
5. **EOS / 12-session / tool_wait**：engine 尊重 EOS（`llama_vocab_is_eog`）；12-session workload 复用 N 个 slot；agent 步间加 200ms tool_wait。
6. **测量口径**：`llama_decode` 是异步的，需 `cudaDeviceSynchronize`，否则测出 0.4ms 假 TPOT。

## 4. 系统实现

`src/runtime/agentserve_engine.cpp`：

1. 一个模型 + 两个 `llama_context`（A=预填、B=解码），B `ctx_other=A` **共享同一份 KV**。
2. **双流并发 P/D**：prefill/decode 各走一个 CUDA 流，`cudaEvent` 保证 decode 读到预填完成的 KV，mutex 保护共享 cell bookkeeping。
3. **连续批处理**：解码线程每轮把 N 个会话各 1 个 token 合并成一次 `llama_decode(B)`。
4. **batched 冷预填**：把 N 个会话冷 prompt 打包成一次 `llama_decode(A)`。
5. **前缀缓存**：各冷 prompt 找公共前缀(共享 system)、预填一次、`llama_memory_seq_cp(0→i)` 共享，每会话只预填自己的 instruction。

## 5. 实验结果（干净、per-paradigm、四方对比）

- [`results/perparadigm-4way.md`](results/perparadigm-4way.md) — Qwen2.5-3B，ReAct + P&E 分开。
- [`results/perparadigm-4way-7b.md`](results/perparadigm-4way-7b.md) — Qwen2.5-7B，ReAct + P&E 分开。

### 5.1 Qwen2.5-3B（N=3→6，tool_wait=0）

> vLLM/SGLang 用**修正后的上下文**重测（vLLM `--max-model-len 32768`、SGLang `--max-total-tokens 49152`）。
> 5.2（7B）里的 vLLM/SGLang 是**旧上下文**测得，仅供参考。

**ReAct**

| metric | shared-KV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput (tok/s) | 149.8→175.2 | 131.1→145.0 | 167.7→222.8 | 164.2→226.3 |
| TPOT p95 (ms) | **11.9→13.5** | 60.0→72.7 | 19.5→24.0 | 24.0→27.9 |
| cold TTFT (ms) | **308.6→367.6** | 439.5→737.8 | 302.7→621.4 | 302.2→454.8 |

**Plan-and-Execute**

| metric | shared-KV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput (tok/s) | 162.5→192.2 | 145.1→147.7 | 170.4→259.1 | 179.5→243.1 |
| TPOT p95 (ms) | **11.8→13.5** | 50.2→70.7 | 22.0→38.7 | 24.8→31.4 |
| cold TTFT (ms) | **316.5→383.8** | 412.3→773.2 | 343.0→603.4 | 335.0→478.2 |

### 5.2 Qwen2.5-7B（N=3→6，tool_wait=0）

**ReAct**

| metric | shared-KV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput (tok/s) | 84.6→103.3 | 81.9→92.4 | 88.5→125.0 | 55.2→41.3 |
| TPOT p95 (ms) | **20.8→22.3** | 65.0→125.8 | 27.9→45.0 | 30.2→24.8 |
| cold TTFT (ms) | **524.2→625.0** | 800.6→1179.0 | 617.6→1055.2 | 960.2→3331.0 |

**Plan-and-Execute**

| metric | shared-KV engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| throughput (tok/s) | 93.2→115.8 | 92.1→108.1 | 93.2→132.7 | 55.3→47.5 |
| TPOT p95 (ms) | **20.4→22.1** | 71.5→111.1 | 21.9→83.1 | 21.0→21.1 |
| cold TTFT (ms) | **540.9→646.0** | 803.2→1252.2 | 756.9→1041.9 | 1079.5→4250.8 |

## 6. 关键发现

1. **engine 是延迟稳定性领先者**（ReAct / P&E、3B / 7B 一致）：TPOT p95 全程平稳（3B 12–14ms、
   7B 20–22ms），而 llama.cpp 爆到 50–126ms，vLLM/SGLang 随 N 上升（20–39ms）。其冷 TTFT 因**共享
   system 前缀缓存**最低且几乎不随 N 增长，而所有基线都随 N 恶化。
2. **吞吐不是 engine 强项**：**vLLM ≈ SGLang > engine > llama.cpp**。修正 SGLang 上下文后，SGLang
   吞吐与 vLLM 相当（3B 164→243 / 168→259），都**超过 engine**（150→192）。engine 是 competitive、not maximal。
3. **优势随模型规模放大**：7B 上 TPOT-p95 差距更大（22 vs 126ms @N=6），冷 TTFT 前缀缓存优势更明显。
4. **P&E vs ReAct**：engine 在两种范式下行为一致（TPOT/TTFT 都平）。vLLM/SGLang 在 P&E 上吞吐最高
   但 TPOT p95 / 冷 TTFT 随 N 上升——典型的 latency/throughput 强权衡。
5. **前缀缓存是通用优化**：会话冷 prompt 共享 system 前缀（约 87%），缓存后每会话只预填自己的
   instruction，冷 TTFT 降 ~2.4×；prompt 完全相同时收益更大。
6. **连续批处理已保护 decode**，无需绿分区（见 §7）。

### 吞吐变体权衡（engine 内部）

`main`（innovation-1）= batched 多序列 resume-prefill + **独立** batched decode，TPOT p95 稳定
12–13.6ms；`chunk_prefill` 分支 = **一次** `llama_decode` 混合 resume-prefill + decode（连续批处理），
吞吐 +1–4% 但 **TPOT p95 恶化**（react N6 13.5→20.3ms，P&E N6 13.6→41.2ms）——这正是论文警告的
chunked prefill 扰动短 decode 现象。详见 [`docs/notes/known-limitations.md`](docs/notes/known-limitations.md)。

## 7. 与论文的差异（诚实边界）

| 论文机制 | 本文 |
|---|---|
| 单引擎、共享 KV、无跨进程拷贝 | ✅ **复现**（两个 `llama_context` 经 `ctx_other` 共享一份 KV 池） |
| 双线程 P/D 分离 | ✅（双流 + `cudaEvent`/mutex） |
| Prefill/decode 分离 | ✅ |
| CUDA Green Context SM 分区 | ⚠️ 用**连续批处理**替代（实测 SM 分区在 3090 上反而降性能：吞吐 127.6→82.6、TTFT 983→2197ms） |
| 10 绿槽 + TPOT-driven `Rmin` 重绑定 | ➖ 未实现（对本文演示的收益正交，且加 SM 约束反而伤性能） |
| 前缀/上下文复用 | ✅（system-prompt 前缀缓存） |

**说明**：论文的 Green-Context 收益本意是"防止长 prefill 饿死 decode"；连续批处理通过分离阶段
已经消除了这种饿死，因此在此场景下 SM 预留变得多余。

## 8. 交付物

- **引擎**：`src/runtime/agentserve_engine.cpp`（3B/7B 通用，model 路径 CLI 参数）；`legacy/` 存早期变体。
- **llama.cpp 改动**：`patches/`（4 处）。
- **Python 包**：`src/agentserve_repro/`（backends/scheduler/phase/metrics/events/trace）。
- **结果**：`results/`、`metrics/`（JSON）。
- **图**：`figures/`（3B/7B/per-paradigm/跨模型总览）。
- **文档**：`docs/`（design / build / benchmark / notes 四类）。
- 项目已 `git init`，含 LICENSE、Makefile、pyproject.toml、tests。

## 9. 边界与后续

1. 所有对比均为**同 GPU、同 trace** 下自测基线，跨硬件绝对数值不具备可比性。
2. 尚未逐条实现"10 个离散 SM 分摊槽 + 自适应调度器"；用**连续批处理**达成 decode 保护，且实测 SM 分摊反而降性能。
3. `chunk_prefill`（连续批处理）是纯吞吐变体，会牺牲 decode 稳定性；默认用 `main`（innovation-1）讲稳定性故事。
4. 想进一步提升吞吐：**提高并发 N**（共享 KV 池为此设计，吞吐近线性增长且稳定性保持），或探索**投机/并行解码**。

## 10. 复现

```bash
# unified task set（ReAct / P&E 共用，已持久化）：
data/unified_tasks.json

# engine（12 sessions, N 并发, tool_wait=0）：
/tmp/as_conc_batch <sessions_react.txt|sessions_plan_and_execute.txt> <N> -1 1 <n_ctx> <model.gguf> 12 0
# 例：/tmp/as_conc_batch sessions_react.txt 4 -1 1 49152 /root/autodl-tmp/models/Qwen2.5-3B-f16.gguf 12 0

# llama.cpp（自包含多阶段 prompt）：
python scripts/serve_llama.py --config configs/serving_react_u.yaml --agents N --sessions 12

# vLLM / SGLang：
python scripts/serve_backend.py --backend {vllm,sglang} --model-path /root/models/Qwen2.5-3B --config configs/serving_react_u.yaml --agents N --sessions 12
```

数据来源：`metrics/perparadigm/{react,pe,react7,pe7}.json`；
画图：`scripts/plot_perparadigm.py`。
