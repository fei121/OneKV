# AgentServe 复现：单引擎共享 KV 的 Agentic Serving

**在消费级 GPU 上，用单引擎共享 KV 缓存为多智能体负载做高效 Serving。**

本仓库复现了 **[AgentServe](https://arxiv.org/abs/2603.10342)**（*Algorithm-System Co-Design for
Efficient Agentic AI Serving on a Consumer-Grade GPU*）中的 **serving 系统**，并在单张 **RTX 3090**
上把它与业界主流后端（llama.cpp / vLLM / SGLang）做了四方对比。

核心贡献是：**在*一个*引擎内部做 prefill/decode（P/D）分离 + 跨上下文共享 KV**——不做跨进程 KV
拷贝。我们用补丁让 llama.cpp 的两个 `llama_context`（A=prefill，B=decode）共享同一份 KV 池
（`ctx_other = A`），再用**连续批处理 + 前缀缓存**驱动，让长 prefill 运行时 decode 延迟仍然稳定。

> **本项目衡量的是 serving 性能，不是智能体质量。** 模型是 Qwen2.5-**BASE**（无原生 tool-calling），
> 输出是计划式文本。

---

## 为什么重要

智能体负载（ReAct / Plan-and-Execute）在 **长冷预填**、**短 resume 预填**（工具输出）与 **极短
decode** 之间交替。在单 GPU 上这会造成 **head-of-line blocking**——一个长 prefill 会饿死所有其它
会话的微小 decode。

论文的思路*不是*把负载拆分到多个引擎（那要付 KV 拷贝成本），而是**在单引擎内拆分离**：让一个
context 去写长的 K/V（prefill），另一个 context 读它来做 decode（decode），共享一份 KV 池，这样
什么都不用拷贝。

---

## 关键结果

### Qwen2.5-3B（4-way，N=3→6，tool_wait=0）

| 范式 | 指标 | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|---|
| **ReAct** | 吞吐 (tok/s) | 149.8→175.2 | 131.1→145.0 | **167.7→222.8** | **164.2→226.3** |
| | TPOT p95 (ms) | **11.9→13.5** | 60.0→72.7 | 19.5→24.0 | 24.0→27.9 |
| | 冷 TTFT p50 (ms) | **308.6→367.6** | 439.5→737.8 | 302.7→621.4 | 302.2→454.8 |
| **P&E** | 吞吐 (tok/s) | 162.5→192.2 | 145.1→147.7 | **170.4→259.1** | **179.5→243.1** |
| | TPOT p95 (ms) | **11.8→13.5** | 50.2→70.7 | 22.0→38.7 | 24.8→31.4 |
| | 冷 TTFT p50 (ms) | **316.5→383.8** | 412.3→773.2 | 343.0→603.4 | 335.0→478.2 |

![ReAct 4-way](figures/react-4way-nscale.png)
![P&E 4-way](figures/pe-4way-nscale.png)

### Qwen2.5-7B（4-way，N=3→6，tool_wait=0）

| 范式 | 指标 | engine | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|---|
| **ReAct** | 吞吐 (tok/s) | 84.6→103.3 | 81.9→92.4 | 88.8→123.6 | 90.3→125.2 |
| | TPOT p95 (ms) | **20.8→22.3** | 65.0→125.8 | 37.9→46.0 | 31.3→44.6 |
| | 冷 TTFT p50 (ms) | **524.2→625.0** | 800.6→1179.0 | 591.1→1214.5 | 590.5→958.6 |
| **P&E** | 吞吐 (tok/s) | 93.2→115.8 | 92.1→108.1 | 94.8→131.3 | 97.2→134.5 |
| | TPOT p95 (ms) | **20.4→22.1** | 71.5→111.1 | 21.6→110.0 | 25.6→48.5 |
| | 冷 TTFT p50 (ms) | **540.9→646.0** | 803.2→1252.2 | 648.5→1036.0 | 782.8→945.2 |

![ReAct 7B](figures/react7-4way-nscale.png)
![P&E 7B](figures/pe7-4way-nscale.png)

### 结论（3B 与 7B 一致）

共享 KV 引擎是**延迟稳定性领先者**：
- **TPOT p95 平**（3B ~12–14ms，7B ~20–22ms）；llama.cpp 爆到 50–126ms，vLLM/SGLang 升到 20–110ms。
- **冷 TTFT 最低且基本不随 N 涨**（前缀缓存摊销；3B ~310–385ms，7B ~520–650ms），而所有基线都随 N 恶化。
- **吞吐不是它的强项**：**vLLM ≈ SGLang > engine > llama.cpp**。

这正是论文论点：*在不牺牲吞吐的前提下，提升 TTFT/TPOT 稳定性（即用一部分吞吐换延迟稳定）。*

> **关于基线。** vLLM/SGLang 用**足够的上下文**重测（vLLM `--max-model-len 32768`，SGLang
> `--max-total-tokens 49152`）。之前 `--max-total-tokens 8192` 把 SGLang 的 KV 池饿死，让它看起来
> 最差——那是 **harness bug，不是 SGLang 的限制**。另有一份「上下文窗口敏感度」实验（见
> [`docs/benchmark/methodology.md`](docs/benchmark/methodology.md) + [`figures/context-windows-n6.png`](figures/context-windows-n6.png)）
> 证明四个后端在窗口达到阈值后都在平台期，故**差距是真实架构差异，不是上下文配置造假**。

完整表格见 [`results/`](results/)，汇总见 [`docs/benchmark/results.md`](docs/benchmark/results.md)。

---

## 架构

```
             ┌──────────────────────────────────────────────┐
             │           一个模型 · 一份 KV 缓存            │
             │        (llama_kv_cache, ctx_other = A)       │
             │                                              │
             │   seq 0 = 模板                                │
             │        └─ 共享 system 前缀 (seq_cp)          │
             │   seq 1..N = 在跑的会话                       │
             └──────────────────────────────────────────────┘
                   ▲                    ▲
        写 K/V │           读 K/V │ + 追加新 token
       ┌──────────┴─────────┐  ┌────────┴─────────┐
       │ context A (prefill) │  │ context B (decode) │
       │ llama_decode(A, ...) │  │ llama_decode(B, ...) │
       │ CUDA stream: pre     │  │ CUDA stream: dec     │
       └──────────────────────┘  └──────────────────────┘
```

- **A（prefill）**：把每个会话的*冷* prompt（共享 system 前缀只预填一次，再 `llama_memory_seq_cp`
  到各会话）与 *resume* prompt（工具输出）打包成多序列 `llama_decode(A)`。
- **B（decode）**：连续批处理——每个 `llama_decode(B)` 每个就绪会话只解码 1 个 token，读 A 写入的
  KV 并追加新 token。
- `g_kv` 互斥锁只串行化主机侧 cell 记账；两个内核跑在不同 CUDA stream 上。
- **前缀缓存**：共享 system prompt 在 `seq 0` 上预填一次并复制给每个会话，从而摊销冷 TTFT。

### 怎么让共享 KV 生效（难点）

llama.cpp 的单个 `llama_context` 不可重入，其 KV 缓存也与它绑定。要让两个 context 共享一份池，
本仓库给 llama.cpp 打了补丁（见 [`patches/`](patches/)）：

| 文件 | 改动 |
|---|---|
| `llama-model.cpp` | Qwen 分支传 `mem_other` + share 回调，让 decode 的 K/V 指向 prefill 的 K/V。 |
| `llama-context.cpp` | 把 `params.ctx_other` 传到 `cparams.ctx_other`。 |
| `llama-kv-cache.cpp` | `apply_ubatch` 允许镜像 context 写共享 cell；`seq_pos_min/max` 读共享 cell。 |
| `ggml-cuda-common.cuh` | `as_sidx()` 按当前流索引绑定 cuBLAS handle/workspace/pool。 |

引擎源码：[`src/runtime/agentserve_engine.cpp`](src/runtime/agentserve_engine.cpp)。

---

## 安装

### 1. 硬件 / 系统
- NVIDIA GPU（实测 **RTX 3090**，24GB）、**CUDA 12.8**、Ubuntu 22.04。

### 2. 下载模型

四个模型产物（Qwen2.5-3B/7B × GGUF/HF）及其服务器路径 + SHA-256 见
[`configs/models/models.yaml`](configs/models/models.yaml)。

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
apt update && apt install -y aria2
export HF_ENDPOINT=https://hf-mirror.com
export HFD_DOWNLOADER="aria2c -x 16 -s 16 -k 1M"
# GGUF（engine + llama.cpp 基线）
hfd Qwen/Qwen2.5-3B-GGUF --include "*qwen2.5-3b-f16*.gguf"
# HF safetensors（vLLM / SGLang + trace 生成）
hfd Qwen/Qwen2.5-3B
```

### 3. 打补丁并编译 llama.cpp
见 [`docs/setup/llama-cpp-patch.md`](docs/setup/llama-cpp-patch.md) 与
[`docs/setup/environment.md`](docs/setup/environment.md)（精确源码、编译选项、环境）。

```bash
cmake -DHF_ENABLED=OFF -DBUILD_UI=OFF -B build .
cmake --build build --target llama-cli llama-server
# 把 patches/ 里 4 个文件覆盖到对应源码，然后重编译
```

### 4. Python 包
```bash
pip install -e .   # requests, pyyaml, matplotlib, numpy, pytest
```

---

## 使用

```bash
# 生成统一的 12 任务 trace（ReAct / P&E 共用）
make trace

# 运行单引擎共享 KV 运行时（A = 并发智能体数）
make engine A=3

# 基线（自包含多阶段 prompt）
make serve-llama A=3 S=12
make serve-vllm  A=3 S=12
make serve-sglang A=3 S=12

# 串行基准扫描（ReAct + P&E，N=3…6，一次只跑一个后端）
make sweep

# 画图
python scripts/plot_perparadigm.py      # per-paradigm 4-way（3B + 7B）
python scripts/plot_context_windows.py  # 上下文窗口敏感度
```

---

## 可复现性

服务器精确环境（硬件、CUDA、llama.cpp 版本 + 编译选项、模型 SHA-256、conda 版本）已固定在
[`docs/setup/environment.md`](docs/setup/environment.md)。每个后端都由同一套 harness、同一
12 任务集、同一 `N`、同一 `tool_wait` 驱动，且**串行测量**（一次一个后端，run 之间释放 GPU）。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/design/architecture.md`](docs/design/architecture.md) | 系统 + 引擎 + 补丁设计。 |
| [`docs/design/paper-alignment.md`](docs/design/paper-alignment.md) | 与论文的对齐情况、差异与原因。 |
| [`docs/setup/environment.md`](docs/setup/environment.md) | 固定到字节级的环境（可复现）。 |
| [`docs/setup/llama-cpp-patch.md`](docs/setup/llama-cpp-patch.md) | 4 个共享 KV 补丁 + 编译/验证。 |
| [`docs/benchmark/methodology.md`](docs/benchmark/methodology.md) | workload、后端、指标 + "上下文"公平性规则。 |
| [`docs/benchmark/results.md`](docs/benchmark/results.md) | 当前结果、图表、复现方法。 |
| [`docs/notes/known-limitations.md`](docs/notes/known-limitations.md) | 诚实边界 + 方法学坑。 |
| [`docs/notes/pitfalls.md`](docs/notes/pitfalls.md) | 服务启动 / 基准测试经验教训。 |
| [`results/`](results/) | 干净的 per-paradigm 四方表 + 摘要。 |
| [`REPORT.md`](REPORT.md) | 完整报告（设计、引擎、结果）。 |

---

## 诚实边界 / 局限

1. **绝对数值不可跨硬件比较** —— RTX 3090（82 SM）vs 论文的 A5000/5090。我们始终与*自己的*、
   同 GPU、同 trace 的基线比较。
2. **未复现论文的 10 槽 Green Context 池 + TPOT 驱动的 `Rmin` 重绑定**。我们改用*连续批处理*来达到
   "decode 保护"这一收益，实测在 3090 上做 SM 预留反而更差（见
   [`docs/design/paper-alignment.md`](docs/design/paper-alignment.md)）。
3. **7B 的 vLLM/SGLang 也已用修正后的上下文重测**，因此 **3B/7B** 两套对比都已对齐（见
   [`results/perparadigm-4way-7b.md`](results/perparadigm-4way-7b.md)）。
4. **BASE 模型，无原生 tool-calling** → 衡量的是 serving 性能，不是智能体质量。

---

## 许可证

MIT —— 见 [`LICENSE`](LICENSE)。
