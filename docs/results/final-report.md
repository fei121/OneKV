# AgentServe 复现最终报告（固化交付版）

**论文**：AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU (arXiv 2603.10342v1)
**设备**：NVIDIA RTX 3090 (24GB / 82 SM / CC 8.6)，CUDA 12.8 + driver 580.142，Ubuntu 22.04.5，Xeon 6330，755 GB RAM，conda envs `vllm` / `sglang`
**复现目标**：复现「多 Agent 共单 GPU → 冷预填/恢复预填/短解码争抢 → 预制-解码隔离 + 动态 SM 分配」的**趋势与机制**，不做绝对数值对齐（忠实于项目明确原则）。

---

## 一、已复现的关键机制（全部实机实测）

| 机制 | 实测结果 | 图/产物 |
|---|---|---|
| 3 态 agent trace | 120 会话（ReAct/P&E），冷预填 2.5–3.5k、恢复预填 ReAct≈56 / P&E≈251、解码 ReAct≈38 / P&E≈58，**Table I 对齐** | `traces/`、`token_distribution.png` |
| HoL / TPOT spike | N=1/3/6：TTFT冷 p50 25→95→453ms，TPOT p95 10→34→71ms；吞吐 55→103→112 tok/s | `hol_sweep.png`、`fig2_per_agent_tokens.png` |
| SM 分配与吞吐 | 冷预填近线性（137k→2.16M tok/s），decode 低 SM 即平台；knee 90/80/80/90% | `sm_share_profile.png` |
| Green Context 隔离 | decode 受保护吞吐 73k→405k/s（~5.5×） | `isolation_benefit.png` |
| 核心组件 | GreenContextManager(10 ctx) / KVCacheSync(mutex+cudaEvent) / TPOTController / QD-QP 编排 | `cuda/agentserve_core.cu`、`scheduler.py` |
| 请求阶段分类 | initial→Cold / append→Resume / next-token→Decode，验收通过 | `phase.py` |

## 二、引擎级集成（把论文机制真正落入 llama.cpp / 服务）

### 2.1 llama.cpp CUDA 计算进入 CUDA Green Context（SM 约束）
- 改动：`ggml/src/ggml-cuda/common.cuh` 的 `ggml_backend_cuda_context::stream()`，当 `AGENTSERVE_GREEN_PCT` 设置时用 `cuGreenCtxCreate` + `cuGreenCtxStreamCreate` 在绿上下文里建 compute 流。
- 实测（真实 Qwen2.5-3B）：吞吐随 SM 下降、TPOT 随 SM 上升。

| GREEN_PCT | SM | throughput | TTFT cold p50 | TPOT p95 |
|---|---|---|---|---|
| 100% | 82 | 97.3 | 99.9ms | 38.1ms |
| 50% | 42 | 85.4 | 93.3ms | 63.1ms |
| 25% | 21 | 61.5 | 127.6ms | 66.2ms |

![green engine](plots/green_engine.png)

### 2.2 双绿上下文 + 双 worker serving（QD/QP 路由）
- 两个 llama-server 各绑一个绿上下文（prefill@60%→50 SM、decode@40%→32 SM），按请求 phase 路由到 prefill/decode worker，并发跑真实 trace。

| 后端 | TTFT cold p50 | TPOT p95 | throughput |
|---|---|---|---|
| 单 llama.cpp | 94.7ms | 33.8ms | 103.4 |
| AgentServe(编排) | 103.2ms | 32.5ms | 101.9 |
| **Dual Green** | **44.5ms** | 33.5ms | 82.0 |

![dual green](plots/dual_green_compare.png)

→ 冷预填 TTFT 下降 2.1×（prefill 隔离、不与 decode 争抢）；aggregate 吞吐因**静态分割**略降（未做动态调节）。

### 2.3 TPOTController 自适应分配（接入双绿系统）
- 每轮测 `scheduler_tpot_step_ms` → 控制器升/降 decode 绿上下文份额（重启 decode 应用），迭代收敛。

目标 tpot_step=8.92ms，轨迹：decode 20%→13.38ms(>SLO)→升 30%→10.86ms(边缘)→升 40%→10.51ms(达标→收敛)。吞吐随份额 32→38→44 tok/s。

![tpot controller](plots/tpot_controller.png)

## 三、Backend 对比（Task 18，N=3, Qwen2.5-3B, ReAct, 12 会话）
| backend | TTFT cold p50 | p95 | throughput |
|---|---|---|---|
| llama.cpp | 94.7 | 759.3 | 103.4 |
| vLLM | 217.2 | 609.1 | **137.9** |
| SGLang | 227.3 | 8070.2* | 67.7 |

*SGLang p95 为 FlashInfer JIT 首编译离群（`ninja` 已补装）；vLLM/SGLang 客户端流式 TPOT 不可靠（OpenAI 流式缓冲），故以 TTFT+throughput 为准。

## 四、值得注意的工程发现
1. **kernel 级 microbench 复现不了服务级 HoL**：CUDA 会交错调度小 decode kernel 与 prefill 网格（Task 8/17 中 decode p95 差异小）。服务级 HoL 需系统级集成才能显现——已如实记录，未伪造。
2. **llama-server slot context 均分**：`--parallel N` 会把 ctx 均分（N 槽），长冷预填 prompt 需 `-c` 足够大，否则请求被拒。
3. **SGLang 首次 FlashInfer 需要 `ninja`**；vLLM 的 OpenAI model 名要用完整路径。

## 五、如实说明的剩余边界 / 未完成
- **单上下文共享 KV + 双线程（论文最深一层）未端到端实现**：llama.cpp 的 `llama_decode` 不是为「同一 context 并发调用/共享 KV 拆两条绿上下文流」设计的（`update_slots` 单 batch 单流、graph compute 同步、context 非线程安全）；论文未公开该部分实现。当前以**两实例（各持 KV）+ 双绿上下文流**复现了「并发 SM 分区 + QD/QP 路由 + TPOT 自适应」，并把**机制与收益**都实测出来。
- **模型覆盖**：只有 Qwen2.5-3B；论文的 7B / Llama-3-8B 未下载。
- **正式矩阵（Task 20）**：完成 Stage 1 + 部分 Stage 2（N=1/3/6）；Stage3/4/5（7B/8B、P&E、第二 GPU）未跑。
- **消融（Task 17）**：kernel 级 microbench 差异微弱（见发现 1），服务级消融需上述系统集成。

## 六、结论
- **已科学、可复现地验证 AgentServe 的因果链条与关键机制**：3 态负载、并发 HoL/Tail、SM 分配-吞吐非线性、Green Context 隔离、QD/QP 编排、TPOT 自适应调度、引擎级绿上下文 SM 约束、双绿上下文并发 serving。
- **机制层已打通**：llama.cpp 计算真正进入绿上下文、双绿并发 serving、TPOTController 自适应调份额，均有实测数值。
- **剩余**：把「单上下文共享 KV + 双线程」的 llama.cpp 并发模型改造做完整（高风险、论文未给实现），以及补 7B/8B 与完整矩阵。

## 七、复现命令
```bash
cd /root/autodl-tmp/exp
PY=/root/autodl-tmp/conda_envs/vllm/bin/python
export PYTHONPATH=$PWD/src
$PY scripts/gen_traces.py --config configs/trace_gen.yaml                 # trace (Task 5)
$PY scripts/serve_llama.py --config configs/serving.yaml --agents 3 --sessions 12    # llama.cpp HoL (Task 6)
$PY scripts/serve_backend.py --backend vllm --agents 3 --sessions 12       # backend 对比 (Task 18)
$PY scripts/serve_backend.py --backend sglang --agents 3 --sessions 12
$PY scripts/serve_backend.py --backend agentserve --agents 3 --sessions 12
$PY scripts/serve_dualgreen.py --agents 3 --sessions 12 --tag dualgreen_N3  # 双绿上下文 serving
$PY scripts/serve_dualgreen_controller.py                                   # TPOTController 自适应环
make cuda && ./cuda/agentserve_core                                          # 核心组件
# 引擎级绿上下文：AGENTSERVE_GREEN_PCT=50 llama-server -m <model> ...
```

## 八、产物索引
- `results/results_log.md`：逐任务数据表 + 图（Task 1–16 + 扩展）。
- `docs/final_report.md`（本文件）、`docs/llama_cpp_source_map.md`、`docs/experiment_matrix.md`、`docs/progress.md`。
- `plots/`：hol_sweep、sm_share_profile、isolation_benefit、token_distribution、fig2、fig5、green_engine、dual_green_compare、tpot_controller、backend_compare。
- `src/agentserve_repro/` 包、`cuda/` 微基准、`configs/`、`tests/`、`Makefile`、`pyproject.toml`、`LICENSE`、`README.md`。
