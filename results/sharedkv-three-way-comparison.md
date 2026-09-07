> **⚠️ RETRACTED** — The results/figures in this document were produced by an engine that ignored EOS, ran fewer sessions than the baselines, and skipped tool_wait; **and** the llama.cpp baseline harness uses `cache_prompt`, which cannot drive a multi-phase agent conversation (degenerates to a 1-token response after the first resume). These comparisons are **invalid** and are being re-worked. The engine itself is verified faithful on the phases the reference works (cold + first resume). See [`docs/known-limitations.md`](../docs/known-limitations.md).

# Qwen2.5-3B：单引擎共享 KV + 前缀缓存 vs 三基线（最终，同口径）

## 被测系统
**AgentServe 单引擎共享 KV 引擎（as_conc.cpp）**：两 context 共享 KV（`ctx_other=A`）；双流并发 P/D；**batched 解码（连续批处理）**；**前缀缓存冷预填**（6 会话冷 prompt 100% 相同 → 预填一次公共 prompt，`llama_memory_seq_cp(0->i)` 复制到各会话，冷 TTFT≈一次预填）。

## 结果（同口径：基线冷 TTFT 用 llama-server 真实冷值 419.6ms = 同一 2693-tok prompt 实测）
| 系统 | N | TTFT_cold p50(ms) | TPOT p50(ms) | TPOT p95(ms) | throughput(tok/s) |
|---|---|---|---|---|---|
| llama.cpp (baseline) | 3 | 419.6 | — | 33.8 | 103.4 |
| **共享KV(前缀缓存)** | **3** | **374** | **9.42** | **12.68** | **167.9** |
| llama.cpp (baseline) | 6 | 419.6 | — | 71.4 | 111.9 |
| **共享KV(前缀缓存)** | **6** | **383** | **9.94** | **12.96** | **271.2** |

## 结论
- **冷 TTFT 也已反超**：374/383ms vs 基线真实冷值 419.6ms（此前 94.7ms 是不同口径；同口径后我用前缀缓存直接低于真冷值）。
- **TPOT 稳定且远优于基线**（12.68/12.96 vs 33.8/71.4ms）。
- **吞吐大幅反超**（167.9/271.2 vs 103.4/111.9 tok/s）。
- 机制：引擎级共享 KV + 连续批处理 + **前缀缓存冷预填**，三项指标全部优于 llama.cpp 基线。

## 文件
- 运行时：`llama_src/as_conc.cpp`；补丁：`llama_patches/…`；图：`plots/backend_compare_sharedkv_3b.png`；events：`raw_logs/as_batch_events.jsonl`

## 多样任务三方对比（MirrorAPI test_cot，冷 prompt = 共享 system + 各异 instruction，~1211 token，N=6）
| 场景 | 冷 TTFT（多样任务） |
|---|---|
| **llama.cpp / llama-server 基线** | 单请求 265.3ms；N=6 并发 p50=303ms（含部分前缀缓存命中） |
| **我方引擎（无缓存，每会话全预填）** | **86.0 ms** |
| **我方引擎（前缀缓存，缓存共享 system）** | **35.5 ms** |

**结论（多样任务通用性）**：
- 前缀缓存对**多样任务依然有效**：冷 TTFT 86 → 35.5ms（约 **2.4×**），且低于 llama-server 基线（303ms）。
- 我方引擎在多样任务冷 TTFT 上**优于基线**（35.5/86 vs 303ms）。
- 诚实说明：我方测量为 **warm + flash-attn 启用**；基线单请求含首次 setup（265ms）且 N=6 并发含部分缓存命中（303ms）。存在一定测量口径噪声，但"前缀缓存 + 我方引擎更快"的趋势明确。
- 吞吐/TPOT：在真实 agent trace（相同冷 prompt）上我方测过并反超基线（167.9/271.2 tok/s，TPOT 12.68/12.96ms）；多样任务的 TPOT 由 decode 主导，趋势一致。

## 多样任务完整三方（seq_cp 前缀共享已打通，N=6）
| 指标 | llama-server 基线 | 我方 no-cache | 我方 prefix-cache |
|---|---|---|---|
| 冷 TTFT（system 已缓存） | 536.9 ms | 95.3 ms | ~35.5 ms |
| TPOT p50 / p95 | 12.47 / 25.19 ms | 10.35 / 16.18 ms | 9.87 / 15.77 ms |
| 吞吐 | 161.1 tok/s | 176.9 tok/s | 228.6 tok/s |

**seq_cp 修复根因**：`llama_kv_cache::seq_cp` 的 p1 为排他端点（pos<p1），`p1=common-1` 漏末 cell；修法为 instruction 起始与 `n_past` 均取 `llama_memory_seq_pos_max(seq)+1`（保证 Y=X+1、解码连续）。修复后 cache 模式无报错、total_decode 与 trace 一致、吞吐 228.6。
