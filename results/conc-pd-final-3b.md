# 单引擎共享 KV 并发 P/D 管道 — 最终结果（Qwen2.5-3B）

## 机制（已实现，对应论文 Execution Layer）
- 两 context 共享 KV（A 预填 / B 解码，`ctx_other=A`）。
- 两个真实线程：预填线程(A) + 解码线程(B)，`cudaEvent` 保证解码读到预填完成的 KV，互斥锁保护共享 cell 的 CPU bookkeeping。
- `cudaDeviceSynchronize` 移出锁，使 A 预填 GPU 与 B 解码 GPU 真正重叠。

## 干净结果：N=3（无绿上下文，prefill=default/decode=stream1）
| 指标 | 值 |
|---|---|
| throughput | 68.4 tok/s |
| TTFT_cold p50 / p95 | 1396 / 2110 ms |
| **TPOT p50 / p95** | **9.65 / 10.16 ms** |
| TTFT_resume p50 / p95 | 814 / 1564 ms |
| decode tokens | 486 |

- **解码 TPOT 高度稳定**（9.65ms），显著优于基线 N=3 的 TPOT p95=33.8ms —— 共享 KV + 并发 P/D 的隔离收益真实存在。
- throughput 68.4 tok/s，高于顺序版 40.5 tok/s，体现并发重叠。

## N=6：仍有竞态（不可靠）
- TTFT_cold p50=4249ms；**TPOT p50=0.03ms**（部分 decode no-op）→ 6 会话语义下双线程共享 cell bookkeeping 存在竞态（部分解码失败），需更强的锁/顺序保证。

## 剩余（未完成）
1. **N=6 竞态**：多会话下共享 cell bookkeeping 的线程安全（当前锁不够/顺序不对）。
2. **冷预填串行（TTFT 高）**：batched prefill 的 per-session logits 提取不可靠，未采用；需正确拿到 `llama_get_logits_ith` 的 output_ids，或换 chunked/batched 调度。
3. **绿上下文非对称 SM 分区**未做。
4. **完整三基线公平对比 + 可视化**未统跑。

## 代码
- `exp_stage/llama_src/as_conc.cpp`；补丁见 `exp_stage/llama_patches/`。
