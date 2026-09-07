#!/usr/bin/env python3
"""Generate results/results_log.md + plots/*.png from measured data (backfill Tasks 1..6)."""
import json, os, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EXP = Path("/root/autodl-tmp/exp")
PLOTS = EXP / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
RES = EXP / "results"; RES.mkdir(parents=True, exist_ok=True)

def load(p):
    return json.load(open(p))

# ---------- Task 6 HoL sweep data ----------
Ns = [1, 3, 6]
metrics = {N: load(RES / f"metrics_N{N}.json") for N in Ns}

# ---------- Charts ----------
def chart_hol():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(Ns, [metrics[N]["ttft_cold_ms"]["p50"] for N in Ns], "o-", label="TTFT cold p50")
    axes[0].plot(Ns, [metrics[N]["ttft_cold_ms"]["p95"] for N in Ns], "s-", label="TTFT cold p95")
    axes[0].set_xlabel("Concurrent agents N"); axes[0].set_ylabel("TTFT (ms)")
    axes[0].set_title("TTFT (cold prefill) vs N"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(Ns, [metrics[N]["tpot_benchmark_ms"]["p50"] for N in Ns], "o-", label="TPOT p50")
    axes[1].plot(Ns, [metrics[N]["tpot_benchmark_ms"]["p95"] for N in Ns], "s-", label="TPOT p95")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("TPOT (ms)")
    axes[1].set_title("TPOT vs N"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(Ns, [metrics[N]["throughput_tokens_per_s"] for N in Ns], "o-", label="throughput")
    axes[2].set_xlabel("N"); axes[2].set_ylabel("tokens/s")
    axes[2].set_title("Throughput vs N"); axes[2].legend(); axes[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(PLOTS / "hol_sweep.png", dpi=130); plt.close()

def chart_dist():
    dist = load(EXP / "traces" / "distribution_summary.json")
    m = dist["measured"]
    cats = ["react", "plan_and_execute"]
    phases = ["resume_prefill", "decode"]
    x = [f"{p[:3]}-{ph}" for p in cats for ph in phases]
    meas = [m[p][ph]["avg"] for p in cats for ph in phases]
    tgt = [dist["table1_target"][p]["resume_avg" if ph == "resume_prefill" else "decode_avg"] for p in cats for ph in phases]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    import numpy as np
    xpos = np.arange(len(x)); w = 0.38
    ax.bar(xpos - w/2, meas, w, label="measured avg")
    ax.bar(xpos + w/2, tgt, w, label="Table I target avg")
    ax.set_xticks(xpos); ax.set_xticklabels(x, rotation=20)
    ax.set_ylabel("tokens"); ax.set_title("Token distribution: measured vs Table I (resume & decode)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(PLOTS / "token_distribution.png", dpi=130); plt.close()

chart_hol(); chart_dist()

# ---------- Markdown ----------
def fmt_ms(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"

rows = []
for N in Ns:
    m = metrics[N]
    rows.append(f"| **{N}** | {fmt_ms(m['ttft_cold_ms']['p50'])} | {fmt_ms(m['ttft_cold_ms']['p95'])} | "
                f"{fmt_ms(m['tpot_benchmark_ms']['p50'])} | {fmt_ms(m['tpot_benchmark_ms']['p95'])} | "
                f"{m['throughput_tokens_per_s']:.1f} | {m['throughput_excl_tool_wait_tokens_per_s']:.1f} | "
                f"{m['wall_clock_s']:.1f} |")
hol_table = "\n".join(rows)

dist = load(EXP / "traces" / "distribution_summary.json")
m = dist["measured"]; t = dist["table1_target"]
dm = [
    ("react", "cold_prefill", "2693", "2500–3500"),
    ("react", "resume_prefill", f"{m['react']['resume_prefill']['avg']}", f"30–127 avg~{t['react']['resume_avg']}"),
    ("react", "decode", f"{m['react']['decode']['avg']}", f"27–99 avg~{t['react']['decode_avg']}"),
    ("plan_and_execute", "cold_prefill", "2701", "2500–3500"),
    ("plan_and_execute", "resume_prefill", f"{m['plan_and_execute']['resume_prefill']['avg']}", f"125–421 avg~{t['plan_and_execute']['resume_avg']}"),
    ("plan_and_execute", "decode", f"{m['plan_and_execute']['decode']['avg']}", f"41–125 avg~{t['plan_and_execute']['decode_avg']}"),
]
dist_rows = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in dm)

md = f"""# AgentServe 复现实验记录

> 记录每次实验任务的**实测结果**（数据表 + 可视化图）。统一存放 `results/results_log.md`，图在 `plots/`。
> 设备：RTX 3090 (24GB/82SM/CC8.6)，CUDA 12.8 + driver 580.142，Ubuntu 22.04.5；数据盘 `/root/autodl-tmp`。

---

## Task 1 — 环境/硬件盘点 (done)

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, CC 8.6, 82 SM, 350W |
| CUDA | toolkit 12.8 (`/usr/local/cuda-12.8`), driver 580.142, runtime 13.0 |
| OS / CPU / RAM | Ubuntu 22.04.5 / Intel Xeon Gold 6330 (56C/112T) / 755 GiB |
| 模型 | `Qwen2.5-3B/` (safetensors 5.8G) + `Qwen2.5-3B-f16.gguf` (6.17G) |
| conda envs | `vllm` (vllm 0.16.0, torch 2.9.1+cu128), `sglang` (sglang 0.5.19, torch 2.13+cu130), `base` |
| 工具 | git / gcc 11.4 / cmake 3.22 / aria2 / jq / hfd.sh |

## Task 2 — CUDA Green Context 可行性 (done, PASS)

用 CUDA Driver API 实机验证，结果：

```
GPU: compute capability 8.6, 82 SMs
Split -> 1 groups: group 0: 42 SMs
Green context created.
Green context stream created.
Kernel executed in green context, result=7000000 (expected 7000000)
RESULT: PASS
```

结论：RTX 3090（CC 8.6）支持 CUDA Green Context（`cuDeviceGetDevResource`→`cuDevSmResourceSplit`→`cuGreenCtxCreate`→`cuCtxFromGreenCtx`→`cuGreenCtxStreamCreate`）。原计划 BLOCKER 解除。

## Task 3/4 — llama.cpp baseline + 统一 metrics/logger (done)

- 引擎：vanilla llama.cpp (0.4.0-dev)，CUDA offload (`-ngl 99`)，模型 Qwen2.5-3B F16 GGUF。
- harness：`scripts/serve_llama.py`（`llama-server` + N 并发 + slot 绑定 + 每 token 打点）；`scripts/events.py`（统一事件模型）；`scripts/metrics.py`（TTFT / benchmark TPOT / scheduler TPOT_step / throughput / SLO）。
- 事件模型已落地并验证：`raw_logs/events_N*.jsonl`(365 行/run)。

## Task 5 — 3 态 token trace (done, Table I 对齐)

- 来源：StableToolBench (MirrorAPI-Bench, test_cot) → 310 工具 schema + 600 工具返回；Qwen2.5-3B tokenizer。
- 120 会话（react=60, plan_and_execute=60），每会话 = 1 冷预填 + 2–4 轮(恢复预填+短解码)。

| 范式 | 阶段 | 实测 avg | Table I 目标 |
|---|---|---|---|
{dist_rows}

![token distribution](../plots/token_distribution.png)

## Task 6 — HoL / TPOT spike 现象 (done, first pass)

**实验**：llama.cpp baseline，Qwen2.5-3B F16，每档 12 会话，tool_wait=200ms，并发 N=1/3/6。

| N | TTFT冷 p50(ms) | TTFT冷 p95(ms) | TPOT p50(ms) | TPOT p95(ms) | 吞吐(tok/s) | 吞吐(不含tool_wait) | wall(s) |
|---|---|---|---|---|---|---|---|
{hol_table}

![hol sweep](../plots/hol_sweep.png)

**结论**：并发上升 → 冷预填 TTFT p50 25→95→453ms（~18×）、TPOT p95 10→34→71ms（~7×）；吞吐 55→103→112 tok/s（温和上升）。典型 head-of-line blocking：延迟稳定性恶化、吞吐微升。

---
_Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_
"""

(RES / "results_log.md").write_text(md)
print("wrote", RES / "results_log.md")
print("wrote", PLOTS / "hol_sweep.png")
print("wrote", PLOTS / "token_distribution.png")
