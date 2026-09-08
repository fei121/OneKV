#!/usr/bin/env python3
"""Context-window sensitivity figure for the 4-way benchmark (N=6, 3B).

Shows engine / vLLM / SGLang / llama.cpp throughput, TPOT p95 and cold TTFT across three context
windows (32768 / 49152 / 65536) for ReAct and P&E. Proves that once context is adequate, every
backend is on a plateau — so cross-backend gaps are real, not a context artifact.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(REPO, "metrics/context_windows/n6_ctx_windows.json")))
WINDOWS = ["32768", "49152", "65536"]
BACKENDS = ["engine", "sglang", "vllm", "llama"]
LABEL = {"engine": "shared-KV engine", "sglang": "SGLang", "vllm": "vLLM", "llama": "llama.cpp"}
COLOR = {"engine": "#00a86b", "vllm": "#8e44ad", "sglang": "#2980b9", "llama": "#c0392b"}
PANELS = [("throughput", "throughput (tok/s)"), ("tpot_p95", "TPOT p95 (ms)"),
          ("ttft_cold", "cold TTFT p50 (ms)")]

fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
for r, para in enumerate(["react", "pe"]):
    for c, (key, ylab) in enumerate(PANELS):
        ax = axes[r][c]
        for b in BACKENDS:
            ys = [DATA[para][b][w][key] for w in WINDOWS]
            xs = [int(w) for w in WINDOWS]
            ax.plot(xs, ys, "-o", color=COLOR[b], lw=1.8, label=LABEL[b])
        ax.set_xlabel("context window (tokens)")
        ax.set_ylabel(ylab)
        ax.set_xticks([int(w) for w in WINDOWS]); ax.set_xticklabels(WINDOWS)
        ax.set_title(f"{para} · {key}", fontsize=10)
        ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8, loc="best")
fig.suptitle("Context-window sensitivity (N=6, Qwen2.5-3B) — every backend is flat once context is adequate",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(REPO, "figures/context-windows-n6.png")
fig.savefig(out, dpi=140, facecolor="white")
print("wrote", out)
