#!/usr/bin/env python3
"""Cross-model overview: Qwen2.5-3B (solid) vs Qwen2.5-7B (dashed) per backend/metric.

Reads metrics/perparadigm/{react,pe,react7,pe7}.json (4-way N-scale, tool_wait=0) and plots, for ReAct
and P&E, throughput / TPOT p95 / cold TTFT vs N. Each backend is a colour; solid = 3B, dashed = 7B.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name):
    return json.load(open(os.path.join(REPO, f"metrics/perparadigm/{name}.json")))

# DATA[(model, para)] = {N: {backend: {metric}}}
DATA = {}
for m in ("3B", "7B"):
    for para in ("react", "pe"):
        fn = f"{para}{'7' if m == '7B' else ''}"
        DATA[(m, para)] = load(fn)

BACKENDS = ["engine", "llama", "vllm", "sglang"]
LABEL = {"engine": "shared-KV engine", "llama": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
COLOR = {"engine": "#00a86b", "vllm": "#8e44ad", "sglang": "#2980b9", "llama": "#c0392b"}
METRICS = [("throughput", "throughput (tok/s)"), ("tpot_p95", "TPOT p95 (ms)"),
           ("ttft_cold", "cold TTFT p50 (ms)")]

Ns = [3, 4, 5, 6]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
for r, para in enumerate(["react", "pe"]):
    for c, (key, ylab) in enumerate(METRICS):
        ax = axes[r][c]
        for b in BACKENDS:
            for m, ls in [("3B", "-"), ("7B", "--")]:
                d = DATA[(m, para)]
                ys = [d[str(n)][b][key] for n in Ns]
                ax.plot(Ns, ys, ls, color=COLOR[b], lw=1.8,
                        label=f"{LABEL[b]} ({m})" if (c == 0 and r == 0) else None)
        ax.set_xlabel("N (concurrent agents)")
        ax.set_ylabel(ylab)
        ax.set_xticks(Ns)
        ax.set_title(f"{para} · {key}", fontsize=10)
        ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=7, ncol=4, loc="upper left")
fig.suptitle("Cross-model overview: Qwen2.5-3B (solid) vs Qwen2.5-7B (dashed) — N=3…6, tool_wait=0",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(REPO, "figures/crossmodel-overview.png")
fig.savefig(out, dpi=140, facecolor="white")
print("wrote", out)
