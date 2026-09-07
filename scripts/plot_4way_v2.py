#!/usr/bin/env python3
"""Plot the 4-way N-scale comparison (real-task trace, fixed baselines).

Reads metrics/v2/v2_fourway.json (throughput / TPOT p95 / TTFT_cold for engine, llama.cpp,
vLLM, SGLang at N=3..10) and renders three panels.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    data = json.load(open(os.path.join(REPO, "metrics/v2/v2_fourway.json")))
    N = sorted(int(k) for k in data)
    names = ["engine", "llama", "vllm", "sglang"]
    labels = {"engine": "shared-KV engine", "llama": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
    colors = {"engine": "#00a86b", "vllm": "#8e44ad", "sglang": "#2980b9", "llama": "#c0392b"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = [("throughput", "throughput (tok/s)"),
              ("tpot95", "TPOT p95 (ms)"),
              ("ttft", "cold TTFT p50 (ms)")]
    for ax, (key, ylabel) in zip(axes, panels):
        for name in names:
            xs = N
            ys = [data[str(n)][key][name] for n in xs]
            ax.plot(xs, ys, "-o", color=colors[name], lw=1.8, label=labels[name])
        ax.set_xlabel("N (concurrent agents)")
        ax.set_ylabel(ylabel)
        ax.set_title({"throughput": "Throughput vs N", "tpot95": "TPOT p95 vs N",
                      "ttft": "Cold TTFT vs N"}[key])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(REPO, "figures/v2-4way-nscale.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("wrote", out)

if __name__ == "__main__":
    main()
