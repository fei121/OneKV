#!/usr/bin/env python3
"""Plot 4-way session-level SLO attainment rate vs concurrency (Qwen2.5-3B).

A session passes SLO if its cold TTFT <= tau_ttft AND its session-level TPOT p95 <= tau_tpot,
where the thresholds derive from an isolated (N=1) model profile scaled by a constant factor.

Run:
    python scripts/plot_4way_slo.py --out figures/4way-nscale-slo.png

Data: metrics/nscale-4way/slo_attainment.csv
"""
import argparse, csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/4way-nscale-slo.png")
    ap.add_argument("--csv", default=os.path.join("metrics", "nscale-4way", "slo_attainment.csv"))
    args = ap.parse_args()

    data = {"engine": {}, "vllm": {}, "sglang": {}, "llama": {}}
    with open(os.path.join(REPO, args.csv)) as f:
        for row in csv.DictReader(f):
            n = int(row["N"])
            for k in data:
                data[k][n] = float(row[k])

    colors = {"engine": "#00a86b", "vllm": "#8e44ad", "sglang": "#2980b9", "llama": "#c0392b"}
    labels = {"engine": "shared-KV engine", "vllm": "vLLM", "sglang": "SGLang", "llama": "llama.cpp"}
    order = ["engine", "vllm", "sglang", "llama"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for key in order:
        xs = sorted(data[key])
        ys = [data[key][n] for n in xs]
        ax.plot(xs, ys, "-o", color=colors[key], lw=1.9, label=labels[key])
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("N (concurrent agents)")
    ax.set_ylabel("SLO attainment rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Session-level SLO attainment vs N (Qwen2.5-3B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
