#!/usr/bin/env python3
"""Plot the 4-way N-scale comparison (throughput + TPOT p95) for Qwen2.5-3B.

Reads committed per-N backend metrics and the engine summary CSV, then renders:
  - left panel : throughput (tok/s) vs N
  - right panel: TPOT p95 (ms) vs N  -- decode stability under concurrency

Run:
    python scripts/plot_4way_nscale.py --out figures/backend-4way-nscale-3b.png

Sources (all under metrics/nscale-4way/):
  - engine : engine_summary.csv (from engine DONE line + TS-per-token event logs)
  - baselines: {llama,vllm,sglang}_N{N}.json (N=3..10, 12-session, per-token TPOT)
"""
import argparse, csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RANGE = list(range(3, 11))
SUBDIR = os.path.join("metrics", "nscale-4way")


def _load(path):
    with open(os.path.join(REPO, path)) as f:
        return json.load(f)


def engine_table(path):
    tbl = {}
    with open(os.path.join(REPO, path)) as f:
        for row in csv.DictReader(f):
            tbl[int(row["N"])] = (float(row["throughput_tok_s"]), float(row["tpot_p95_ms"]))
    return tbl


def baseline_table(name):
    tbl = {}
    for N in N_RANGE:
        p = os.path.join(SUBDIR, f"{name}_N{N}.json")
        if not os.path.exists(os.path.join(REPO, p)):
            continue
        d = _load(p)
        tbl[N] = (d["throughput_tokens_per_s"], (d.get("tpot_benchmark_ms") or {}).get("p95"))
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/backend-4way-nscale-3b.png")
    ap.add_argument("--engine-csv",
                    default=os.path.join(SUBDIR, "engine_summary.csv"))
    args = ap.parse_args()

    eng = engine_table(args.engine_csv)
    tables = {"engine": eng, "llama": baseline_table("llama"),
              "vllm": baseline_table("vllm"), "sglang": baseline_table("sglang")}

    colors = {"engine": "#00a86b", "vllm": "#8e44ad", "sglang": "#2980b9", "llama": "#c0392b"}
    labels = {"engine": "shared-KV engine", "vllm": "vLLM", "sglang": "SGLang", "llama": "llama.cpp"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    def plot(ax, key, idx):
        tbl = tables[key]
        xs = sorted(tbl)
        ys = [tbl[N][idx] for N in xs if tbl[N][idx] is not None]
        xs = [N for N in xs if tbl[N][idx] is not None]
        ax.plot(xs, ys, "-o", color=colors[key], lw=1.8, label=labels[key])

    for key in ("engine", "vllm", "sglang", "llama"):
        plot(axes[0], key, 0)
    axes[0].set_xlabel("N (concurrent agents)")
    axes[0].set_ylabel("throughput (tok/s)")
    axes[0].set_title("Throughput vs N (Qwen2.5-3B)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for key in ("engine", "vllm", "sglang", "llama"):
        plot(axes[1], key, 1)
    axes[1].set_xlabel("N (concurrent agents)")
    axes[1].set_ylabel("TPOT p95 (ms)")
    axes[1].set_title("TPOT p95 vs N — decode stability (Qwen2.5-3B)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
