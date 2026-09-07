import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
data = {
    "N=3": {
        "llama.cpp baseline": {"TTFT": 94.7, "TPOT": 33.8, "TP": 103.4},
        "AgentServe shared-KV": {"TTFT": 969, "TPOT": 9.66, "TP": 68.5},
    },
    "N=6": {
        "llama.cpp baseline": {"TTFT": 452.7, "TPOT": 71.4, "TP": 111.9},
        "AgentServe shared-KV": {"TTFT": 2133, "TPOT": 10.47, "TP": 61.3},
    },
}
plt.rcParams.update({"font.size":10,"axes.titlesize":12})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
colors = {"llama.cpp baseline":"#c0392b", "AgentServe shared-KV":"#00a86b"}
for ax,(title,key) in zip(axes, [("TPOT p50 (ms)","TPOT"),("TTFT_cold p50 (ms)","TTFT"),("throughput (tok/s)","TP")]):
    ns=["N=3","N=6"]; names=["llama.cpp baseline","AgentServe shared-KV"]; x=np.arange(len(ns)); w=0.38
    for j,nm in enumerate(names):
        vals=[data[n][nm][key] for n in ns]
        ax.bar(x+(j-0.5)*w, vals, w, label=nm, color=colors[nm])
        for i,n in enumerate(ns): ax.text(x[i]+(j-0.5)*w, vals[i]*1.05, "%.1f"%vals[i], ha="center", va="bottom", fontsize=8)
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(ns); ax.set_ylabel(title); ax.grid(alpha=0.3, axis="y")
    if key=="TTFT": ax.set_yscale("log")
axes[0].legend()
plt.tight_layout()
plt.savefig("/root/autodl-tmp/exp/plots/backend_compare_sharedkv_3b.png", dpi=140, facecolor="white")
print("saved plot (batched)")
