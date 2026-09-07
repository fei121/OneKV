import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
# Same-methodology comparison. Baseline cold TTFT = llama-server TRUE cold measured on same 2693-tok prompt (419.6ms).
data = {
 "N=3": {"llama.cpp baseline":{"TTFT":419.6,"TPOT":33.8,"TP":103.4}, "AgentServe shared-KV":{"TTFT":374,"TPOT":12.68,"TP":167.9}},
 "N=6": {"llama.cpp baseline":{"TTFT":419.6,"TPOT":71.4,"TP":111.9}, "AgentServe shared-KV":{"TTFT":383,"TPOT":12.96,"TP":271.2}},
}
plt.rcParams.update({"font.size":10,"axes.titlesize":12})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
colors={"llama.cpp baseline":"#c0392b","AgentServe shared-KV":"#00a86b"}
for ax,(title,key) in zip(axes, [("TPOT p95 (ms)","TPOT"),("TTFT_cold p50 (ms)","TTFT"),("throughput (tok/s)","TP")]):
    ns=["N=3","N=6"]; names=["llama.cpp baseline","AgentServe shared-KV"]; x=np.arange(2); w=0.38
    for j,nm in enumerate(names):
        vals=[data[n][nm][key] for n in ns]
        ax.bar(x+(j-0.5)*w, vals, w, label=nm, color=colors[nm])
        for i,n in enumerate(ns): ax.text(x[i]+(j-0.5)*w, vals[i]*1.05, "%.1f"%vals[i], ha="center", va="bottom", fontsize=8)
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(ns); ax.set_ylabel(title); ax.grid(alpha=0.3, axis="y")
axes[0].legend(); plt.tight_layout()
plt.savefig("/root/autodl-tmp/exp/plots/backend_compare_sharedkv_3b.png", dpi=140, facecolor="white"); print("ok")
