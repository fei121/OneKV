#!/usr/bin/env python3
"""Per-paradigm 4-way N-scale figure. Reads metrics/perparadigm/{model}.json where model=react|pe|react7|pe7."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def plot_one(para, outname):
    data=json.load(open(os.path.join(REPO, f"metrics/perparadigm/{para}.json")))
    N=sorted(int(k) for k in data)
    names=["engine","llama","vllm","sglang"]
    labels={"engine":"shared-KV engine","llama":"llama.cpp","vllm":"vLLM","sglang":"SGLang"}
    colors={"engine":"#00a86b","vllm":"#8e44ad","sglang":"#2980b9","llama":"#c0392b"}
    fig,axes=plt.subplots(1,3,figsize=(16,4.4))
    panels=[("throughput","throughput (tok/s)"),("tpot_p95","TPOT p95 (ms)"),("ttft_cold","cold TTFT p50 (ms)")]
    for ax,(key,ylabel) in zip(axes,panels):
        for name in names:
            ys=[data[str(n)][name][key] for n in N]
            ax.plot(N,ys,"-o",color=colors[name],lw=1.8,label=labels[name])
        ax.set_xlabel("N (concurrent agents)"); ax.set_ylabel(ylabel)
        ax.set_title({"throughput":"Throughput vs N","tpot_p95":"TPOT p95 vs N","ttft_cold":"Cold TTFT vs N"}[key])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out=os.path.join(REPO,f"figures/{outname}.png")
    fig.savefig(out,dpi=140,facecolor="white"); print("wrote",out)
if __name__=="__main__":
    jobs=[("react","react-4way-nscale"),("pe","pe-4way-nscale"),
          ("react7","react7-4way-nscale"),("pe7","pe7-4way-nscale")]
    for para,out in jobs: plot_one(para,out)
