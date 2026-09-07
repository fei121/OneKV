#!/usr/bin/env python3
"""Task 18 — backend comparison table + chart (N=3, Qwen2.5-3B, ReAct synthetic)."""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EXP=Path("/root/autodl-tmp/exp")
files={"llama.cpp":EXP/"results/metrics_N3.json",
       "vLLM":EXP/"results/vllm_N3_metrics.json",
       "SGLang":EXP/"results/sglang_N3_metrics.json"}
data={}
for name,f in files.items():
    data[name]=json.load(open(f))
    print(name, "ttft_cold_p50=%.1f p95=%.1f tpot_p50=%.4f p95=%.4f throughput=%.1f" % (
        data[name]["ttft_cold_ms"]["p50"], data[name]["ttft_cold_ms"]["p95"],
        data[name]["tpot_benchmark_ms"]["p50"], data[name]["tpot_benchmark_ms"]["p95"],
        data[name]["throughput_tokens_per_s"]))

# table
tab="| backend | TTFT_cold p50(ms) | p95(ms) | TPOT p50(ms) | p95(ms) | throughput(tok/s) |\n|---|---|---|---|---|---|\n"
for name in ["llama.cpp","vLLM","SGLang"]:
    d=data[name]
    tab+=f"| {name} | {d['ttft_cold_ms']['p50']:.1f} | {d['ttft_cold_ms']['p95']:.1f} | {d['tpot_benchmark_ms']['p50']:.3f} | {d['tpot_benchmark_ms']['p95']:.3f} | {d['throughput_tokens_per_s']:.1f} |\n"
(EXP/"results/task18_backend_compare.md").write_text("## Task 18 — backend comparison (N=3, Qwen2.5-3B, ReAct, 12 sessions)\n\n"+tab)

# chart
names=list(data.keys())
fig,axes=plt.subplots(1,2,figsize=(11,4))
tt=[data[n]['ttft_cold_ms']['p50'] for n in names]
th=[data[n]['throughput_tokens_per_s'] for n in names]
axes[0].bar(names,tt,color=['tab:blue','tab:orange','tab:green']); axes[0].set_ylabel("TTFT cold p50 (ms)"); axes[0].set_title("TTFT by backend")
axes[1].bar(names,th,color=['tab:blue','tab:orange','tab:green']); axes[1].set_ylabel("throughput (tok/s)"); axes[1].set_title("throughput by backend")
plt.tight_layout(); plt.savefig(EXP/"plots/backend_compare.png",dpi=130); plt.close()

# append to results_log
rl=EXP/"results/results_log.md"; s=rl.read_text()
note="\n## Task 18 — vLLM / SGLang baseline (backend comparison, N=3)\n\n"+tab+"\n![backend compare](../plots/backend_compare.png)\n"
if "## Task 18" not in s: rl.write_text(s+note)
print("wrote backend_compare.png + task18_backend_compare.md + results_log note")
