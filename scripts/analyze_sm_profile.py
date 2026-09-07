#!/usr/bin/env python3
"""Analyze SM share profile: normalize curves, find saturation knee, plot, update results log."""
import sys, json, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EXP = Path("/root/autodl-tmp/exp")
THRESH = yaml.safe_load(open(EXP/"configs/profiling.yaml"))["saturation_knee_threshold"]

def parse(txt):
    rows=[]
    for line in txt.splitlines():
        line=line.strip()
        if line.startswith("share_pct") or not line or line.startswith("device"):
            continue
        parts=line.split()
        if len(parts)<10: continue
        pct=int(parts[0]); sm=int(parts[1])
        pref,rr,rp,dec = [float(x) for x in parts[2:6]]
        rows.append({"share":pct,"sm":sm,"prefill":pref,"resume_react":rr,"resume_pe":rp,"decode":dec})
    return rows

def norm_curves(rows):
    base={k: rows[-1][k] for k in ("prefill","resume_react","resume_pe","decode")}
    out={}
    for k in base:
        out[k]=[r[k]/base[k] for r in rows]
    return base, out

def knee(rows, curve, thresh):
    for r, val in zip(rows, curve):
        if val >= thresh:
            return r["share"]
    return rows[-1]["share"]

def main(txtfile):
    rows = parse(Path(txtfile).read_text())
    base, curves = norm_curves(rows)
    knees = {k: knee(rows, curves[k], THRESH) for k in curves}
    res = {"sm_profile": rows, "throughput_100pct": base,
           "normalized": curves,
           "saturation_knee_threshold": THRESH,
           "saturation_knee_share_pct": knees}
    (EXP/"results/sm_profile.json").write_text(json.dumps(res, indent=2))

    # plot normalized curves
    fig, ax = plt.subplots(figsize=(9,5))
    shares=[r["share"] for r in rows]
    for k, label in [("prefill","Cold Prefill"),("resume_pe","Resume P&E (251)"),
                     ("resume_react","Resume ReAct (56)"),("decode","Decode")]:
        ax.plot(shares, curves[k], "o-", label=label)
    ax.axhline(THRESH, color="gray", ls="--", label=f"knee threshold {THRESH}")
    ax.set_xlabel("SM share (%)"); ax.set_ylabel("normalized throughput")
    ax.set_title("Task 7 — SM share profiling (normalized to 100% SMs)")
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(EXP/"plots/sm_share_profile.png", dpi=130); plt.close()

    # append Task 7 to results log
    md = EXP/"results/results_log.md"
    text = md.read_text()
    kt = " | ".join(f"{k}={v}%" for k,v in knees.items())
    section = f"""
## Task 7 — SM share profiling (done)

**方法**：CUDA Green Context 划分 SM（10%–100%，CC8.x 按 [4, 偶数] 对齐），分别在绿上下文内用 cuBLAS(`cublasSetSmCountTarget`) 跑代表性 GEMM：Cold Prefill(M=3000)、Resume ReAct(M=56)、Resume P&E(M=251)、Decode(M=16, batch decode)。

| share% | 实际SM | 冷预填 tok/s | 恢复ReAct | 恢复P&E | decode tok/s |
|---|---|---|---|---|---|
""" + "\n".join(f"| {r['share']} | {r['sm']} | {r['prefill']:.0f} | {r['resume_react']:.0f} | {r['resume_pe']:.0f} | {r['decode']:.0f} |" for r in rows) + f"""

- 归一化（除以 100% 值）后曲线：`plots/sm_share_profile.png`
- **Saturation knee**（threshold={THRESH}）：{kt}
- 结论：M 越大越吃 SM（冷预填近线性），decode 低 SM 即平台化；此曲线是 Resource-Aware Scheduler 的 `µP(S−R)`/`µD(R)` 输入，knee 用于初始化 R_base（ENGINEERING-CHOICE）。

![sm share profile](../plots/sm_share_profile.png)
"""
    if "## Task 7" not in text:
        md.write_text(text + section)
    print("knees:", knees)
    print("wrote sm_profile.json + sm_share_profile.png + updated results_log.md")

if __name__ == "__main__":
    main(sys.argv[1])
