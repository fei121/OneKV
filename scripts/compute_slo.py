#!/usr/bin/env python3
"""Compute session-level SLO attainment rate from unified / engine event logs.

A session passes iff (cold TTFT <= tau_ttft) AND (session-level TPOT p95 <= tau_tpot).
Thresholds come from an isolated (N=1) model-device profile scaled by a constant factor
(see configs/metrics.yaml: scale_ttft / scale_tpot).

This is a server-side utility: it reads the per-N event logs that live under the experiment
root (engine *-jsonl under /tmp, baselines under raw_logs/) and writes
metrics/nscale-4way/slo_attainment.csv.

Usage (run inside the exp root on the bench server):
  python scripts/compute_slo.py <tau_ttft_ms> <tau_tpot_ms> [--base /root/autodl-tmp/exp]
"""
import argparse, collections, csv, json, os

def pctl(vals, q):
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]

def session_metrics_unified(path):
    evs = [json.loads(l) for l in open(path)]
    evs = [e for e in evs if isinstance(e, dict) and e.get("__meta__") is None]
    req = collections.defaultdict(dict)
    for e in evs:
        key = (e.get("session_id"), e.get("request_id"))
        r = req[key]; ev = e["event"]
        if ev == "REQUEST_ARRIVE": r["arrive"] = e["ts_ns"]
        elif ev == "TOKEN_EMIT": r.setdefault("tokens", []).append(e["ts_ns"])
    cold_ttft = {}
    for e in evs:
        if e["event"] == "REQUEST_ARRIVE" and e.get("phase") == "cold_prefill":
            sid, rid, arrive = e["session_id"], e["request_id"], e["ts_ns"]
            toks = sorted(req[(sid, rid)].get("tokens", []))
            if toks:
                cold_ttft[sid] = (toks[0] - arrive) / 1e6
    sessions = collections.defaultdict(list)
    for (sid, _rid), r in req.items():
        toks = sorted(r.get("tokens", []))
        for i in range(1, len(toks)):
            sessions[sid].append((toks[i] - toks[i - 1]) / 1e6)
    return {sid: {"ttft_cold": cold_ttft.get(sid),
                  "tpot_p95": pctl(g, 0.95) if g else None}
            for sid, g in sessions.items()}

def engine_metrics_unified(path):
    evs = [json.loads(l) for l in open(path)]
    sess = collections.defaultdict(lambda: {"ttft_cold": None, "gaps": []})
    for e in evs:
        sid = e.get("session"); ev = e.get("event")
        if not sid: continue
        if ev == "TTFT_cold": sess[sid]["ttft_cold"] = e["ms"]
        elif ev == "TPOT": sess[sid]["gaps"].append(e["ms"])
    return {sid: {"ttft_cold": d["ttft_cold"],
                  "tpot_p95": pctl(d["gaps"], 0.95) if d["gaps"] else None}
            for sid, d in sess.items()}

def slo(metrics, tau_ttft, tau_tpot):
    passed = total = 0
    for m in metrics.values():
        ct, tp = m["ttft_cold"], m["tpot_p95"]
        if ct is None or tp is None: continue
        total += 1
        if ct <= tau_ttft and tp <= tau_tpot: passed += 1
    return (passed / total) if total else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tau_ttft_ms", type=float)
    ap.add_argument("tau_tpot_ms", type=float)
    ap.add_argument("--base", default="/root/autodl-tmp/exp")
    ap.add_argument("--engine-dir", default="/tmp")
    ap.add_argument("--out", default="metrics/nscale-4way/slo_attainment.csv")
    args = ap.parse_args()
    N = list(range(3, 11))
    rows = []
    for n in N:
        eng = engine_metrics_unified(f"{args.engine_dir}/eng_N{n}.jsonl")
        row = {"N": n, "engine": slo(eng, args.tau_ttft_ms, args.tau_tpot_ms)}
        for name in ("vllm", "sglang", "llama"):
            p = f"{args.base}/raw_logs/{name}_N{n}_events.jsonl"
            if os.path.exists(p):
                row[name] = slo(session_metrics_unified(p), args.tau_ttft_ms, args.tau_tpot_ms)
            else:
                row[name] = None
        rows.append(row)
    cols = ["N", "engine", "vllm", "sglang", "llama"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.2f}" if r[c] is not None else "") for c in cols})
    print("wrote", args.out)
    for r in rows:
        print(f"{r['N']:>3} | engine {r['engine']:.2f} | vllm {r['vllm']} | sglang {r['sglang']} | llama {r['llama']}")

if __name__ == "__main__":
    main()
