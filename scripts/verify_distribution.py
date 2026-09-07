#!/usr/bin/env python3
"""Recompute the trace token distribution and compare against paper Table I."""
import json, sys, collections
from pathlib import Path

TABLE1 = {
    "react":            {"cold_prefill": (2500, 3500, None),
                          "resume_prefill": (30, 127, 56),
                          "decode": (27, 99, 38)},
    "plan_and_execute": {"cold_prefill": (2500, 3500, None),
                          "resume_prefill": (125, 421, 251),
                          "decode": (41, 125, 58)},
}

def agg(v):
    return {"min": min(v), "max": max(v), "avg": round(sum(v)/len(v),1), "n": len(v)}

def main(path):
    lines = [json.loads(l) for l in open(path)]
    stats = {}
    for paradigm in TABLE1:
        rows = [l for l in lines if l["paradigm"] == paradigm]
        cold = [s["input_tokens"] for r in rows for s in [r["plan"][0]]]
        res  = [s["input_tokens"] for r in rows for s in r["plan"] if s["phase"]=="resume_prefill"]
        dec  = [s["decode_tokens"] for r in rows for s in r["plan"]]
        stats[paradigm] = {"cold_prefill": agg(cold), "resume_prefill": agg(res), "decode": agg(dec)}

    print(f"{'paradigm':18} {'phase':16} {'min':>7} {'max':>7} {'avg':>7}   target(min,max,avg)")
    for paradigm, phases in stats.items():
        for phase, v in phases.items():
            t = TABLE1[paradigm][phase]
            flag = "OK" if (t[0] <= v["min"] and v["max"] <= t[1]) else "WARN"
            print(f"{paradigm:18} {phase:16} {v['min']:>7} {v['max']:>7} {v['avg']:>7}   {t[0]}..{t[1]} avg~{t[2]}   [{flag}]")
    return stats

if __name__ == "__main__":
    main(sys.argv[1])
