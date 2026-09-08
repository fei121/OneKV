#!/usr/bin/env python3
"""Task 4 — metric computation from the unified event log.

Computes (per the plan / paper):
  - TTFT (cold and resume): first TOKEN_EMIT - REQUEST_ARRIVE
  - benchmark TPOT: inter-token gaps within each decode step (p50/p95/mean)
  - scheduler TPOT_step: (DECODE_STEP_END - DECODE_STEP_START) / output_tokens
  - throughput: total output tokens / wall-clock (scope-configurable)
  - session-level SLO attainment (joint TTFT + TPOT)
"""
import json, sys, argparse, statistics, collections

def load_events(path):
    """Skip __meta__ header; return list of event dicts."""
    evs = []
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("__meta__"):
            continue
        evs.append(d)
    return evs

def group_requests(evs):
    """Group TOKEN_EMIT and prefill/decode markers per (session_id, request_id)."""
    reqs = collections.defaultdict(dict)
    for e in evs:
        key = (e.get("session_id"), e.get("request_id"))
        r = reqs[key]
        r.setdefault("phase", e.get("phase"))
        ev = e["event"]
        if ev == "REQUEST_ARRIVE":
            r["arrive"] = e["ts_ns"]
        elif ev in ("COLD_PREFILL_START", "RESUME_PREFILL_START"):
            r["prefill_start"] = e["ts_ns"]
        elif ev in ("COLD_PREFILL_END", "RESUME_PREFILL_END"):
            r["prefill_end"] = e["ts_ns"]
        elif ev == "DECODE_STEP_START":
            r["decode_start"] = e["ts_ns"]
        elif ev == "DECODE_STEP_END":
            r["decode_end"] = e["ts_ns"]
            r["output_tokens"] = e.get("output_tokens", 0)
        elif ev == "TOKEN_EMIT":
            r.setdefault("tokens", []).append(e["ts_ns"])
    return reqs

def pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]

def mm(vals):
    return {"p50": pctl(vals, 0.50), "p95": pctl(vals, 0.95),
            "mean": round(statistics.fmean(vals), 3) if vals else None,
            "n": len(vals)}

def compute(events, slo_cfg=None):
    reqs = group_requests(events)
    sessions = collections.defaultdict(dict)
    for (sid, rid), r in reqs.items():
        sessions[sid].setdefault("requests", {})[rid] = r

    # raw latencies
    ttft_cold, ttft_resume = [], []
    tpot_gaps = []            # benchmark TPOT (inter-token gaps)
    tpot_step = []            # scheduler TPOT_step per decode step
    session_ttft_cold, session_tpot_p95 = {}, {}
    all_token_ts = []         # for throughput (tokens + ts)
    n_output_tokens = 0
    tool_wait_ns = 0

    for sid, data in sessions.items():
        rlist = list(data["requests"].values())
        # TTFT
        for r in rlist:
            if "arrive" in r and r.get("tokens"):
                ttft = r["tokens"][0] - r["arrive"]
                if r.get("phase") == "cold_prefill":
                    ttft_cold.append(ttft)
                    session_ttft_cold[sid] = ttft
                else:
                    ttft_resume.append(ttft)
            # benchmark TPOT gaps within this request's decode step
            toks = sorted(r.get("tokens", []))
            if len(toks) >= 2:
                for i in range(1, len(toks)):
                    tpot_gaps.append(toks[i] - toks[i - 1])
            # scheduler TPOT_step
            if r.get("decode_start") and r.get("decode_end") and r.get("output_tokens", 0) > 0:
                tpot_step.append((r["decode_end"] - r["decode_start"]) / r["output_tokens"])
            n_output_tokens += r.get("output_tokens", len(toks))
            all_token_ts.extend(toks)
        # session-level TPOT (max p95 over its decode steps) for SLO
        sess_gaps = []
        for r in rlist:
            toks = sorted(r.get("tokens", []))
            for i in range(1, len(toks)):
                sess_gaps.append(toks[i] - toks[i - 1])
        session_tpot_p95[sid] = pctl(sess_gaps, 0.95)

    # tool wait total
    for e in events:
        if e["event"] == "TOOL_WAIT_START":
            # match to its END; simplest: assume pair ordering
            pass
    # compute tool wait by pairing START/END per session
    tw = collections.defaultdict(list)
    for e in events:
        if e["event"] == "TOOL_WAIT_START":
            tw[(e["session_id"])].append(e["ts_ns"])
        elif e["event"] == "TOOL_WAIT_END":
            if tw[e["session_id"]]:
                tool_wait_ns += e["ts_ns"] - tw[e["session_id"]].pop(0)

    ts_list = [e["ts_ns"] for e in events if e["event"] in ("SESSION_START", "SESSION_END")]
    session_start = min(ts_list) if ts_list else min(e["ts_ns"] for e in events)
    session_end = max(ts_list) if ts_list else max(e["ts_ns"] for e in events)
    wall_ns = session_end - session_start
    wall_no_tw_ns = max(1, wall_ns - tool_wait_ns)

    # SLO
    slo = None
    if slo_cfg and slo_cfg.get("tau_ttft_ms") and slo_cfg.get("tau_tpot_ms"):
        t_ttft = slo_cfg["tau_ttft_ms"] * 1_000_000
        t_tpot = slo_cfg["tau_tpot_ms"] * 1_000_000
        passed = 0
        for sid, data in sessions.items():
            ct = session_ttft_cold.get(sid)
            tps = session_tpot_p95.get(sid)
            if ct is not None and tps is not None and ct <= t_ttft and tps <= t_tpot:
                passed += 1
        slo = {"passed": passed, "total": len(sessions),
               "rate": round(passed / len(sessions), 4) if sessions else None}

    return {
        "n_sessions": len(sessions),
        "ttft_cold_ms": mm([t / 1e6 for t in ttft_cold]),
        "ttft_resume_ms": mm([t / 1e6 for t in ttft_resume]),
        "tpot_benchmark_ms": mm([t / 1e6 for t in tpot_gaps]),
        "tpot_step_ms": mm([t / 1e6 for t in tpot_step]),
        "throughput_tokens_per_s": round(n_output_tokens / (wall_ns / 1e9), 3),
        "throughput_excl_tool_wait_tokens_per_s": round(n_output_tokens / (wall_no_tw_ns / 1e9), 3),
        "wall_clock_s": round(wall_ns / 1e9, 3),
        "tool_wait_s": round(tool_wait_ns / 1e9, 3),
        "slo": slo,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_log")
    ap.add_argument("--slo", default=None, help="JSON string or path with {tau_ttft_ms, tau_tpot_ms}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    slo_cfg = None
    if args.slo:
        if args.slo.endswith(".json"):
            slo_cfg = json.load(open(args.slo))
        else:
            slo_cfg = json.loads(args.slo)
    evs = load_events(args.event_log)
    res = compute(evs, slo_cfg)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)

if __name__ == "__main__":
    main()
