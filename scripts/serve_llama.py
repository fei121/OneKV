#!/usr/bin/env python3
"""Task 3/4 — llama.cpp baseline serving harness.

Starts llama-server, drives N concurrent agent sessions from a generated trace,
records every token arrival (TTFT / TPOT / throughput / SLO) into the unified
event log (scripts/events.py), then computes metrics (scripts/metrics.py).

Each session gets a pinned server slot (slot_id = worker index) so its KV cache
survives across that session's prefill -> decode steps (matches a live agent).
Concurrency N = number of worker threads / slots.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json, sys, time, yaml, argparse, threading, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
from agentserve_repro.events import EventLogger, SessionClock


def wait_server_ready(host, port, timeout=120):
    url = f"http://{host}:{port}/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def start_server(model_cfg, server_cfg, n_parallel, port):
    cmd = [server_cfg["binary"], "-m", model_cfg["model_path"],
           "-c", str(model_cfg["context_length"]),
           "-ngl", str(model_cfg["gpu_layers"]),
           "--parallel", str(max(int(n_parallel), 1)),
           "--batch-size", str(model_cfg["batch_size"]),
           "--ubatch-size", str(model_cfg["ubatch_size"]),
           "--no-webui",
           "--host", server_cfg["host"], "--port", str(port)]
    print("[server] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def stream_completion(sess, host, port, prompt, n_predict, slot_id, cache_prompt=True):
    """Yields per-token strings; timestamps are recorded by the caller."""
    url = f"http://{host}:{port}/completion"
    body = {"prompt": prompt, "n_predict": int(n_predict), "temperature": 0.0,
            "stream": True, "cache_prompt": cache_prompt, "slot_id": int(slot_id)}
    try:
        with sess.post(url, json=body, stream=True, timeout=600) as resp:
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                line = line.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("stop"):
                    break
                tok = d.get("content")
                if tok is None:
                    tok = d.get("token")
                if tok is not None:
                    yield tok
    except Exception as e:
        print(f"[error] stream_completion slot {slot_id}: {e}", flush=True)


def run_session(logger, sess_http, host, port, session, slot_id, tool_wait_s,
                prefill_events):
    log = logger.log
    sid = session["session_id"]
    plan = session["plan"]
    clock = SessionClock(logger, sid)
    log("SESSION_START", session_id=sid)
    # Build a SELF-CONTAINED multi-phase prompt that includes the model's own previous output,
    # so cache_prompt reuses a coherent prefix (the slot-KV-only approach degenerates because the
    # generated history diverges from the next prompt).  (Fixes the cache_prompt multi-phase bug.)
    full_prompt = None
    prev_output = ""
    for i, phase in enumerate(plan):
        p = phase["phase"]
        if i == 0:
            prompt = phase["prompt"]                      # cold prompt
        else:
            delta = phase["prompt"][len(plan[i-1]["prompt"]):]   # this round's <tool_result> block
            prompt = full_prompt + prev_output + delta          # full conversation + prev model output
        n_pred = phase["decode_tokens"]
        rid = clock.next_request_id(p)
        log("REQUEST_ARRIVE", session_id=sid, request_id=rid, phase=p)
        label = "COLD_PREFILL_START" if p == "cold_prefill" else "RESUME_PREFILL_START"
        log(label, session_id=sid, request_id=rid, phase=p,
            input_tokens=phase["input_tokens"])
        end_label = "COLD_PREFILL_END" if p == "cold_prefill" else "RESUME_PREFILL_END"
        first = True
        idx = 0
        out_parts = []
        for _tok in stream_completion(sess_http, host, port, prompt, n_pred, slot_id):
            if first:
                log(end_label, session_id=sid, request_id=rid, phase=p,
                    input_tokens=phase["input_tokens"])
                log("DECODE_STEP_START", session_id=sid, request_id=rid, phase=p)
                first = False
            idx += 1
            out_parts.append(_tok)
            log("TOKEN_EMIT", session_id=sid, request_id=rid, phase=p, output_tokens=idx)
        log("DECODE_STEP_END", session_id=sid, request_id=rid, phase=p, output_tokens=idx)
        prev_output = "".join(out_parts)
        full_prompt = prompt
        if i < len(plan) - 1 and tool_wait_s > 0:
            log("TOOL_WAIT_START", session_id=sid, phase="tool_wait")
            time.sleep(tool_wait_s)
            log("TOOL_WAIT_END", session_id=sid, phase="tool_wait")
    log("SESSION_END", session_id=sid)
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/autodl-tmp/exp/configs/serving_react_u.yaml")
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--agents", type=int, default=None)
    ap.add_argument("--paradigm", default=None)
    ap.add_argument("--tool-wait-ms", type=int, default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--event-log", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--skip-serve", action="store_true", help="reuse already-running server")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg = yaml.safe_load(open(cfg["model_config"]))
    server_cfg = cfg["server"]
    trc = cfg["trace"]
    wl = cfg["workload"]
    if args.agents: wl["concurrent_agents"] = args.agents
    if args.tool_wait_ms is not None: wl["tool_wait_ms"] = args.tool_wait_ms
    if args.paradigm and args.paradigm != "all": trc["paradigm"] = args.paradigm
    if args.port: server_cfg["port"] = args.port
    if args.event_log: cfg["output"]["event_log"] = args.event_log
    if args.results: cfg["output"]["results"] = args.results

    sessions = [json.loads(l) for l in open(trc["file"])]
    if trc["paradigm"] != "all":
        sessions = [s for s in sessions if s["paradigm"] == trc["paradigm"]]
    sessions = sessions[: (args.sessions or trc["max_sessions"])]
    N = int(wl["concurrent_agents"])
    tool_wait_s = wl["tool_wait_ms"] / 1000.0

    # event log
    ev_path = Path(cfg["output"]["event_log"])
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(ev_path)

    proc = None
    if not args.skip_serve:
        proc = start_server(model_cfg, server_cfg, N, server_cfg["port"])
        if not wait_server_ready(server_cfg["host"], server_cfg["port"]):
            print("[error] server not ready; aborting", flush=True)
            if proc: proc.kill()
            sys.exit(2)

    host, port = server_cfg["host"], server_cfg["port"]
    Nsess = len(sessions)
    print(f"[harness] sessions={Nsess} agents={N} tool_wait={wl['tool_wait_ms']}ms", flush=True)

    def worker(slot):
        # Round-robin session assignment so each slot runs a distinct session stream.
        for idx in range(slot, Nsess, N):
            run_session(logger, requests.Session(), host, port,
                        sessions[idx], slot, tool_wait_s, None)
        return f"slot{slot}"

    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(worker, range(N)))

    logger.close()
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print(f"[harness] event log -> {ev_path}", flush=True)
    # run metrics
    met = cfg["output"]["results"]
    print(f"[metrics] computing from {ev_path}", flush=True)
    import agentserve_repro.metrics as M
    mcfg = yaml.safe_load(open(cfg["metrics_config"]))
    slo_cfg = None
    if mcfg["slo"]["tau_ttft_ms"] is not None and mcfg["slo"]["tau_tpot_ms"] is not None:
        slo_cfg = {"tau_ttft_ms": mcfg["slo"]["tau_ttft_ms"],
                   "tau_tpot_ms": mcfg["slo"]["tau_tpot_ms"]}
    res = M.compute(M.load_events(ev_path), slo_cfg)
    Path(met).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(met, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[metrics] -> {met}", flush=True)


if __name__ == "__main__":
    main()
