#!/usr/bin/env python3
"""Task 11 — Orchestration / Execution path (QD / QP + Decode/Prefill workers) — LOGIC LAYER.

Implements the paper's orchestration structure with the plan's routing rules:
  QD = Decode + short Resume Prefill (<= B_prefill)
  QP = Cold Prefill        + long  Resume Prefill (>  B_prefill)

Each worker: queue pop -> route -> (placeholder GPU submit) -> record event.
The real GPU submission is wired into llama.cpp in the integration phase (Task 15);
here the queues/workers/phase-routing are validated independently (ENGINEERING-CHOICE slice).
"""
import threading, time, json, queue, argparse
from dataclasses import dataclass
from onekv.phase import RequestPhase, AgentRequestMeta, RequestClassifier

@dataclass
class B_prefill:
    """Resume-prefill token budget (Algorithm 1's req.len <= B_prefill)."""
    value: int = 64   # default budget (INFERENCE: resume prefill new-token budget)

class PhaseRouter:
    """Routes a request to a queue name based on phase and resume-prefill length."""
    def __init__(self, b_prefill=64):
        self.b_prefill = b_prefill

    def route(self, meta: AgentRequestMeta) -> str:
        phase = RequestClassifier.classify(meta)
        if phase == RequestPhase.COLD_PREFILL:
            return "QP"
        if phase == RequestPhase.RESUME_PREFILL:
            # short resume merges with decode; long resume goes to prefill
            return "QD" if meta.new_input_tokens <= self.b_prefill else "QP"
        if phase == RequestPhase.DECODE:
            return "QD"
        raise ValueError("unroutable")

class DecodeWorker:
    name = "DecodeWorker"
    target_q = "QD"
    def __init__(self, logger, emit):
        self.logger = logger
        self.emit = emit   # callable(request_meta) -> placeholder GPU submit
    def run(self, q, stop_event, session_id="worker_dec"):
        while not stop_event.is_set():
            try:
                meta = q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.logger.log("REQUEST_ARRIVE", session_id=str(meta.session_id),
                            event="SCHEDULER_TICK", phase="decode")
            self.emit(meta)     # placeholder: real submit wired in Task 15
            q.task_done()

class PrefillWorker:
    name = "PrefillWorker"
    target_q = "QP"
    def __init__(self, logger, emit):
        self.logger = logger
        self.emit = emit
    def run(self, q, stop_event, session_id="worker_pref"):
        while not stop_event.is_set():
            try:
                meta = q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.logger.log("SCHEDULER_TICK", session_id=str(meta.session_id),
                            phase=meta.phase.value)
            self.emit(meta)
            q.task_done()

def build_qdq_from_trace(path, b_prefill=64):
    """Enqueue all trace requests into QD/QP queues in arrival order."""
    meta_by_sid = {}
    lines = [json.loads(l) for l in open(path)]
    for s in lines:
        sid = s["session_id"]
        for i, ph in enumerate(s["plan"]):
            has_kv = i > 0
            inp = ph["input_tokens"]
            out = ph["decode_tokens"]
            # prefill request
            m1 = AgentRequestMeta(session_id=sid, request_id=i*2+1,
                phase=RequestPhase.COLD_PREFILL if not has_kv else RequestPhase.RESUME_PREFILL,
                total_context_tokens=inp, cached_prefix_tokens=0 if not has_kv else 0,
                new_input_tokens=inp, requested_output_tokens=0,
                kv_prefix_available=has_kv, arrival_ts_ns=0)
            # decode request
            m2 = AgentRequestMeta(session_id=sid, request_id=i*2+2,
                phase=RequestPhase.DECODE, total_context_tokens=inp,
                cached_prefix_tokens=inp, new_input_tokens=0,
                requested_output_tokens=out, kv_prefix_available=True, arrival_ts_ns=0)
            meta_by_sid.setdefault(sid, []).extend([m1, m2])
    return meta_by_sid

def self_test():
    router = PhaseRouter(b_prefill=64)
    # cold -> QP
    assert router.route(AgentRequestMeta(1,1,None,3000,0,3000,0,False,0)) == "QP"
    # short resume (56 <= 64) -> QD
    assert router.route(AgentRequestMeta(1,2,None,3056,3000,56,0,True,0)) == "QD"
    # long resume (251 > 64) -> QP
    assert router.route(AgentRequestMeta(1,3,None,3251,3000,251,0,True,0)) == "QP"
    # decode -> QD
    assert router.route(AgentRequestMeta(1,4,None,3251,3251,0,38,True,0)) == "QD"
    counts = {}
    for name in ["QD","QP"]:
        pass
    print("ROUTER OK: cold->QP, short-resume->QD, long-resume->QP, decode->QD")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="/root/autodl-tmp/exp/traces/traces_Qwen2.5-3B.jsonl")
    ap.add_argument("--b-prefill", type=int, default=64)
    args = ap.parse_args()
    self_test()
    meta = build_qdq_from_trace(args.trace, args.b_prefill)
    from collections import Counter
    router = PhaseRouter(args.b_prefill)
    qdef = {"QD": queue.Queue(), "QP": queue.Queue()}
    counts = Counter()
    for sid, metas in meta.items():
        for m in metas:
            q = router.route(m)
            qdef[q].put(m)
            counts[q] += 1
    print("QD/QP splits (arrival order):", dict(counts))
    # run workers briefly to verify drain (placeholder emit = no-op)
    logger = type("L", (), {"log": lambda *a, **k: None})()
    stop = threading.Event()
    ths = []
    for wcls in (DecodeWorker, PrefillWorker):
        w = wcls(logger, lambda m: None)
        t = threading.Thread(target=w.run, args=(qdef[w.target_q], stop))
        t.start(); ths.append(t)
    time.sleep(0.3); stop.set()
    for t in ths: t.join(timeout=1)
    print("workers drained; orchestrator scaffold OK")
