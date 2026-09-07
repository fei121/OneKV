#!/usr/bin/env python3
"""Dual green-context + dual-worker AgentServe serving path.

Two llama.cpp servers, each bound to a CUDA Green Context with a different SM share:
  - prefill server  (AGENTSERVE_GREEN_PCT=PREFILL_PCT) -> handles QP (cold + long resume prefill)
  - decode server   (AGENTSERVE_GREEN_PCT=DECODE_PCT)  -> handles QD (short resume + decode)
Requests are routed by phase (PhaseRouter). Events/metrics use the unified logger.

Honest note: this realizes the dual-worker + dual-green-context *concurrent* serving path via two
instances (each an independent context with its own KV). The paper's exact single-context shared-KV
dual-thread design is the deeper integration; here we measure concurrent SM-partitioned serving.
"""
import sys, json, time, yaml, argparse, subprocess, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agentserve_repro.events import EventLogger
from agentserve_repro.scheduler import PhaseRouter
from agentserve_repro.backends import LlamaCppBackend

def start_llama(model_path, pct, port, ctxlen=24576, npar=4):
    env=dict(__import__('os').environ); env["AGENTSERVE_GREEN_PCT"]=str(pct)
    cmd=["/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp/build/bin/llama-server",
         "-m",model_path,"-c",str(ctxlen),"-ngl","99","--parallel",str(npar),
         "--no-webui","--host","127.0.0.1","--port",str(port)]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/dual_{port}.log","w"))

def wait_ready(port, timeout=120):
    import urllib.request
    end=time.time()+timeout
    while time.time()<end:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2).status==200: return True
        except Exception: time.sleep(1)
    return False

def endpoint(prompt,n_pred):
    return {"prompt":prompt,"n_predict":n_pred,"slot_id":-1}

def run_session(logger, router, pre, dec, session, tool_wait_s):
    sid=session["session_id"]; plan=session["plan"]
    log=lambda **k: logger.log(k.pop("event"), **k)
    log(event="SESSION_START", session_id=sid)
    for i,ph in enumerate(plan):
        p=ph["phase"]; inp=ph["input_tokens"]; prompt=ph["prompt"]; n_pred=ph["decode_tokens"]
        # route to prefill or decode worker (QD/QP)
        from agentserve_repro.phase import RequestPhase, AgentRequestMeta
        ph_enum = RequestPhase.COLD_PREFILL if p=="cold_prefill" else RequestPhase.RESUME_PREFILL
        meta=AgentRequestMeta(1,1,None,0,0,inp,0, kv_prefix_available=(p!="cold_prefill"), arrival_ts_ns=0)
        q=router.route(meta)
        backend = pre if q=="QP" else dec
        prefill_backend = (q=="QP")
        log(event="REQUEST_ARRIVE", session_id=sid, request_id=f"{sid}:{i}", phase=p, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
        log(event=("COLD_PREFILL_START" if p=="cold_prefill" else "RESUME_PREFILL_START"),
            session_id=sid, request_id=f"{sid}:{i}", phase=p, input_tokens=inp, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
        first=True; idx=0
        for _tok in backend.submit(endpoint(prompt,n_pred)):
            if first:
                log(event=("COLD_PREFILL_END" if p=="cold_prefill" else "RESUME_PREFILL_END"),
                    session_id=sid, request_id=f"{sid}:{i}", phase=p, input_tokens=inp, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
                log(event="DECODE_STEP_START", session_id=sid, request_id=f"{sid}:{i}", phase=p, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
                first=False
            idx+=1
            log(event="TOKEN_EMIT", session_id=sid, request_id=f"{sid}:{i}", phase=p, output_tokens=idx, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
        log(event="DECODE_STEP_END", session_id=sid, request_id=f"{sid}:{i}", phase=p, output_tokens=idx, queue_prefill=(1 if q=="QP" else 0), queue_decode=(1 if q=="QD" else 0))
        if i<len(plan)-1 and tool_wait_s>0:
            log(event="TOOL_WAIT_START", session_id=sid, phase="tool_wait"); time.sleep(tool_wait_s)
            log(event="TOOL_WAIT_END", session_id=sid, phase="tool_wait")
    log(event="SESSION_END", session_id=sid)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/autodl-tmp/exp/configs/serving.yaml")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--prefill-pct", type=int, default=60)
    ap.add_argument("--decode-pct", type=int, default=40)
    ap.add_argument("--tag", default="dualgreen")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    sessions=[json.loads(l) for l in open(cfg["trace"]["file"])][:args.sessions]
    N=args.agents; tw=cfg["workload"]["tool_wait_ms"]/1000.0
    ev_path=Path(cfg["output"]["event_log"]).with_name(f"{args.tag}_events.jsonl")
    res_path=Path(cfg["output"]["results"]).with_name(f"{args.tag}_metrics.json")
    model_path="/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf"
    p_pid=start_llama(model_path,args.prefill_pct,8080,24576,N)
    d_pid=start_llama(model_path,args.decode_pct,8081,24576,N)
    if not (wait_ready(8080) and wait_ready(8081)): print("server(s) not ready"); sys.exit(2)
    # backend objects (thin clients to each endpoint)
    cfg_llama={"server":{"host":"127.0.0.1","port":8080}}
    cfg_decode={"server":{"host":"127.0.0.1","port":8081}}
    pre=LlamaCppBackend(cfg_llama); dec=LlamaCppBackend(cfg_decode)
    router=PhaseRouter(64)
    logger=EventLogger(ev_path)
    print(f"[dualgreen] prefill@{args.prefill_pct}% decode@{args.decode_pct}%  agents={N} sessions={len(sessions)}", flush=True)
    def worker(slot):
        for i in range(slot,len(sessions),N):
            run_session(logger,router,pre,dec,sessions[i],tw)
    with ThreadPoolExecutor(max_workers=N) as ex: list(ex.map(worker,range(N)))
    logger.close(); p_pid.terminate(); d_pid.terminate()
    import agentserve_repro.metrics as M
    res=M.compute(M.load_events(ev_path),None)
    res_path.parent.mkdir(parents=True,exist_ok=True); json.dump(res,open(res_path,"w"),indent=2)
    print(json.dumps(res,indent=2))
    print(f"[dualgreen] -> {res_path}")

if __name__=="__main__": main()
