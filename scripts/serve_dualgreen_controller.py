#!/usr/bin/env python3
"""TPOTController integrated into the dual-green-context serving path (adaptive control loop).

Each round: run a small batch at a decode SM share, measure scheduler_tpot_step_ms,
let the TPOTController adjust the decode share (restart decode server to apply it),
repeat until it converges. Shows decode SLO being respected while releasing SMs to prefill.
"""
import sys, json, time, subprocess, signal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agentserve_repro.events import EventLogger
from agentserve_repro.scheduler import PhaseRouter
from agentserve_repro.backends import LlamaCppBackend
# TPOTController is in cuda/agentserve_core.cu; provide a python port (Task 13 logic)
class TPOTController:
    def __init__(self, target_tpot_ms, start=40):
        self.target=target_tpot_ms; self.r_share=start; self.history=[]
    def update(self, tpot_step_ms):
        before=self.r_share
        if tpot_step_ms > self.target*1.2 and self.r_share < 100: self.r_share+=10
        elif tpot_step_ms < self.target*0.8 and self.r_share > 20: self.r_share-=10
        self.history.append((before,self.r_share,round(tpot_step_ms,3)))
        return self.r_share

B="/root/autodl-tmp/agentserve-reproduction/third_party/llama.cpp/build/bin/llama-server"
def kill_all():
    subprocess.run("pkill -f 'llama-server' 2>/dev/null", shell=True); time.sleep(3)
def start(pct,port):
    import os
    env=dict(os.environ); env["AGENTSERVE_GREEN_PCT"]=str(pct)
    cmd=[B,"-m","/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf","-c","24576","-ngl","99","--parallel","2","--no-webui","--host","127.0.0.1","--port",str(port)]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def ready(port,t=120):
    import urllib.request
    end=time.time()+t
    while time.time()<end:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2).status==200: return True
        except Exception: time.sleep(1)
    return False

def run_round(decode_pct, sess_file, nsess=4, N=2, tool_wait=0.2):
    kill_all()
    pp=start(100-decode_pct, 8080); dp=start(decode_pct, 8081)
    if not (ready(8080) and ready(8081)): raise RuntimeError("servers not ready")
    sessions=[json.loads(l) for l in open(sess_file)][:nsess]
    pre=LlamaCppBackend({"server":{"host":"127.0.0.1","port":8080}})
    dec=LlamaCppBackend({"server":{"host":"127.0.0.1","port":8081}})
    router=PhaseRouter(64)
    ev="/root/autodl-tmp/exp/raw_logs/_ctl_events.jsonl"; logger=EventLogger(ev)
    from concurrent.futures import ThreadPoolExecutor
    def run_sess(sid,s):
        log=lambda **k: logger.log(k.pop("event"),**k)
        log(event="SESSION_START",session_id=sid)
        for i,ph in enumerate(s["plan"]):
            p=ph["phase"]; inp=ph["input_tokens"]; prompt=ph["prompt"]; n_pred=ph["decode_tokens"]
            from agentserve_repro.phase import RequestPhase, AgentRequestMeta
            ph_e=RequestPhase.COLD_PREFILL if p=="cold_prefill" else RequestPhase.RESUME_PREFILL
            q=router.route(AgentRequestMeta(1,1,None,0,0,inp,0,(p!="cold_prefill"),0))
            be=pre if q=="QP" else dec
            log(event="REQUEST_ARRIVE",session_id=sid,request_id=f"{sid}:{i}",phase=p,queue_prefill=int(q=="QP"),queue_decode=int(q=="QD"))
            log(event=("COLD_PREFILL_START" if p=="cold_prefill" else "RESUME_PREFILL_START"),session_id=sid,request_id=f"{sid}:{i}",phase=p,input_tokens=inp)
            first=True; idx=0
            for _t in be.submit({"prompt":prompt,"n_predict":n_pred,"slot_id":-1}):
                if first:
                    log(event=("COLD_PREFILL_END" if p=="cold_prefill" else "RESUME_PREFILL_END"),session_id=sid,request_id=f"{sid}:{i}",phase=p)
                    log(event="DECODE_STEP_START",session_id=sid,request_id=f"{sid}:{i}",phase=p); first=False
                idx+=1; log(event="TOKEN_EMIT",session_id=sid,request_id=f"{sid}:{i}",phase=p,output_tokens=idx)
            log(event="DECODE_STEP_END",session_id=sid,request_id=f"{sid}:{i}",phase=p,output_tokens=idx)
            if i<len(s["plan"])-1: log(event="TOOL_WAIT_START",session_id=sid,phase="tool_wait"); time.sleep(tool_wait); log(event="TOOL_WAIT_END",session_id=sid,phase="tool_wait")
        log(event="SESSION_END",session_id=sid)
    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(lambda i: run_sess(sessions[i]["session_id"],sessions[i]), range(min(len(sessions),N))))
    logger.close(); pp.terminate(); dp.terminate()
    import agentserve_repro.metrics as M
    return M.compute(M.load_events(ev), None)

def main():
    sess="/root/autodl-tmp/exp/traces/traces_Qwen2.5-3B.jsonl"
    # target: isolated decode tpot at high decode share (measure once at 90%)
    m=run_round(90, sess, nsess=3, N=1)
    target=m["tpot_step_ms"]["p50"] if m["tpot_step_ms"]["p50"] else 25.0
    ctrl=TPOTController(target, start=20)
    print(f"[controller] target_tpot_step={target:.2f}ms  initial_decode_share=20%")
    for rnd in range(5):
        d=ctrl.r_share
        m=run_round(d, sess, nsess=4, N=2)
        tpot=m["tpot_step_ms"]["p50"]
        new_d=ctrl.update(tpot)
        print(f"  round{rnd}: decode_share={d}%  tpot_step={tpot:.2f}ms  -> controller sets {new_d}%  | thr={m['throughput_tokens_per_s']:.1f} ttft_p50={m['ttft_cold_ms']['p50']:.1f}")
        if new_d==d: print("  converged"); break
    print("controller trajectory:", ctrl.history)

if __name__=="__main__": main()
