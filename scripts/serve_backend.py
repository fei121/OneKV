#!/usr/bin/env python3
"""Unified backend benchmark driver — same event-log/metrics for llama.cpp / OneKV / vLLM / SGLang.

Usage:
  python serve_backend.py --backend vllm|sglang|llama|onekv --agents 3 --sessions 12 --tag run
Note: llama.cpp and onekv backends start their own llama-server subprocess (generation backend).
"""
import sys, json, time, yaml, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from onekv.events import EventLogger
from onekv.backends import VllmBackend, SglangBackend, LlamaCppBackend, OneKVBackend

def start_llama_server(model_cfg, server_cfg, n_parallel):
    cmd=[server_cfg["binary"],"-m",model_cfg["model_path"],"-c",str(model_cfg["context_length"]),
         "-ngl",str(model_cfg["gpu_layers"]),"--parallel",str(max(int(n_parallel),1)),
         "--batch-size",str(model_cfg["batch_size"]),"--ubatch-size",str(model_cfg["ubatch_size"]),
         "--no-webui","--host",server_cfg["host"],"--port",str(server_cfg["port"])]
    print("[server] "+" ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wait_ready(host, port, timeout=120):
    import urllib.request
    end=time.time()+timeout
    while time.time()<end:
        try:
            if urllib.request.urlopen(f"http://{host}:{port}/health",timeout=2).status==200: return True
        except Exception: time.sleep(1)
    return False

def run_session(logger, backend, session, tool_wait_s):
    sid=session["session_id"]; plan=session["plan"]
    log=lambda **k: logger.log(k.pop("event"), **k)
    log(event="SESSION_START", session_id=sid)
    # Self-contained multi-phase prompt (includes the model's own previous output) so prefix caching
    # (vLLM paged prefix / SGLang RadixAttention) reuses a coherent prefix rather than a slot-KV
    # history that diverges.  Mirrors the serve_llama.py fix for the same multi-phase bug.
    full_prompt=None; prev_output=""
    for i, ph in enumerate(plan):
        p=ph["phase"]; n_pred=ph["decode_tokens"]
        prompt = ph["prompt"] if i==0 else (full_prompt + prev_output + ph["prompt"][len(plan[i-1]["prompt"]):])
        rid=f"{sid}:p{i}"
        log(event="REQUEST_ARRIVE", session_id=sid, request_id=rid, phase=p)
        log(event=("COLD_PREFILL_START" if p=="cold_prefill" else "RESUME_PREFILL_START"),
            session_id=sid, request_id=rid, phase=p, input_tokens=ph["input_tokens"])
        first=True; idx=0; out_parts=[]
        for _tok in backend.submit({"prompt":prompt,"n_predict":n_pred,"slot_id":i,
                                    "phase":p,"input_tokens":ph["input_tokens"]}):
            if first:
                log(event=("COLD_PREFILL_END" if p=="cold_prefill" else "RESUME_PREFILL_END"),
                    session_id=sid, request_id=rid, phase=p, input_tokens=ph["input_tokens"])
                log(event="DECODE_STEP_START", session_id=sid, request_id=rid, phase=p)
                first=False
            idx+=1; out_parts.append(_tok)
            log(event="TOKEN_EMIT", session_id=sid, request_id=rid, phase=p, output_tokens=idx)
        log(event="DECODE_STEP_END", session_id=sid, request_id=rid, phase=p, output_tokens=idx)
        prev_output="".join(out_parts); full_prompt=prompt
        if i < len(plan)-1 and tool_wait_s>0:
            log(event="TOOL_WAIT_START", session_id=sid, phase="tool_wait")
            time.sleep(tool_wait_s)
            log(event="TOOL_WAIT_END", session_id=sid, phase="tool_wait")
    log(event="SESSION_END", session_id=sid)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["vllm","sglang","llama","onekv"])
    ap.add_argument("--config", default="configs/serving_react_u.yaml")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model-path", default="/root/models/Qwen2.5-3B",
                    help="HF model dir for vLLM/SGLang (e.g. /root/models/Qwen2.5-7B)")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    trc=cfg["trace"]; sessions=[json.loads(l) for l in open(trc["file"])][:args.sessions]
    N=args.agents; tool_wait_s=cfg["workload"]["tool_wait_ms"]/1000.0
    out_prefix=args.tag or f"{args.backend}_N{N}"
    ev_path=Path(cfg["output"]["event_log"]).with_name(f"{out_prefix}_events.jsonl")
    res_path=Path(cfg["output"]["results"]).with_name(f"{out_prefix}_metrics.json")

    server_proc=None; be=None
    if args.backend=="vllm":
        be=VllmBackend(model_path=args.model_path, port=args.port or 8000)
        be.start()
    elif args.backend=="sglang":
        be=SglangBackend(model_path=args.model_path, port=args.port or 30000)
        be.start()
    else:  # llama / onekv: start llama-server, wrap with backend
        model_cfg=yaml.safe_load(open(cfg["model_config"]))
        server_proc=start_llama_server(model_cfg, cfg["server"], N)
        if not wait_ready(cfg["server"]["host"], cfg["server"]["port"]):
            print("[error] llama-server not ready; abort"); sys.exit(2)
        be = OneKVBackend(cfg) if args.backend=="onekv" else LlamaCppBackend(cfg)
        be.start()

    logger=EventLogger(ev_path)
    print(f"[{args.backend}] sessions={len(sessions)} agents={N}", flush=True)
    def worker(slot):
        for i in range(slot, len(sessions), N):
            run_session(logger, be, sessions[i], tool_wait_s)
    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(worker, range(N)))
    logger.close(); be.stop()
    if server_proc:
        server_proc.terminate()
        try: server_proc.wait(timeout=10)
        except Exception: server_proc.kill()
    print(f"[{args.backend}] event log -> {ev_path}", flush=True)
    import onekv.metrics as M
    res=M.compute(M.load_events(ev_path), None)
    res_path.parent.mkdir(parents=True,exist_ok=True)
    json.dump(res, open(res_path,"w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[{args.backend}] metrics -> {res_path}", flush=True)
    if args.backend=="onekv":
        print(f"[onekv] green context SM partitions:\n" + "\n".join(be.green_sms or ["(none)"]), flush=True)

if __name__=="__main__": main()
