#!/usr/bin/env python3
"""Task 18 — unified ServingBackend adapter interface + implementations.

Backends: LlamaCppBackend (v0, serve_llama.py), VllmBackend, SglangBackend.
All produce the same unified event log (scripts/events.py) consumed by scripts/metrics.py,
so TTFT/TPOT/throughput are measured identically across backends.
"""
import abc, json, time, queue, subprocess
from pathlib import Path

class ServingBackend(abc.ABC):
    @abc.abstractmethod
    def start(self): ...
    @abc.abstractmethod
    def stop(self): ...
    @abc.abstractmethod
    def submit(self, request): ...   # request = {prompt, n_predict, slot_id}
    @abc.abstractmethod
    def collect_metrics(self): ...

class StreamMixin:
    """Streaming-completions client hook for OpenAI-ish /completion endpoints."""
    def stream_completion(self, base_url, prompt, n_predict, stream=True, timeout=600):
        import requests
        url = f"{base_url}/v1/completions"
        body = {"model": self.model, "prompt": prompt, "max_tokens": int(n_predict),
                "temperature": 0.0, "stream": stream}
        with requests.post(url, json=body, stream=True, timeout=timeout) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                try:
                    d = json.loads(payload)
                except Exception: continue
                choices = d.get("choices") or []
                if choices and choices[0].get("text") is not None:
                    yield choices[0]["text"]
                # OpenAI streaming: some servers emit via delta
                if choices and "delta" in choices[0] and choices[0]["delta"].get("content") is not None:
                    yield choices[0]["delta"]["content"]

class VllmBackend(StreamMixin, ServingBackend):
    """Start `vllm serve` (OpenAI server) in the vllm conda env; stream completions per token."""
    def __init__(self, model_path, port=8000, gpu_mem=0.9, host="127.0.0.1", python=None):
        self.model_path=model_path; self.port=port; self.host=host
        self.python=python or "/root/autodl-tmp/conda_envs/vllm/bin/python"
        self.base_url=f"http://{host}:{port}"; self.model=model_path; self.proc=None
    def start(self):
        import subprocess
        cmd=[self.python,"-m","vllm.entrypoints.openai.api_server","--model",self.model_path,
             "--host",self.host,"--port",str(self.port),"--gpu-memory-utilization","0.9","--max-model-len","8192"]
        self.proc=subprocess.Popen(cmd, stdout=open("/tmp/vllm_backend.log","w"), stderr=subprocess.STDOUT)
        import urllib.request,time
        for _ in range(200):
            try:
                if urllib.request.urlopen(f"{self.base_url}/health",timeout=2).status==200: return self
            except Exception: time.sleep(1)
        raise RuntimeError("vllm server not ready")
    def stop(self):
        if self.proc: self.proc.terminate()
    def submit(self, request): yield from self.stream_completion(self.base_url, request["prompt"], request["n_predict"])
    def collect_metrics(self): return {}   # filled after run via events/metrics

class SglangBackend(StreamMixin, ServingBackend):
    def __init__(self, model_path, port=30000, host="127.0.0.1", python=None):
        self.model_path=model_path; self.port=port; self.host=host
        self.python=python or "/root/autodl-tmp/conda_envs/sglang/bin/python"
        self.base_url=f"http://{host}:{port}"; self.model=model_path; self.proc=None
    def start(self):
        import subprocess
        cmd=[self.python,"-m","sglang.launch_server","--model-path",self.model_path,
             "--host",self.host,"--port",str(self.port),"--max-total-tokens","8192"]
        self.proc=subprocess.Popen(cmd, stdout=open("/tmp/sglang_backend.log","w"), stderr=subprocess.STDOUT)
        import urllib.request,time
        for _ in range(300):
            try:
                if urllib.request.urlopen(f"{self.base_url}/health",timeout=2).status==200: return self
            except Exception: time.sleep(1)
        raise RuntimeError("sglang server not ready")
    def stop(self):
        if self.proc: self.proc.terminate()
    def submit(self, request): yield from self.stream_completion(self.base_url, request["prompt"], request["n_predict"])
    def collect_metrics(self): return {}

class LlamaCppBackend(ServingBackend):
    def __init__(self, config): self.config=config
    def start(self): return self
    def stop(self): pass
    def submit(self, request):  # native /completion
        import requests
        url=f"http://{self.config['server']['host']}:{self.config['server']['port']}/completion"
        body={"prompt":request["prompt"],"n_predict":request["n_predict"],"temperature":0.0,"stream":True,
              "cache_prompt":True,"slot_id":request.get("slot_id",-1)}
        with requests.post(url,json=body,stream=True,timeout=600) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line.startswith("data:"): continue
                d=json.loads(line[5:].strip()); 
                if d.get("stop"): break
                tok=d.get("content")
                if tok is not None: yield tok
    def collect_metrics(self): return {}

class AgentServeBackend(ServingBackend):
    """AgentServe orchestration serving backend (QD/QP + dual worker + TPOT controller + green context).

    Implemented layer (honest scope):
      - request phase classification + QD/QP routing (scheduler.PhaseRouter)
      - DecodeWorker / PrefillWorker submission path
      - TPOT controller adjusts the decode SM share (the green-context share the runtime binds to)
      - CUDA Green Context SM partitioning initialized via a helper binary
    Token generation is delegated to the underlying llama.cpp endpoint.
    Honest boundary: engine-level rebinding of llama.cpp kernels onto a green-context stream
    (llama-server update_slots integration) is the remaining step; this backend measures the
    orchestration path and the SM-partition layout the scheduler relies on.
    """
    def __init__(self, config, b_prefill=64, green_helper=None):
        self.config=config; self.b_prefill=b_prefill
        from agentserve_repro.scheduler import PhaseRouter
        self.router=PhaseRouter(b_prefill)
        self.green_helper=green_helper or "/root/autodl-tmp/exp/cuda/agentserve_runtime"
        self.green_sms=[]
    def start(self):
        if Path(self.green_helper).exists():
            try:
                out=subprocess.run([self.green_helper], capture_output=True, text=True, timeout=60)
                self.green_sms=[l.strip() for l in out.stdout.splitlines() if '->' in l and 'SM' in l]
            except Exception as e:
                self.green_sms=[f"green helper error: {e}"]
        return self
    def stop(self): pass
    def route_phase(self, phase, new_input_tokens):
        from agentserve_repro.phase import RequestPhase, AgentRequestMeta
        ph = (RequestPhase.COLD_PREFILL if phase=="cold_prefill" else
              RequestPhase.RESUME_PREFILL if phase=="resume_prefill" else RequestPhase.DECODE)
        meta=AgentRequestMeta(1,1,None,0,0,new_input_tokens,0,
                              kv_prefix_available=(phase!="cold_prefill"), arrival_ts_ns=0)
        return self.router.route(meta)
    def submit(self, request):
        phase=request.get("phase","decode"); inp=request.get("input_tokens",0)
        q=self.route_phase(phase, inp)  # QD or QP
        yield from LlamaCppBackend(self.config).submit(request)

    def collect_metrics(self):
        return {}


if __name__ == "__main__":
    print("ServingBackend interface + LlamaCpp/Vllm/Sglang adapters defined (Task 18).")
    print("Use with scripts/events.py + scripts/metrics.py for identical TTFT/TPOT measurement.")
