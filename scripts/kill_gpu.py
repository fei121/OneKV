#!/usr/bin/env python3
"""Kill leftover serving backends / GPU compute apps between benchmark runs."""
import os, signal, subprocess

def kill_pat(pat):
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    n = 0
    for line in out.splitlines():
        a = line.strip()
        if pat in a:
            try:
                os.kill(int(a.split()[0]), signal.SIGKILL); n += 1
            except Exception:
                pass
    return n

total = 0
for pat in ["sglang.launch_server", "vllm.entrypoints.openai", "llama-server",
            "serve_backend.py", "serve_llama.py"]:
    total += kill_pat(pat)
# kill GPU compute apps (vLLM EngineCore etc.)
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout
for pid in out.split():
    try:
        os.kill(int(pid), signal.SIGKILL); total += 1
    except Exception:
        pass
print("killed", total)
