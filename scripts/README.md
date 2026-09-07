# Scripts

| Script | Purpose |
|---|---|
| `gen_traces.py` | Generate real-task ToolBench traces (unified 12-task set shared by ReAct & P&E; per-paradigm output + engine `sessions_*.txt`). |
| `serve_llama.py` | Drive the llama.cpp (llama-server) baseline with a **self-contained** multi-phase prompt. |
| `serve_backend.py` | Drive the vLLM / SGLang baselines (`--model-path`, self-contained prompt, per-token `yield from`). |
| `run_engine.sh` | Build + run the single-engine shared-KV runtime (`src/runtime/agentserve_engine.cpp`). |
| `sweep_robust.sh` | Serial benchmark sweep (ReAct + P&E, N=3…6), one backend at a time, GPU freed between runs (3B). |
| `sweep_robust_7b.sh` | Same serial sweep for Qwen2.5-7B. |
| `kill_gpu.py` | Kill any leftover serving backend / GPU compute process (used between runs). |
| `plot_perparadigm.py` | Per-paradigm (ReAct / P&E) 4-way figures for 3B and 7B. |
| `plot_4way_v2.py` | Combined 4-way N-scale figure (real-task trace + fixed baselines, N=3…10, 3B). |
| `verify_distribution.py` | Verify the generated trace matches the paper's Table-I token distribution. |

> **Unified task set** (`data/unified_tasks.json`, 12 items + schedule) is shared by both paradigms;
> regenerate traces with `--pool-file` to reuse the same tasks across runs.
