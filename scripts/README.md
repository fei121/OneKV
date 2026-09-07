# Scripts

| Script | Purpose |
|---|---|
| `gen_traces.py` | Generate 3-state (cold/resume/decode) agent traces (Table I distribution). |
| `serve_llama.py` | Drive llama-server as a baseline (N concurrent, slot-pinned, per-token events). |
| `serve_backend.py` | Drive vLLM / SGLang / AgentServe backends. |
| `serve_dualgreen.py` / `serve_dualgreen_controller.py` | Dual-instance green-context serving. |
| `run_engine.sh` | Build + run the single-engine shared-KV runtime (`src/runtime/agentserve_engine.cpp`). |
| `backend_compare.py` | Aggregate backend events → comparison table + figures. |
| `make_results_log.py` | Rebuild `results/results_log.md`. |
| `analyze_sm_profile.py` | Analyze SM-scaling profile. |
| `verify_distribution.py` | Verify generated trace matches Table I. |
| `cold_measure.cpp` | Micro-benchmark of cold-prefill cost (full vs system-only vs instruction-only). |
| `gen_diverse.py` / `gen_diverse_full.py` | Generate a diverse-task trace (shared system + unique instruction). |
| `plot_*.py` | Regenerate comparison figures. |
| `plot_4way_nscale.py` | 4-way N-scale figure (engine vs llama.cpp / vLLM / SGLang, N=3…10). |
| `run_llama_baseline.sh` | Thin wrapper for `serve_llama.py`. |
| `run_hol_sweep.sh` / `run_sm_task.sh` | Head-of-line and SM-scaling sweeps. |
