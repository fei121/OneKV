# OneKV — common commands.
# The Python used for trace generation / baselines lives in the vllm conda env.
PY=$(shell echo /root/autodl-tmp/conda_envs/vllm/bin/python)
CUDA=/usr/local/cuda-12.8/bin/nvcc
export PYTHONPATH=$(CURDIR)/src

# Model / trace locations (override via env)
MODEL      ?= /root/models/Qwen2.5-3B          # HF safetensors dir (vLLM / SGLang)
GGUF       ?= /root/autodl-tmp/models/Qwen2.5-3B-f16.gguf   # engine / llama.cpp
TRACE_DIR  ?= /root/autodl-tmp/exp/traces_unified

.PHONY: trace engine serve-llama serve-vllm serve-sglang sweep test clean

trace:
	$(PY) scripts/gen_traces.py --config configs/trace-gen.yaml

# Build + run the single-engine shared-KV runtime (innovation-1). A = number of concurrent agents.
engine:
	AGENTS=$(A) bash scripts/run_engine.sh

# llama.cpp / vLLM / SGLang baselines (self-contained multi-phase prompt, unified ReAct config).
serve-llama:
	$(PY) scripts/serve_llama.py --config configs/serving_react_u.yaml --agents $(A) --sessions $(S)

serve-vllm:
	$(PY) scripts/serve_backend.py --backend vllm --model-path $(MODEL) --config configs/serving_react_u.yaml --agents $(A) --sessions $(S)

serve-sglang:
	$(PY) scripts/serve_backend.py --backend sglang --model-path $(MODEL) --config configs/serving_react_u.yaml --agents $(A) --sessions $(S)

# Serial benchmark sweep (one backend at a time, GPU freed between runs).
sweep:
	bash scripts/sweep_robust.sh

test:
	$(PY) -m pytest tests/ -q

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
