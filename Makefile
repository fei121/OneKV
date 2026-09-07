PY=$(shell echo /root/autodl-tmp/conda_envs/vllm/bin/python)
CUDA=/usr/local/cuda-12.8/bin/nvcc
export PYTHONPATH=$(CURDIR)/src

.PHONY: trace serve-llama serve-vllm serve-sglang compare engine test report clean

trace:
	$(PY) scripts/gen_traces.py --config configs/trace-gen.yaml

serve-llama:
	$(PY) scripts/serve_llama.py --config configs/serving.yaml --agents $(A) --sessions $(S)

serve-vllm:
	$(PY) scripts/serve_backend.py --backend vllm --agents $(A) --sessions $(S)

serve-sglang:
	$(PY) scripts/serve_backend.py --backend sglang --agents $(A) --sessions $(S)

compare:
	$(PY) scripts/backend_compare.py

engine:
	bash scripts/run_engine.sh --agents 3

test:
	$(PY) -m pytest tests/ -q

report:
	$(PY) scripts/backend_compare.py
	$(PY) scripts/make_results_log.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
