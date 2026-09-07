#!/bin/bash
# Task 6 (first pass) — HoL / TPOT spike under concurrent agents.
cd /root/autodl-tmp/exp
PY=/root/autodl-tmp/conda_envs/vllm/bin/python
mkdir -p raw_logs results
for N in 1 3 6; do
  pkill -f 'llama-server' 2>/dev/null; sleep 2
  echo "############ N=$N ############"
  $PY scripts/serve_llama.py --config configs/serving.yaml --agents $N --sessions 12 \
    --event-log /root/autodl-tmp/exp/raw_logs/events_N${N}.jsonl \
    --results /root/autodl-tmp/exp/results/metrics_N${N}.json 2>&1 | tail -40
done
echo "SWEEP DONE"
