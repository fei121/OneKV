#!/bin/bash
set -u
EXP=/root/autodl-tmp/exp
VL=/root/autodl-tmp/conda_envs/vllm/bin/python3
SG=/root/autodl-tmp/conda_envs/sglang/bin/python3
MODEL=/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf
KG=$EXP/scripts/kill_gpu.py
PYV=/root/autodl-tmp/conda_envs/vllm/bin/python3
cd $EXP || exit 1
cleanup(){ $PYV $KG >/dev/null 2>&1; sleep 3; }
run_para() {
  local PARA=$1; local TRACE_ENG=$2; local CFG=$3
  for N in 3 4 5 6; do
    echo "===== $PARA N=$N ====="
    NCX=$([ $N -le 4 ] && echo 49152 || echo 65536)
    cleanup
    env -u AGENTSERVE_PREFILL_PCT -u AGENTSERVE_DECODE_PCT /tmp/as_conc_batch $TRACE_ENG $N -1 1 $NCX $MODEL 12 0 2>&1 | grep -E "DONE" >> $EXP/results/r_${PARA}_engine.log
    cp $EXP/raw_logs/as_batch_events.jsonl $EXP/raw_logs/${PARA}_engine_N${N}_events.jsonl
    echo "engine $PARA N=$N done"
    cleanup
    timeout 200 $VL $EXP/scripts/serve_llama.py --config $CFG --agents $N --sessions 12 --event-log $EXP/raw_logs/${PARA}_llama_N${N}_events.jsonl --results $EXP/results/${PARA}_llama_N${N}_metrics.json >/dev/null 2>&1
    echo "llama $PARA N=$N done"
    cleanup
    timeout 300 $VL $EXP/scripts/serve_backend.py --backend vllm --config $CFG --agents $N --sessions 12 --tag ${PARA}_vllm_N${N} >/dev/null 2>&1
    echo "vllm $PARA N=$N done"
    cleanup
    timeout 360 $SG $EXP/scripts/serve_backend.py --backend sglang --config $CFG --agents $N --sessions 12 --tag ${PARA}_sglang_N${N} >/dev/null 2>&1
    echo "sglang $PARA N=$N done"
    cleanup
  done
}
run_para react $EXP/traces_unified/sessions_react.txt $EXP/configs/serving_react_u.yaml
echo "----- REACT DONE -----"
run_para pe $EXP/traces_unified/sessions_plan_and_execute.txt $EXP/configs/serving_pe_u.yaml
echo "----- PE DONE -----"
echo "ALL DONE"
