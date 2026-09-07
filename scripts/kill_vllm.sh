#!/bin/bash
pkill -f 'vllm.entrypoints' 2>/dev/null
pkill -f 'openai.api_server' 2>/dev/null
pkill -f 'sglang.launch_server' 2>/dev/null
pkill -f 'sglang' 2>/dev/null
sleep 3
echo 'killed vllm/sglang servers'
