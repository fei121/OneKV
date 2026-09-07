# llama.cpp green-context patch (engine-level, AGENTSERVE_GREEN_PCT)

**File**: `third_party/llama.cpp/ggml/src/ggml-cuda/common.cuh`
**Change**: in `ggml_backend_cuda_context::stream(int device, int stream)`, when env `AGENTSERVE_GREEN_PCT` is set, create the compute stream inside a CUDA Green Context so the model's CUDA compute is constrained to a fraction of SMs:
`cuDeviceGetDevResource -> cuDevSmResourceSplitByCount -> cuDevResourceGenerateDesc -> cuGreenCtxCreate -> cuGreenCtxStreamCreate -> cuCtxFromGreenCtx`.
Adds `#include <cuda.h>`; falls back to default stream if unavailable.

**Rebuild**: `cmake --build build --target llama-cli llama-server` (disable WebUI download: `cmake -DHF_ENABLED=OFF -DBUILD_UI=OFF .`).
**Verify**: `AGENTSERVE_GREEN_PCT=50 build/bin/llama-server -m <model> ...` -> log `[agentserve] green context stream: pct=50 sm=42`.
