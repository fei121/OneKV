# Runtime engine (`onekv_engine.cpp`)

A single-engine, shared-KV serving engine for Qwen-style models, built on a patched llama.cpp.

## Build

```bash
B=/path/to/llama.cpp/build
g++ -std=c++17 -O2 \
  -I$B/../include -I$B/../ggml/include -I/usr/local/cuda/include \
  -o onekv_engine onekv_engine.cpp \
  -L$B/bin -L/usr/local/cuda/lib64 -lllama -lggml -lggml-cuda -lggml-cpu -lggml-base \
  -lcuda -lcudart -Wl,-rpath,$B/bin
```

## Run

```bash
onekv_engine <sessions.txt> <N> [pre_stream] [dec_stream] [n_ctx]
# e.g.
onekv_engine sessions.txt 6 -1 1 49152
```

- `N` — number of concurrent agent sessions.
- `pre_stream` / `dec_stream` — CUDA stream index for prefill / decode (`-1` = default, `0/1` =
  green context streams).
- `n_ctx` — context size (large enough for `N` sessions' shared KV).

## Trace format

`sid|cold_b64|d0|app1_b64|d1|app2_b64|d2|...`

## Output

Writes an events JSONL (`TTFT_cold`, `TTFT_resume`, `TPOT`) to
`raw_logs/as_batch_events.jsonl` and prints throughput. Enables/disables the prefix cache with the
`CACHE` define (see the cold-prefill block).

## Configuration

The engine has two knobs relevant to reproduction:
- **Prefix caching**: prefill the shared system prompt once and `llama_memory_seq_cp` it to each
  session, then prefill only each session's instruction. The instruction start and `n_past` are
  both taken as `llama_memory_seq_pos_max(seq)+1` — this is required because `seq_cp`'s upper
  bound is exclusive, and it avoids the "inconsistent sequence positions" error.
- **CUDA green contexts**: set `pre_stream`/`dec_stream` to `0`/`1` and enable the
  `AGENTSERVE_PREFILL_PCT` env to route streams onto green contexts. (Measured: this *degrades*
  throughput here, so the default uses default streams.)

See [`../../docs/design/architecture.md`](../../docs/design/architecture.md) for details.
