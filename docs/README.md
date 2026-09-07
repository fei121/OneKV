# Documentation

The repo is documented by topic. Each group lives in its own folder so it's easy to find what you
need.

## Design — how it works & how it maps to the paper

| Doc | Content |
|---|---|
| [`design/architecture.md`](design/architecture.md) | Engine, shared-KV, P/D scheduling, continuous batching, prefix caching. |
| [`design/paper-alignment.md`](design/paper-alignment.md) | What we reproduced vs. the paper, and where we intentionally deviate. |

## Setup — how to set up and reproduce

| Doc | Content |
|---|---|
| [`setup/environment.md`](setup/environment.md) | Pinned server environment (hardware, CUDA, llama.cpp flags, model SHA, conda). |
| [`setup/llama-cpp-patch.md`](setup/llama-cpp-patch.md) | The four shared-KV llama.cpp patches + build/verify steps. |

## Benchmark — how it's measured and what we found

| Doc | Content |
|---|---|
| [`benchmark/methodology.md`](benchmark/methodology.md) | Workload, backends, metrics, and the "context" fairness rules. |
| [`benchmark/results.md`](benchmark/results.md) | Current valid results + figures + how to reproduce. |

## Notes — honest boundaries & lessons

| Doc | Content |
|---|---|
| [`notes/known-limitations.md`](notes/known-limitations.md) | Honest status, what we did / didn't reproduce, retraction history. |
| [`notes/pitfalls.md`](notes/pitfalls.md) | Hands-on lessons learned (service startup, harness, benchmark, shell). |

> Data artifacts (`results/`, `figures/`, `metrics/`) live at the repo root; they are linked from
> [`benchmark/results.md`](benchmark/results.md).
