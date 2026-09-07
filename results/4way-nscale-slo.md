# Qwen2.5-3B — 4-Way Session-Level SLO Attainment (N=3…10)

Follows the paper's metric definition: a session **passes** iff
`cold TTFT ≤ τ_TTFT` **and** `session-level TPOT p95 ≤ τ_TPOT`. The thresholds are
derived from an **isolated (N=1) model-device profile** on the RTX 3090, scaled by a
constant factor (as in `configs/metrics.yaml`).

## Thresholds

Profiled on the RTX 3090 with Qwen2.5-3B (solo N=1, same trace):

| quantity | isolated value | ×2.0 → τ |
|---|---|---|
| cold TTFT | **372 ms** (uncached cold prefill, 2693 tokens) | **744 ms** |
| session TPOT p95 | **10.15 ms** | **20.3 ms** |

> Note: the cold TTFT uses the **uncached** value (372 ms, first serial session), matching the
> paper's definition of cold prefill as the *long uncached system prompt*. With `cache_prompt`
> enabled, llama-server reuses the shared system prefix across serial sessions and reports
> ~25 ms — that is *not* the isolated cold latency and would make τ unrealistically tight.

## SLO attainment rate (fraction of sessions meeting both bounds)

| N | **shared-KV engine** | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| 3 | **1.00** | 1.00 | 0.75 | 0.25 |
| 4 | **1.00** | 0.75 | 0.25 | 0.00 |
| 5 | **1.00** | 1.00 | 0.33 | 0.08 |
| 6 | **1.00** | 0.75 | 0.25 | 0.00 |
| 7 | **1.00** | 0.83 | 0.25 | 0.00 |
| 8 | **1.00** | 0.58 | 0.25 | 0.00 |
| 9 | **1.00** | 0.75 | 0.50 | 0.00 |
| 10 | **1.00** | 0.50 | 0.75 | 0.00 |

![4-way SLO attainment](figures/4way-nscale-slo.png)

## Findings

- **The shared-KV engine holds 100% SLO attainment at every N.** Its per-session TTFT (shared
  system prefill amortized via prefix caching) and its flat TPOT p95 (12.5–15.4 ms) both stay
  comfortably inside the bounds even at N=10.
- **llama.cpp fails SLO from N=4 onward** (rate ≈ 0–25%): its decode latency explodes
  (TPOT p95 56 → 126 ms) far above τ_TPOT = 20.3 ms, so most sessions miss the TPOT bound.
- **vLLM / SGLang are in between** (0.25–1.00), degrading with concurrency; SGLang's higher
  latency tail (TPOT p95 ~24–28 ms) makes it fail more often than vLLM at mid-N.
- SLO is the paper's fourth headline metric (Fig. 6) and was **not previously computed** in the
  reproduction — this fills that gap. Earlier runs all reported `slo: null` because
  `tau_ttft_ms` / `tau_tpot_ms` were left null in `configs/metrics.yaml` (waiting on an isolated
  profile).

## Reproduction

- Thresholds: `scripts/compute_slo.py` regenerates `metrics/nscale-4way/slo_attainment.csv` from
  the per-N event logs (server-side) given the τ values above.
- Figure: `scripts/plot_4way_slo.py` renders `figures/4way-nscale-slo.png`.
- SLO data: `metrics/nscale-4way/slo_attainment.csv`. The scale factor (2.0) is the candidate in
  `configs/metrics.yaml`.
