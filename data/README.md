# Data

The raw data used to drive the benchmark. Everything here can regenerate the figures/results in
[`../results/`](../results/) with the scripts in [`../scripts/`](../scripts/).

## Layout

| Path | What it is |
|---|---|
| `unified_tasks.json` | The **unified 12-task set** (tasks + schedule) shared by the ReAct and Plan-and-Execute traces. Generated once and reused across runs. |
| `traces/` | The **generated 3-state agent traces** (per paradigm, per model). |

### `traces/`

- `traces_react_Qwen2.5-{3B,7B}.jsonl` — ReAct traces (one JSON object per session, per-phase
  prompt + decode budget).
- `traces_plan_and_execute_Qwen2.5-{3B,7B}.jsonl` — Plan-and-Execute traces.
- `distribution_summary.json` — the Table-I token-distribution the traces were sampled to match.

> The engine's flattened input (`sessions_{react|plan_and_execute}.txt`) and the per-paradigm traces are
> **generated** by `scripts/gen_traces.py` (see [`../configs/trace-gen.yaml`](../configs/trace-gen.yaml)),
> so they are not committed — regenerate them on demand:
> ```bash
> python scripts/gen_traces.py --config configs/trace-gen.yaml
> ```
> The 3B and 7B trace JSONL files differ only by the model tag in the session id (they share the Qwen
> tokenizer, so the task text is identical).

## Source dataset (external dependency)

The traces are built from the **ToolBench `MirrorAPI-Bench`** toolset (the `test_cot` split: tool
schemas + requests + outputs). This is a **third-party dataset** and is **not vendored** here (it is
~5.6 MB for the `test_cot` split, 57 MB for the whole benchmark). `configs/trace-gen.yaml`
(`toolbench.dir` + `toolbench.cot_split = test_cot`) points to its location on the machine; to fetch it
elsewhere, obtain `MirrorAPI-Bench` from its upstream repo (or via `hfd`) and update the config path.

The generated `traces/*.jsonl` already embed the task/schema/output used, so the benchmark and its
figure/results are reproducible from this repo **without** the source dataset.

## Provenance

- Traces generated with `seed = 1234`, `sessions_per_paradigm = 60`, `tool_rounds_per_session = [2,3,4]`,
  matching Table-I (see [`configs/trace-gen.yaml`](../configs/trace-gen.yaml) +
  [`../docs/setup/environment.md`](../docs/setup/environment.md)).
- The **unified 12-task set** used for the reported per-paradigm comparison is
  [`perparadigm`](../metrics/perparadigm/) (the first 12 items of each paradigm trace).
