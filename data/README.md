# Data

The raw data used to drive the benchmark. Everything here can regenerate the figures/results in
[`../results/`](../results/) with the scripts in [`../scripts/`](../scripts/).

## Layout

| Path | What it is |
|---|---|
| `unified_tasks.json` | The **unified 12-task set** (tasks + schedule) shared by the ReAct and Plan-and-Execute traces. Generated once and reused across runs. |
| `traces/` | The **generated 3-state agent traces** (per paradigm, per model). |
| `toolbench/MirrorAPI-Bench/test_cot/` | The **ToolBench `MirrorAPI-Bench` `test_cot` split** (source of tool schemas, requests, and outputs) that `scripts/gen_traces.py` reads to build the traces. |

### `traces/`

- `traces_react_Qwen2.5-{3B,7B}.jsonl` — ReAct traces (one JSON object per session, per-phase
  prompt + decode budget).
- `traces_plan_and_execute_Qwen2.5-{3B,7B}.jsonl` — Plan-and-Execute traces.
- `sessions_react.txt`, `sessions_plan_and_execute.txt` — flattened engine input format
  (`sid|cold_b64|d0|app1_b64|d1|...`), shared by 3B/7B (same tokenizer → identical tokens).
- `distribution_summary.json` — the Table-I token-distribution the traces were sampled to match.

> The 3B and 7B trace JSONL files are byte-identical apart from the model tag in the filename —
> 3B and 7B share the Qwen tokenizer, so the task text is the same.

### `toolbench/MirrorAPI-Bench/test_cot/`

The `id_high.json` / `id_low.json` / `id_medium.json` / `ood.json` / `ood_failure.json` subsets
of the **MirrorAPI-Bench** toolset that `configs/trace-gen.yaml` (`toolbench.cot_split = test_cot`)
uses as the *source* of tool schemas + outputs. The full dataset is an external dependency; only the
`test_cot` split actually consumed by the generator is vendored here.

## Regenerate

```bash
# source dataset is vendored at data/toolbench/MirrorAPI-Bench; config may point to another copy
python scripts/gen_traces.py --config configs/trace-gen.yaml
```

## Provenance

- Traces generated with `seed = 1234`, `sessions_per_paradigm = 60`, `tool_rounds_per_session = [2,3,4]`,
  matching Table-I (see [`configs/trace-gen.yaml`](../configs/trace-gen.yaml) +
  [`../docs/setup/environment.md`](../docs/setup/environment.md)).
- The **unified 12-task set** used for the reported per-paradigm comparison is
  [`perparadigm`](../metrics/perparadigm/) (the first 12 items of each paradigm trace).
