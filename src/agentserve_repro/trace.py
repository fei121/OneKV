#!/usr/bin/env python3
"""
AgentServe reproduction - synthesize 3-state token traces from StableToolBench (MirrorAPI-Bench).

Each agent session = 1 cold prefill + R rounds of (resume_prefill + short decode).

IMPORTANT (fixed): the traces now use the REAL ToolBench request as the user task and the REAL
tool output as each resume append, so the model has an actual task to perform (previously the user
message was a generic placeholder, which caused the model to refuse and then repeat random outputs).

Token counts are aligned to the paper's Table I:
    Cold Prefill      : 2.5k-3.5k input tokens (system prompt + tool descriptions)
    ReAct resume      : 30-127 (avg ~56)   |  ReAct decode     : 27-99  (avg ~38)
    P&E   resume      : 125-421 (avg ~251) |  P&E   decode     : 41-125 (avg ~58)

Output: JSONL (one session per line) + summary stats.
"""
import os, sys, json, yaml, random, argparse, collections
from pathlib import Path

REACT_HEAD = (
    "You are an expert AI agent that operates in a ReAct loop. "
    "You have access to a set of tools. For each step:\n"
    "  1. Thought: reason about the current situation.\n"
    "  2. Action: choose ONE tool and provide required arguments.\n"
    "  3. Observation: read the tool result, then continue.\n"
    "You MUST emit tool calls as a JSON object with keys 'name' and 'arguments' "
    "inside <tool_call></tool_call> tags. Never fabricate tool outputs. "
    "When the task is finished, produce a concise final answer.\n"
)

PELAN_HEAD = (
    "You are an expert AI agent that operates in a Plan-and-Execute loop. "
    "You have access to a set of tools. Your workflow:\n"
    "  1. Plan: produce a numbered list of steps before acting.\n"
    "  2. Execute: carry out each step by calling tools.\n"
    "  3. Observe: incorporate tool results and revise the plan as needed.\n"
    "You MUST emit tool calls as a JSON object with keys 'name' and 'arguments' "
    "inside <tool_call></tool_call> tags. Never fabricate tool outputs. "
    "When every plan step is done, produce a concise final answer.\n"
)

def _extract_request(inst):
    """Pull the "Request:" block out of a StableToolBench instruction (the actual task query)."""
    marker = "\nRequest:"
    idx = inst.find(marker)
    if idx == -1:
        return None
    return inst[idx + len(marker):].strip()


def parse_toolbench(toolbench_dir, split="test_cot"):
    """Parse MirrorAPI-Bench split -> list of {task, schema, output} items.

    Each item pairs the REAL tool request (the task) with the REAL tool output, so an agent
    session can have a genuine task and consistent tool results.
    """
    items = []
    split_dir = Path(toolbench_dir) / split
    for fp in sorted(split_dir.glob("*.json")):
        data = json.load(open(fp))
        if not isinstance(data, list):
            continue
        for it in data:
            inst = it.get("instruction", "")
            out = str(it.get("output", ""))
            task = _extract_request(inst)
            if not task:
                continue
            # Extract the API schema from the "API doc:" block.
            if "API doc:" in inst:
                doc_part = inst.split("API doc:", 1)[1].strip()
                cand = _balanced_dict(doc_part)
                if cand:
                    for loader in (json.loads, _literal):
                        try:
                            api = loader(cand)
                            if isinstance(api, dict) and api.get("api_name"):
                                items.append({"task": task, "schema": api, "output": out})
                                break
                        except Exception:
                            continue
    return items


def _literal(s):
    import ast
    return ast.literal_eval(s)

def _balanced_dict(s):
    """Return the first balanced { ... } block in s (handles nested braces)."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def get_head(paradigm):
    return REACT_HEAD if paradigm == "react" else PELAN_HEAD


def serialize_tool(schema):
    """Render one tool schema to a compact JSON block for the system prompt."""
    out = [schema.get("tool_name") or schema.get("api_name", "tool")]
    d = {
        "api_name": schema.get("api_name"),
        "description": schema.get("api_description") or schema.get("tool_description"),
        "required": schema.get("required_parameters") or [],
        "optional": schema.get("optional_parameters") or [],
    }
    out.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(out)


def make_task(items, paradigm):
    """Build a concrete user task from a set of real ToolBench requests."""
    if paradigm == "react":
        head = ("Use the available tools to complete the following requests, one at a time. After each "
                "tool result, decide the next action. When every request is done, give a concise final answer.\n\n")
    else:
        head = ("Plan the following requests, then execute them one at a time with the tools. "
                "Incorporate each tool result, and when done give a concise final answer.\n\n")
    body = "\n".join(f"{i+1}. {it['task']}" for i, it in enumerate(items))
    return head + body


def build_cold_prompt(tokenizer, paradigm, schemas, task_str, lo, hi):
    """Grow the system+tool prompt until token count lands in [lo, hi]."""
    head = get_head(paradigm)
    sel = []

    def render():
        sys_content = head + "\n\nYou have access to the following tools:\n\n# Tools\n" + \
                      "\n\n".join("### " + serialize_tool(t) for t in sel)
        msgs = [{"role": "system", "content": sys_content},
                {"role": "user", "content": task_str}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def ntok():
        return len(tokenizer.encode(render()))

    # grow until we meet the low bound (need a few tools at minimum)
    for t in schemas:
        sel.append(t)
        if len(sel) >= 2 and ntok() >= lo:
            break
    # trim until within high bound
    while len(sel) > 2 and ntok() > hi:
        sel.pop()
    # if pool too small to reach lo, pad by repeating the tool list section (keeps it realistic-ish)
    if ntok() < lo and schemas:
        base = list(sel)
        k = 1
        while ntok() < lo and k < 6:
            sel = base * (k + 1)
            k += 1
    prompt = render()
    return prompt, ntok(), sel


def sample_target(avg, lo, hi, rng):
    """Sample a token-target centered on avg with adaptive sigma (avoids clip bias)."""
    span = min(avg - lo, hi - avg, (hi - lo) / 6)
    sigma = max(2.0, span / 3)
    t = int(round(rng.gauss(avg, sigma)))
    return max(lo, min(hi, t))


MAX_RAW_CHARS = 60000

def _clamp(s):
    return s if len(s) <= MAX_RAW_CHARS else s[:MAX_RAW_CHARS]


def fit_append(tokenizer, output, target):
    """Return a real tool-output string whose token length is as close as possible to `target`."""
    txt = _clamp(output)
    ids = tokenizer.encode(txt)
    if len(ids) == target:
        return txt, len(ids)
    if len(ids) > target:
        return tokenizer.decode(tokenizer.encode(txt)[:target]), target
    # shorter than target: pad with a neutral note (keeps it a coherent single tool result)
    pad = "\n(The tool returned no additional detail.)"
    ids_pad = tokenizer.encode(pad)
    while len(ids) + len(ids_pad) < target:
        ids = ids + ids_pad
    return tokenizer.decode(ids[:target]), len(tokenizer.encode(tokenizer.decode(ids[:target])))


def build_session(tokenizer, items, cfg, paradigm, model_name, idx, rng):
    p = cfg["trace"]["paradigms"][paradigm]
    cl, ch = cfg["trace"]["cold_prefill_tokens"]
    rounds = rng.choice(cfg["trace"]["tool_rounds_per_session"])

    # Pick `rounds` real ToolBench items (task + tool + output).
    if len(items) < rounds:
        picks = [rng.choice(items) for _ in range(rounds)]
    else:
        picks = rng.sample(items, rounds)
    unique_schemas = []
    seen = set()
    for it in picks:
        key = it["schema"].get("api_name", it["schema"].get("tool_name"))
        if key not in seen:
            seen.add(key)
            unique_schemas.append(it["schema"])

    task_str = make_task(picks, paradigm)
    cold_prompt, cold_tokens, _ = build_cold_prompt(tokenizer, paradigm, unique_schemas, task_str, cl, ch)

    plan = [{"phase": "cold_prefill", "prompt": cold_prompt,
             "input_tokens": cold_tokens,
             "decode_tokens": sample_target(p["decode_avg"], p["decode"][0], p["decode"][1], rng)}]

    prev = cold_prompt
    for r in range(rounds):
        rt = sample_target(p["resume_avg"], p["resume_prefill"][0], p["resume_prefill"][1], rng)
        append, at = fit_append(tokenizer, picks[r]["output"], rt)
        prev = prev + "\n\n<tool_result>\n" + append + "\n</tool_result>\n"
        plan.append({"phase": "resume_prefill", "prompt": prev, "append": append,
                     "input_tokens": at,
                     "decode_tokens": sample_target(p["decode_avg"], p["decode"][0], p["decode"][1], rng)})

    return {"session_id": f"{paradigm.replace('_','-')}_{model_name.replace('/','-')}_s{idx:03d}",
            "paradigm": paradigm, "model": model_name, "plan": plan}


def summarize(lines, cfg):
    stats = {}
    for paradigm in cfg["trace"]["paradigms"]:
        rows = [l for l in lines if l["paradigm"] == paradigm]
        cold = [s["input_tokens"] for s in (r["plan"][0] for r in rows)]
        res = [s["input_tokens"] for r in rows for s in r["plan"] if s["phase"] == "resume_prefill"]
        dec = [s["decode_tokens"] for r in rows for s in r["plan"]]
        def agg(v):
            return {"min": min(v), "max": max(v), "avg": round(sum(v) / len(v), 1), "n": len(v)}
        stats[paradigm] = {"cold_prefill": agg(cold), "resume_prefill": agg(res), "decode": agg(dec)}
    return stats


def main():
    ap = argparse.ArgumentParser(description="Generate AgentServe token traces from StableToolBench.")
    ap.add_argument("--config", default="/root/autodl-tmp/exp/configs/trace_gen.yaml")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--toolbench-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.model_dir: cfg["model"]["dir"] = args.model_dir
    if args.toolbench_dir: cfg["toolbench"]["dir"] = args.toolbench_dir
    if args.out_dir: cfg["trace"]["out_dir"] = args.out_dir
    if args.sessions: cfg["trace"]["sessions_per_paradigm"] = args.sessions
    if args.seed: cfg["trace"]["seed"] = args.seed

    model_dir = cfg["model"]["dir"]
    tb_dir = cfg["toolbench"]["dir"]
    out_dir = Path(cfg["trace"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

    items = parse_toolbench(tb_dir, cfg["toolbench"]["cot_split"])
    if not items:
        raise RuntimeError("no ToolBench items parsed (check toolbench dir / split)")
    print(f"[load] tool items={len(items)}", flush=True)

    rng = random.Random(cfg["trace"]["seed"])
    lines = []
    for paradigm in cfg["trace"]["paradigms"]:
        for i in range(cfg["trace"]["sessions_per_paradigm"]):
            lines.append(build_session(tok, items, cfg, paradigm,
                                       cfg["model"]["name"], i, rng))
        print(f"[gen] {paradigm}: {cfg['trace']['sessions_per_paradigm']} sessions", flush=True)

    out_path = out_dir / f"traces_{cfg['model']['name'].replace('/','-')}.jsonl"
    with open(out_path, "w") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    stats = summarize(lines, cfg)
    stats_path = out_dir / "distribution_summary.json"
    json.dump({"table1_target": cfg["trace"]["paradigms"], "measured": stats},
              open(stats_path, "w"), indent=2)
    print(f"[write] {out_path} ({len(lines)} sessions)")
    print(f"[write] {stats_path}")
    print("[summary]")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
