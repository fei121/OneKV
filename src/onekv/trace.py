#!/usr/bin/env python3
"""
OneKV - synthesize 3-state token traces from StableToolBench (MirrorAPI-Bench).

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


def pick_unified_pool(items, size, seed):
    """Pick `size` distinct real ToolBench items as the UNIFIED task set (shared by ReAct and P&E)."""
    r = random.Random(seed)
    return r.sample(items, min(size, len(items)))


def assign_sessions(pool, n, rounds_choices, seed):
    """Deterministic per-session item assignment (SAME for ReAct and P&E -> controlled comparison).

    Returns a list of length `n`; element i is the list of items for session i.
    """
    r = random.Random(seed)
    sched = []
    for _ in range(n):
        rounds = r.choice(rounds_choices)
        sched.append(r.sample(pool, rounds))
    return sched


def build_session(tokenizer, session_items, cfg, paradigm, model_name, idx):
    p = cfg["trace"]["paradigms"][paradigm]
    cl, ch = cfg["trace"]["cold_prefill_tokens"]
    rounds = len(session_items)
    picks = session_items
    # Deterministic per-paradigm token sampling (same task, different length distribution).
    rng = random.Random(f"{cfg['trace']['seed']}:{paradigm}:{idx}")
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


import base64 as _b64

def save_unified_pool(path, pool, sched):
    """Persist the unified task pool + per-session assignment so future runs reuse the SAME tasks."""
    data = {"pool": pool, "sched": sched}
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_unified_pool(path):
    """Load a previously saved unified task pool + schedule."""
    with open(path) as f:
        d = json.load(f)
    return d.get("pool", []), d.get("sched", [])


def flatten_sessions(rows, model):
    """Flatten sessions to the engine's sessions.txt format:
    sid|b64(cold)|d0|b64(app1)|d1|...  (app* = bare append; engine wraps with <tool_result>)."""
    lines = []
    for s in rows:
        plan = s["plan"]; sid = s["session_id"]
        parts = [sid, _b64.b64encode(plan[0]["prompt"].encode()).decode(), str(plan[0]["decode_tokens"])]
        for r in range(1, len(plan)):
            app = plan[r].get("append") or plan[r]["prompt"][len(plan[r-1]["prompt"]):]
            parts.append(_b64.b64encode(app.encode()).decode())
            parts.append(str(plan[r]["decode_tokens"]))
        lines.append("|".join(parts))
    return lines


def main():
    ap = argparse.ArgumentParser(description="Generate OneKV token traces from StableToolBench.")
    ap.add_argument("--config", default="/root/autodl-tmp/exp/configs/trace_gen.yaml")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--toolbench-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--unified-pool-size", type=int, default=12,
                    help="extract N distinct real items as the UNIFIED task set (shared by ReAct=P&E)")
    ap.add_argument("--pool-file", default=None, help="persist/reuse the unified task set (json)")
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
    n_sessions = cfg["trace"]["sessions_per_paradigm"]
    model_name = cfg["model"]["name"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

    items = parse_toolbench(tb_dir, cfg["toolbench"]["cot_split"])
    if not items:
        raise RuntimeError("no ToolBench items parsed (check toolbench dir / split)")
    print(f"[load] tool items={len(items)}", flush=True)

    # UNIFIED task set: one pool + one per-session schedule, shared by BOTH paradigms
    # (so ReAct and P&E run the SAME real tasks; only prompt template + token counts differ).
    pool_file = Path(args.pool_file) if args.pool_file else out_dir / "unified_tasks.json"
    if pool_file.exists():
        pool, sched = load_unified_pool(pool_file)
        print(f"[pool] REUSED unified task set from {pool_file.name} ({len(pool)} items)", flush=True)
    else:
        pool = pick_unified_pool(items, args.unified_pool_size, cfg["trace"]["seed"])
        sched = assign_sessions(pool, n_sessions, cfg["trace"]["tool_rounds_per_session"],
                                cfg["trace"]["seed"] + 1000)
        save_unified_pool(pool_file, pool, sched)
        print(f"[pool] unified task set = {len(pool)} items -> saved {pool_file.name}",
              flush=True)

    all_lines = []
    for paradigm in cfg["trace"]["paradigms"]:
        rows = []
        for i in range(n_sessions):
            rows.append(build_session(tok, sched[i], cfg, paradigm, model_name, i))
        out_path = out_dir / f"traces_{paradigm}_{model_name.replace('/','-')}.jsonl"
        with open(out_path, "w") as f:
            for l in rows:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        # engine sessions.txt (flat) for this paradigm
        sfile = out_dir / f"sessions_{paradigm}.txt"
        with open(sfile, "w") as f:
            f.write("\n".join(flatten_sessions(rows, model_name)) + "\n")
        all_lines.extend(rows)
        print(f"[gen] {paradigm}: {n_sessions} sessions -> {out_path.name}, {sfile.name}", flush=True)

    stats = summarize(all_lines, cfg)
    stats_path = out_dir / "distribution_summary.json"
    json.dump({"table1_target": cfg["trace"]["paradigms"], "measured": stats},
              open(stats_path, "w"), indent=2)
    print(f"[write] summary -> {stats_path}")
    print("[summary]")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
