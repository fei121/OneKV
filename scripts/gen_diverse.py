import json, base64, random
random.seed(1234)
files=["id_high.json","id_medium.json","id_low.json"]
items=[]
for f in files:
    d=json.load(open("/root/autodl-tmp/agentserve-reproduction/workloads/toolbench/MirrorAPI-Bench/test_cot/%s"%f))
    items.extend(d)
random.shuffle(items)
N=6
rows=[]
for i in range(N):
    it=items[i]
    # cold = system + instruction (shared system, unique instruction per task)
    cold = it["system"].rstrip("\n") + "\n\n" + it["instruction"].strip()
    d0 = 16  # small decode to measure TTFT; not significant for cold TTFT
    rows.append("div_%03d|%s|%d"%(i, base64.b64encode(cold.encode()).decode(), d0))
open("/root/autodl-tmp/exp/traces/diverse_sessions.txt","w").write("\n".join(rows)+"\n")
# report
print("generated %d diverse sessions"%N)
# report shared-system token estimate vs unique
sys_len=min(len(it["system"]) for it in items[:N])
print("system chars(min):",sys_len)
print("instruction chars sample:",[len(items[i]["instruction"]) for i in range(N)])
