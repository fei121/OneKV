import json, base64, random
random.seed(777)
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
    cold = it["system"].rstrip("\n") + "\n\n" + it["instruction"].strip()
    d0 = 20                       # decode after cold prefill
    resume = ("Tool result for task %d: ok" % i)  # short synthetic tool output
    d1 = 20
    # sessions format: sid|cold|d0|resume|d1
    rows.append("div_%03d|%s|%d|%s|%d"%(i, base64.b64encode(cold.encode()).decode(), d0, base64.b64encode(resume.encode()).decode(), d1))
open("/root/autodl-tmp/exp/traces/diverse_full.txt","w").write("\n".join(rows)+"\n")
print("generated diverse_full N=%d"%N)
