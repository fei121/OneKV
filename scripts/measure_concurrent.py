import requests,time,json,threading
url="http://127.0.0.1:8089/completion"
res={}
def send(i):
    p=open("/tmp/div_%d.txt"%i).read(); t0=time.time(); ft=None
    r=requests.post(url,json={"prompt":p,"n_predict":2,"stream":True,"temperature":0},stream=True,timeout=120)
    for line in r.iter_lines():
        if not line:continue
        s=line.decode("utf-8").strip()
        if s.startswith("data:"):
            d=json.loads(s[5:].strip())
            if d.get("content") and ft is None: ft=time.time()
    res[i]=(ft-t0)*1000 if ft else None
ths=[threading.Thread(target=send,args=(i,)) for i in range(6)]
for t in ths:t.start()
for t in ths:t.join()
vals=sorted(v for v in res.values() if v is not None)
print("llama-server DIVERSE N=6 concurrent cold TTFT:",["%.0f"%v for v in vals],"p50=%.0f p95=%.0f"%(vals[len(vals)//2],vals[int(len(vals)*.95)]))
