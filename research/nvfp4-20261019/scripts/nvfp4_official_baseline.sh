#!/bin/bash
# baseline: official 0.1.45 engine (Strata-145, sm120) running the SAME NVFP4 files, vs the fork engine
set -u
NV=/root/strata/Strata-nvfp4
S145=/root/strata/Strata-145
M=/root/strata/Strata-data/models/qwen-nvfp4-gptq
OUT=/root/strata/nvfp4-official-baseline; mkdir -p $OUT
python3 - <<'PY'
import json
NV="/root/strata/Strata-nvfp4"; S145="/root/strata/Strata-145"; M="/root/strata/Strata-data/models/qwen-nvfp4-gptq"
d = json.load(open(f"{NV}/strata-nvfp4-gptq.json"))
d["exe"] = f"{S145}/engine/strata"
d["log"] = f"{NV}/strata-nvfp4-official.log"
json.dump(d, open(f"{NV}/strata-nvfp4-official.json","w"), indent=1, ensure_ascii=False)
print("official-engine config written")
PY
cat > $NV/run-nvfp4-official.sh <<EOF
#!/bin/sh
cd "$NV"
exec "/root/strata/Strata-145/.venv/bin/python" "$NV/serve/server.py" --engine strata --config "$NV/strata-nvfp4-official.json" --port 8080
EOF
chmod +x $NV/run-nvfp4-official.sh
python3 /root/strata/stop_strata.py >/dev/null 2>&1; sleep 3
setsid nohup bash $NV/run-nvfp4-official.sh > $OUT/serve.log 2>&1 < /dev/null &
for i in $(seq 1 140); do sleep 3; grep -q "^ready:" $OUT/serve.log && break; pgrep -f "server.py --engine strata --config .*nvfp4-official" >/dev/null || { echo SERVER_EXITED; break; }; done
echo "--- boot:"; tail -3 $OUT/serve.log | cut -c1-200
grep -iE "error|refus|type 40|nvfp4|w4a8|w4a4|VRAM free" $NV/strata-nvfp4-official.log | tail -8 | cut -c1-220
if curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(python3 -c 'import json;print(json.load(open("/root/strata/Strata-nvfp4/strata-nvfp4-gptq.json"))["api_key"])')" http://127.0.0.1:8080/health | grep -q 200; then
python3 - <<'PY'
import json, re, time, os, requests, random, uuid
NV="/root/strata/Strata-nvfp4"; M="/root/strata/Strata-data/models/qwen-nvfp4-gptq"
K=json.load(open(f"{NV}/strata-nvfp4-gptq.json"))["api_key"]; H={"Authorization":f"Bearer {K}"}
S=requests.Session(); S.trust_env=False
LOGF=f"{NV}/strata-nvfp4-official.log"
RE=re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
W="the quick brown fox jumps over the lazy dog while quantum mechanics perplexes researchers".split()
def hs(chars, seed):
    r=random.Random(seed); p=[f"S {uuid.uuid4().hex}.\n"]; t=0
    while t<chars:
        s=" ".join(r.choice(W) for _ in range(400))+".\n"; p.append(s); t+=len(s)
    c=f"{r.randint(10000,99999)}-QX-ALPHA"; p.insert(len(p)//2, f"IMPORTANT NOTE: THE HIDDEN MAGIC NUMBER IS {c}.\n")
    return ("Read then answer.\n\n"+"".join(p)+"\n\nQuestion: hidden magic number? Answer with the code only.\nAnswer:"), c
off=os.path.getsize(LOGF)
tot_g=tot_ms=0
for q in ["写一段约300字的散文，描写秋天傍晚的江边。","用 Python 实现一个线程安全的 LRU 缓存类，带详细中文注释。","数列满足 a1=1, a(n+1)=2a(n)+n，求通项公式并验证前5项。","Write a Python function that parses ISO 8601 durations like P3DT4H5M into seconds."]:
    for rep in range(2):
        S.post("http://127.0.0.1:8080/v1/chat/completions",headers=H,json={"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":q}],"max_tokens":700,"temperature":0,"reasoning_effort":"medium"},timeout=1200)
        time.sleep(0.5)
time.sleep(1)
d=open(LOGF,errors="replace").read()[off:]
for m in RE.finditer(d):
    g=m.groups(); tot_g+=int(g[5]); tot_ms+=int(g[6])
print(f"official-engine decode pooled {tot_g/tot_ms*1000:.1f} tok/s ({tot_g} tok)")
off=os.path.getsize(LOGF)
for name,chars in (("p9k",55000),("p53k",316000)):
    q,c=hs(chars, 7)
    S.post("http://127.0.0.1:8080/v1/chat/completions",headers=H,json={"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":q}],"max_tokens":200,"temperature":0,"reasoning_effort":"low"},timeout=2400)
    time.sleep(1)
    d=open(LOGF,errors="replace").read()[off:]
    m=list(RE.finditer(d))
    print(name, "prefill", m[-1].group(5) if m else "?", "tok/s", "| needle", c in d)
PY
fi
echo "--- stop official-engine test, restore fork NVFP4"
python3 /root/strata/stop_strata.py >/dev/null 2>&1; sleep 3
setsid nohup bash $NV/run-nvfp4-gptq.sh > $OUT/serve-restore.log 2>&1 < /dev/null &
for i in $(seq 1 140); do sleep 3; grep -q "^ready:" $OUT/serve-restore.log && break; done
grep "^ready:" $OUT/serve-restore.log | cut -c1-90
