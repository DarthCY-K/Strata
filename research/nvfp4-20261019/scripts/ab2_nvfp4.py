#!/usr/bin/env python3
"""NVFP4 decode A/B: base / CJK draft vocab / CJK vocab + 5060Ti helper. 6 arms interleaved (2 rounds),
same tasks (3 zh + 1 en) x 3 reps, pooled decode, ms/step, accept per category, prefill 9K/53K + needle."""
import copy, hashlib, json, os, random, re, subprocess, time, uuid
import requests

T = "/root/strata/Strata-nvfp4"
M = "/root/strata/Strata-data/models/qwen-nvfp4-gptq"
PROD = f"{T}/strata-nvfp4-gptq.json"
BASE = json.load(open(PROD))
CFG = f"{T}/strata-ab2.json"
RUN = f"{T}/run-ab2.sh"
ELOG = f"{T}/strata-ab2.log"
OUT = "/root/strata/nvfp4-ab2-20261019"
os.makedirs(OUT, exist_ok=True)
JL, PROG = f"{OUT}/results.jsonl", f"{OUT}/progress.log"
B = "http://127.0.0.1:8080"
KEY = BASE["api_key"]; H = {"Authorization": f"Bearer {KEY}"}
S = requests.Session(); S.trust_env = False
MODEL = "Qwen3.8-Flash-Next"
ARMS = [("base_a", "base"), ("cjk_a", "cjk"), ("cjkh_a", "cjk_helper"),
        ("base_b", "base"), ("cjk_b", "cjk"), ("cjkh_b", "cjk_helper")]
DECODE = {
    "zh_prose": "写一段约300字的散文，描写秋天傍晚的江边。",
    "code_zh": "用 Python 实现一个线程安全的 LRU 缓存类，带详细中文注释。",
    "math_zh": "数列满足 a1=1, a(n+1)=2a(n)+n，求通项公式并验证前5项。",
    "code_en": "Write a Python function that parses an ISO 8601 duration string (like P3DT4H5M) into total seconds, with a docstring and a few test asserts.",
}
WORDS = "the quick brown fox jumps over the lazy dog while quantum mechanics perplexes researchers".split()
RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%.*?\(([\d.]+)% of all")
RE_C1 = re.compile(r"CUDA1: (\d+) expert entries.*?host (\d+) ms staging\+launching, (\d+) ms waiting")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(PROG, "a", encoding="utf-8").write(s + "\n")


def rec(o):
    open(JL, "a", encoding="utf-8").write(json.dumps(o, ensure_ascii=False) + "\n")


def mkcfg(kind):
    d = copy.deepcopy(BASE)
    a = list(d["args"])
    if kind in ("cjk", "cjk_helper"):
        i = a.index("--mtp"); a[i + 1] = f"{M}/mtp-cjk"
    if kind == "cjk_helper":
        a += ["--expert-cache-device1", "auto", "--remote-expert-opt"]
        d["env"] = {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    else:
        d["env"] = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    d["args"] = a
    d["log"] = ELOG
    json.dump(d, open(CFG, "w"), indent=1, ensure_ascii=False)
    return d


open(RUN, "w").write(f'#!/bin/sh\ncd "{T}"\nexec "/root/strata/Strata-145/.venv/bin/python" "{T}/serve/server.py" '
                     f'--engine strata --config "{CFG}" --port 8080\n')
os.chmod(RUN, 0o755)


class Tail:
    def __init__(self):
        self.off = os.path.getsize(ELOG) if os.path.exists(ELOG) else 0

    def take(self):
        if not os.path.exists(ELOG):
            return []
        with open(ELOG, "rb") as f:
            f.seek(self.off); d = f.read()
        self.off += len(d)
        return d.decode("utf-8", "replace").splitlines()


def stop():
    subprocess.run(["python3", "/root/strata/stop_strata.py"], capture_output=True); time.sleep(3)


def start(tag, run=RUN):
    tail = Tail()
    subprocess.Popen(["bash", run], stdout=open(f"{OUT}/serve-{tag}.log", "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 420:
        time.sleep(3)
        try:
            if S.get(B + "/health", headers=H, timeout=4).json().get("loaded"):
                return round(time.time() - t0, 1), tail
        except Exception:
            pass
    return None, tail


def parse(lines):
    reqs = [m.groups() for m in (RE.search(x) for x in lines) if m]
    o = {}
    if reqs:
        cold = max(reqs, key=lambda g: int(g[2]))
        o.update(prompt=int(cold[0]), reused=int(cold[1]), read=int(cold[2]), prefill=float(cold[4]))
        g = sum(int(r[5]) for r in reqs); ms = sum(int(r[6]) for r in reqs)
        o.update(gen=g, gen_ms=ms, tps=round(g / ms * 1000, 2) if ms else None,
                 acc=sum(int(r[8]) for r in reqs), drafted=sum(int(r[9]) for r in reqs))
    h = [m.groups() for m in (RE_HIT.search(x) for x in lines) if m]
    if h:
        o.update(hit=float(h[-1][0]), pcie_share=float(h[-1][1]))
    c = [m.groups() for m in (RE_C1.search(x) for x in lines) if m]
    if c:
        o.update(c1_wait=int(c[-1][2]))
    return o


def chat(tail, content, mx, effort):
    tail.take()
    j = S.post(B + "/v1/chat/completions", headers=H, timeout=2400,
               json={"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": mx,
                     "temperature": 0, "reasoning_effort": effort, "uncensored": False}).json()
    time.sleep(0.8)
    m = j["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + "\n<<>>\n" + (m.get("content") or "")
    o = parse(tail.take())
    o.update(md5=hashlib.md5(txt.encode()).hexdigest()[:10], content=(m.get("content") or "")[-60:])
    return o


def haystack(chars, seed):
    rnd = random.Random(seed)
    parts, tot = [f"Session {uuid.uuid4().hex}.\n"], 0
    while tot < chars:
        s = " ".join(rnd.choice(WORDS) for _ in range(400)) + ".\n"; parts.append(s); tot += len(s)
    code = f"{rnd.randint(10000,99999)}-QX-{rnd.choice(['ALPHA','DELTA','SIGMA'])}"
    parts.insert(len(parts)//2, f"IMPORTANT NOTE: THE HIDDEN MAGIC NUMBER IS {code}. PLEASE REMEMBER IT.\n")
    return ("Read the long document then answer.\n\n" + "".join(parts) +
            "\n\nQuestion: What is the hidden magic number? Answer with the exact code only.\nAnswer:"), code


def summarize():
    rows = [json.loads(x) for x in open(JL, encoding="utf-8")]
    summ = {}
    for arm, kind in ARMS:
        d = [r for r in rows if r["arm"] == arm and r["kind"] == "decode" and r.get("gen_ms")]
        if not d:
            summ[arm] = {"error": True}; continue
        g = sum(r["gen"] for r in d); ms = sum(r["gen_ms"] for r in d)
        acc = sum(r["acc"] for r in d); dr = sum(r["drafted"] for r in d)
        bycat = {}
        for c in DECODE:
            dc = [r for r in d if r["cat"] == c]
            gc = sum(r["gen"] for r in dc); mc = sum(r["gen_ms"] for r in dc)
            bycat[c] = {"tps": round(gc / mc * 1000, 1) if mc else None,
                        "acc": round(sum(r["acc"] for r in dc) / max(1, sum(r["drafted"] for r in dc)), 3)}
        pf = {r["cat"]: (r.get("prefill"), r.get("needle")) for r in rows if r["arm"] == arm and r["kind"] == "prefill"}
        hits = [r["hit"] for r in d if r.get("hit") is not None]
        waits = [r["c1_wait"] for r in d if r.get("c1_wait") is not None]
        boot = next((r for r in rows if r["arm"] == arm and r["kind"] == "boot"), {})
        summ[arm] = {"kind": kind, "tps": round(g / ms * 1000, 2), "ms_step": round(ms / max(1, g - acc), 3),
                     "accept": round(acc / dr, 4) if dr else None, "by_cat": bycat, "prefill": pf,
                     "hit": round(sum(hits) / len(hits), 2) if hits else None,
                     "c1_wait_mean": round(sum(waits) / len(waits), 1) if waits else None, "boot": boot.get("lines")}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)


status = "ok"
try:
    log("=== NVFP4 A/B2 start", [a for a, _ in ARMS])
    for arm, kind in ARMS:
        d = mkcfg(kind)
        log(f"--- arm {arm} kind={kind} mtp={d['args'][d['args'].index('--mtp') + 1].split('/')[-1]} cvis={d['env']['CUDA_VISIBLE_DEVICES']}")
        stop()
        ready, tail = start(arm)
        boot = [x for x in tail.take() if re.search(r"draft head over|CUDA1: \d+ additional|expert cache \d+ slots|VRAM free|ADAPT_FETCH", x)]
        rec({"arm": arm, "kind": "boot", "ready_s": ready, "lines": boot[-6:]})
        log(f"{arm} ready={ready} | " + " || ".join(boot[-4:])[:300])
        if ready is None:
            continue
        for i in range(2):
            chat(tail, f"热身{i}：一句话介绍你自己。", 64, "low")
        for rep in range(3):
            for cat, q in DECODE.items():
                p = chat(tail, q, 700, "medium")
                rec({"arm": arm, "kind": "decode", "cat": cat, "rep": rep, **p})
                log(f"{arm} {cat} r{rep}: {p.get('tps')} tok/s acc {p.get('acc')}/{p.get('drafted')} hit {p.get('hit')} c1wait {p.get('c1_wait')}")
        for name, chars in (("p9k", 55_000), ("p53k", 316_000)):
            q, code = haystack(chars, seed=sum(map(ord, name)))
            p = chat(tail, q, 200, "low")
            p["needle"] = code in p["content"]
            rec({"arm": arm, "kind": "prefill", "cat": name, **p})
            log(f"{arm} {name}: prompt {p.get('prompt')} prefill {p.get('prefill')} needle {p['needle']}")
except BaseException as ex:
    status = f"fatal: {ex!r}"; log(status)
finally:
    try:
        summarize()
    except Exception as ex:
        log("summary err", ex)
    log("--- restore production NVFP4")
    stop()
    rdy, _ = start("restore", run=f"{T}/run-nvfp4-gptq.sh")
    log(f"restore ready {rdy}")
    json.dump({"status": status, "restore": rdy, "t": time.time()}, open(f"{OUT}/completion.json", "w"))
    log("=== ALL DONE ===", status)
