#!/usr/bin/env python3
"""Controlled opt123 screen; independent engine per arm, one binary and unchanged production JSON.
Restores PcieAuto-fixed NVFP4 on success, failure or SIGTERM. Never promotes a winner.
"""
import os,sys,pathlib,json,time,random,hashlib,subprocess,signal,socket,re,fcntl
T=pathlib.Path('/root/strata/Strata-nvfp4')
PROD=T/'strata-nvfp4-gptq.json'
BASE_EXE='/root/strata/releases/strata-nvfp4-pa-20261019-1729'
CAND='/root/strata/releases/strata-nvopt123-20261019a'
OUT=pathlib.Path('/root/strata/nvopt123-ab-20261019'); OUT.mkdir(exist_ok=True,mode=0o700)
if '--launch' in sys.argv:
    with (OUT/'worker.log').open('a') as f:
        p=subprocess.Popen([sys.executable,__file__],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'worker_pid':p.pid,'out':str(OUT)})); sys.exit(0)
import requests
lock=(OUT/'window.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'results.jsonl').exists(),'Refuse to mix repeated runs'
raw=PROD.read_bytes(); base=json.loads(raw)
assert base['exe']==BASE_EXE and '--pcie-frac' not in base['args']
assert pathlib.Path(CAND).is_file()
(OUT/'production.before.json').write_bytes(raw); (OUT/'production.before.json').chmod(0o600)
S=requests.Session(); S.trust_env=False
B='http://127.0.0.1:8080'; H={'Authorization':'Bearer '+base['api_key']}
MODEL='Qwen3.8-Flash-Next'
RE=re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')
FLAGS=['STRATA_REMOTE_REDUCE_MAPPED','STRATA_ATTN_DAILY','STRATA_POOL_FUSED']
TRACE=['STRATA_REMOTE_REDUCE_TRACE','STRATA_ATTN_DAILY_TRACE','STRATA_POOL_FUSED_TRACE']
KINDS={'base':[0,0,0],'helper':[1,0,0],'attn':[0,1,0],'pool':[0,0,1],'all':[1,1,1]}
order=['helper','attn','pool','all']; random.Random(20261019).shuffle(order)
ARMS=[('base_a','base')]+[(k+'_a',k) for k in order]+[(k+'_b',k) for k in reversed(order)]+[('base_b','base')]
META={'arms':ARMS,'candidate':CAND,'candidate_sha256':hashlib.sha256(pathlib.Path(CAND).read_bytes()).hexdigest(),'baseline':BASE_EXE,'seed':20261019,'params':{'temperature':0,'reasoning_effort':'medium','max_tokens':900},'note':'Screening: two clean processes per kind; not a thousand-request soak or full quality suite'}
(OUT/'manifest.json').write_text(json.dumps(META,indent=2))
def log(x): print(time.strftime('%H:%M:%S'),x,flush=True)
def rec(x):
    with (OUT/'results.jsonl').open('a') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n')
def req(path,payload=None,timeout=900):
    r=S.get(B+path,headers=H,timeout=timeout) if payload is None else S.post(B+path,headers=H,json=payload,timeout=timeout)
    r.raise_for_status(); return r.json()
def live():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            ex=str((p/'exe').resolve()); args=(p/'cmdline').read_bytes().split(b'\0')
            engine=ex in [BASE_EXE,CAND] and b'--serve' in args
            server=str(T/'serve/server.py').encode() in args and any(a==str(PROD).encode() or a.startswith(str(OUT).encode()+b'/cfg-') for a in args)
            if engine or server: found.append((int(p.name),ex,server))
        except OSError: pass
    return found
def stop():
    for pid,_,_ in live():
        try: os.kill(pid,signal.SIGINT)
        except ProcessLookupError: pass
    for _ in range(25):
        if not live(): break
        time.sleep(1)
    for pid,_,_ in live():
        try: os.kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
    time.sleep(2)
    assert not live(),'engine/server remains'
    for port in [8080,8081]:
        with socket.socket() as s: assert s.connect_ex(('127.0.0.1',port))!=0,str(port)+' occupied'
def start(arm,kind=None,trace=False):
    global B
    B='http://127.0.0.1:'+('8080' if kind is None else '8081')
    if kind is None:
        cmd=['bash',str(T/'run-nvfp4-gptq.sh')]; exe=BASE_EXE; elog=pathlib.Path(base['log'])
    else:
        c=json.loads(raw); c['exe']=CAND; c['log']=str(OUT/('engine-'+arm+'.log')); exe=CAND; elog=pathlib.Path(c['log'])
        e=dict(c.get('env') or {}); e.update({k:str(v) for k,v in zip(FLAGS,KINDS[kind])}); e.update({k:'1' if trace else '0' for k in TRACE})
        e.update({'STRATA_ATTN_LANECELL':'0','STRATA_ATTN_PRE75':'1','STRATA_ATTN_MERGE_V2':str(KINDS[kind][1])})
        c['env']=e; cfg=OUT/('cfg-'+arm+'.json'); cfg.write_text(json.dumps(c,ensure_ascii=False,indent=2)); cfg.chmod(0o600)
        cmd=['/root/strata/Strata-145/.venv/bin/python',str(T/'serve/server.py'),'--engine','strata','--config',str(cfg),'--port','8081']
    f=(OUT/('serve-'+arm+'.log')).open('w')
    proc=subprocess.Popen(cmd,cwd=T,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    t=time.monotonic()
    while time.monotonic()-t<420:
        if proc.poll() is not None: raise RuntimeError(arm+' server died '+str(proc.returncode))
        try:
            if req('/health',timeout=3).get('loaded'):
                ps=[(pid,ex) for pid,ex,srv in live() if not srv]
                assert len(ps)==1 and ps[0][1]==exe,ps
                log(arm+' ready '+str(round(time.monotonic()-t,1))+'s')
                rec({'arm':arm,'kind':'boot','exe':exe,'pid':ps[0][0],'seconds':time.monotonic()-t})
                return elog,ps[0][0]
        except requests.RequestException: pass
        time.sleep(2)
    raise RuntimeError(arm+' readiness timeout')
def parse(text):
    groups=[m.groups() for m in RE.finditer(text)]; assert groups,'Missing engine decode metrics'
    cold=max(groups,key=lambda g:int(g[2])); gen=sum(int(g[5]) for g in groups); ms=sum(int(g[6]) for g in groups); acc=sum(int(g[8]) for g in groups); drafts=sum(int(g[9]) for g in groups)
    waits=re.findall(r'host (\d+) ms staging\+launching, (\d+) ms waiting',text)
    return {'prompt':int(cold[0]),'reused':int(cold[1]),'read':int(cold[2]),'prefill_ms':int(cold[3]),'prefill_tps':float(cold[4]),'gen':gen,'gen_ms':ms,'accepted':acc,'drafted':drafts,'steps_est':gen-acc,'tps':1000*gen/max(1,ms),'c1_host_ms':sum(int(a) for a,b in waits),'c1_wait_ms':sum(int(b) for a,b in waits),'segments':len(groups)}
def chat(elog,messages,max_tokens=900,tools=None):
    off=elog.stat().st_size
    payload={'model':MODEL,'messages':messages,'temperature':0,'reasoning_effort':'medium','max_tokens':max_tokens,'uncensored':True}
    if tools: payload['tools']=tools
    j=req('/v1/chat/completions',payload); time.sleep(.15)
    with elog.open('rb') as f: f.seek(off); lines=f.read().decode(errors='replace')
    message=j['choices'][0]['message']; data=parse(lines)
    txt=(message.get('reasoning_content') or '')+'\n'+(message.get('content') or '')
    data.update({'sha256':hashlib.sha256(txt.encode()).hexdigest(),'finish':j['choices'][0]['finish_reason']})
    return message,data
PROMPTS={'zh_prose':'写一段约400字的散文，描写秋天傍晚的江边。','code_zh':'用Python实现一个线程安全的LRU缓存，带中文注释和测试。','math_zh':'数列满足a1=1，a(n+1)=2a(n)+n，求通项公式并验证前5项。','code_en':'Write a Python function that parses ISO 8601 durations like P3DT4H5M into seconds, with docstring and assertions.'}
rng=random.Random(4187); words='the quick brown fox jumps over the lazy dog while quantum mechanics perplexes researchers'.split()
hay='\n'.join(' '.join(rng.choice(words) for _ in range(400)) for _ in range(320))
def quality(elog,arm):
    m,d=chat(elog,[{'role':'user','content':'只回复标记 CHECK-7P4。'}],256); assert 'CHECK-7P4' in (m.get('content') or '')
    rec({'arm':arm,'kind':'quality','test':'text','pass':True})
    tools=[{'type':'function','function':{'name':'lookup_marker','description':'Get the hidden marker, do not guess.','parameters':{'type':'object','properties':{},'required':[]}}}]
    msgs=[{'role':'user','content':'先调用lookup_marker查询标记，然后原样返回工具结果中的marker。'}]
    m,d=chat(elog,msgs,600,tools); tc=m.get('tool_calls') or []; assert len(tc)==1 and tc[0]['function']['name']=='lookup_marker','tool call absent'
    assert json.loads(tc[0]['function']['arguments'])=={}
    msgs+=[m,{'role':'tool','tool_call_id':tc[0]['id'],'content':'{"marker":"OPT123-TOOL-8H2"}'}]
    m,d=chat(elog,msgs,600,tools); assert 'OPT123-TOOL-8H2' in (m.get('content') or ''),'tool followup absent'
    rec({'arm':arm,'kind':'quality','test':'tool_roundtrip','pass':True})
def summarize():
    rows=[json.loads(x) for x in (OUT/'results.jsonl').read_text().splitlines()]
    report={}
    for arm,kind in ARMS:
        rr=[r for r in rows if r.get('arm')==arm and r.get('kind')=='decode']; groups={}
        for cat in ['short','long']:
            rs=[r for r in rr if r['context']==cat]
            if rs:
                g=sum(r['gen'] for r in rs); ms=sum(r['gen_ms'] for r in rs); ac=sum(r['accepted'] for r in rs); dr=sum(r['drafted'] for r in rs)
                groups[cat]={'n':len(rs),'tokens':g,'ms':ms,'steps_est':sum(r['steps_est'] for r in rs),'tps':1000*g/ms,'ms_step_est':ms/max(1,g-ac),'accept':ac/max(1,dr),'c1_wait_ms':sum(r['c1_wait_ms'] for r in rs)}
        report[arm]={'kind':kind,'by_context':groups}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
def run_parity():
    gpu_lock=open('/root/strata/nvopt123-gpu-parity.lock','w'); fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    try:
        for dev in [0,1]:
            for daily in [0,1]:
                env=dict(os.environ); env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(dev),STRATA_ATTN_DAILY=str(daily),STRATA_ATTN_PRE75='1',STRATA_ATTN_MERGE_V2=str(daily),STRATA_ATTN_LANECELL='0',STRATA_ATTN_DAILY_TRACE='1')
                p=OUT/f'attn-gpu{dev}-daily{daily}.log'
                with p.open('w') as f:
                    r=subprocess.run(['/root/strata/Strata-nvopt-main/build/qsa_merge_daily_parity','--device','0'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
                text=p.read_text(errors='replace')
                assert r.returncode==0 and '96 captures, 288 replays with new inputs, 0 failed -> PASS' in text,(str(p),r.returncode,text[-2000:])
                rec({'kind':'parity','test':'attention','gpu':dev,'daily':daily,'summary':[l for l in text.splitlines() if 'parity:' in l]}); log(p.name+' PASS')
        helper=pathlib.Path('/root/strata/Strata-nvopt-helper')
        env=dict(os.environ); env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        with (OUT/'helper-parity.log').open('w') as f:
            r=subprocess.run(['python3','tests/remote_reduce/run.py','build-parity/remote_reduce_parity','--devices','0,1','--output',str(OUT/'helper-parity')],cwd=helper,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        assert r.returncode==0,'helper parity failed'
        rr=json.loads((OUT/'helper-parity/results.json').read_text())
        assert len(rr)==12 and all(x['passed'] for x in rr) and len({x['hash'] for x in rr})==1
        counts={k:sum(x[k] for x in rr) for k in ['cases','graph_checks','graph_skips']}
        assert counts=={'cases':5760,'graph_checks':576,'graph_skips':0},counts
        rec({'kind':'parity','test':'helper',**counts}); log('helper parity PASS '+str(counts))
    finally:
        fcntl.flock(gpu_lock,fcntl.LOCK_UN); gpu_lock.close()
def interrupt(sig,frame): raise RuntimeError('signal '+str(sig))
signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
status='running'; started=False
try:
    # Trace arm is not timed; all subsequent performance arms disable traces.
    state=req('/status',timeout=10); assert state.get('busy') is False and state.get('queued',0)==0,'production has active requests'
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1',8081))!=0,'test port occupied'
    log('WINDOW BEGIN; production will be restored to '+BASE_EXE); started=True; stop()
    run_parity()
    elog,pid=start('trace_all','all',True)
    for _ in range(2): chat(elog,[{'role':'user','content':PROMPTS['code_zh']}])
    txt=elog.read_text(errors='replace')
    trace_lines=[s for s in txt.splitlines() if any(k in s for k in ['attn-daily:','strata: remote reduce ','[STRATA_POOL_FUSED]'])]
    rec({'arm':'trace_all','kind':'trace','lines':trace_lines}); log('trace lines '+json.dumps(trace_lines))
    assert any('attn-daily:' in l and 'chunk=1' in l and 'merge_v2=1' in l for l in trace_lines),'attention trace missing'
    assert any('strata: remote reduce ' in l and 'enabled=1 zero_copy=1 path=mapped-host' in l for l in trace_lines),'helper mapped path trace missing'
    assert any('[STRATA_POOL_FUSED] completed native dispatch gu_type=40 d_type=40' in l for l in trace_lines),'CPU type40 fused path trace missing'
    for arm,kind in ARMS:
        stop(); elog,pid=start(arm,kind)
        chat(elog,[{'role':'user','content':'一句话介绍编程。'}],128)
        for rep in range(2):
            cats=list(PROMPTS); random.Random(18+rep).shuffle(cats)
            for cat in cats:
                msg,d=chat(elog,[{'role':'user','content':PROMPTS[cat]}])
                assert d['gen']>=64,'unrepresentative early stop'
                rec({'arm':arm,'kind':'decode','context':'short','cat':cat,'rep':rep,**d})
        # Unique first bytes per process force a cold long prefix; later questions reuse it.
        prefix='Document run '+arm+'.\n'+hay[:len(hay)//2]+'\nHidden marker: LONG-6F8-NEEDLE.\n'+hay[len(hay)//2:]+'\nQuestion: '
        for idx,cat in enumerate(['code_zh','zh_prose']):
            msg,d=chat(elog,[{'role':'user','content':prefix+PROMPTS[cat]}])
            assert d['prompt']>=100000,('long context too short',d['prompt'])
            if idx==0: assert d['read']>=100000,('long prefix not cold',d)
            rec({'arm':arm,'kind':'decode','context':'long','cat':cat,'rep':0,**d})
        msg,d=chat(elog,[{'role':'user','content':prefix+'文档中的Hidden marker是什么？只回复标记。'}],400)
        assert 'LONG-6F8-NEEDLE' in (msg.get('content') or ''),'needle failed'
        rec({'arm':arm,'kind':'quality','test':'long_needle','pass':True,'prompt':d['prompt']})
        quality(elog,arm)
        assert any(p==pid and ex==CAND for p,ex,srv in live() if not srv),'engine changed during arm'
        rates=[l for l in elog.read_text(errors='replace').splitlines() if 'PCIe share: the CPU pool' in l]
        rec({'arm':arm,'kind':'pcie_auto','lines':rates})
        summarize(); log(arm+' DONE')
    status='ok'
except BaseException as e:
    status='error: '+repr(e); log(status)
finally:
    restored=False
    if started:
        try:
            stop(); assert PROD.read_bytes()==raw,'production config changed externally; refusing overwrite'
            elog,pid=start('restore'); quality(elog,'restore'); restored=True
        except BaseException as e: log('RESTORE FAILURE '+repr(e)); status+='; restore error '+repr(e)
    try: summarize()
    except Exception as e: log('summary error '+repr(e))
    completion={'status':status,'restored':restored,'production_exe':base['exe'],'t':time.time()}
    (OUT/'completion.json').write_text(json.dumps(completion,indent=2)); log('WINDOW END '+json.dumps(completion))
