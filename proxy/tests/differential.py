#!/usr/bin/env python3
"""Only synthetic data; Python is a test oracle, never a production dependency."""
import argparse, importlib.util, json, logging, os, pathlib, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.reference/'src'))
from st_proxy.comfy_routes import classify_comfy_route,comfy_upstream_path
from st_proxy.logging import CatppuccinFormatter,color_strata_line
from st_proxy.cli import build_parser,config_from_args
root=pathlib.Path(__file__).resolve().parents[2]
cases=[];expected=[]
def add(v,result):cases.append(v);expected.append(result)
policy=json.loads((root/'proxy/reference/comfy-routes.json').read_text())
paths=set(['/','/api/jobs/cancel','/api/jobs/abc/cancel','/api/jobs//cancel','/api/jobs/abc/cancel/','/history/delete','/unknown','/api/userdata/test','/helto_director/library/projects/a','/helto_director/library/characters/a/duplicate','/aio_image_generate/ideogram4_prompt_library/prompts/a/use'])
for key,value in policy.items():
 if isinstance(value,list):
  for item in value:
   if isinstance(item,str) and item.startswith('/'):paths.add(item)
   elif isinstance(item,list):paths.add(item[1])
for path in sorted(paths):
 for suffix in ('','/','/child'):
  for method in ('GET','HEAD','OPTIONS','POST','PUT','PATCH','DELETE','TRACE','post'):
   path2=path+suffix;add(dict(op='route',method=method,path=path2),dict(kind=classify_comfy_route(method,path2),path=comfy_upstream_path(path2)))
messages=['request completed: target=TabbyAPI method=POST path=/v1/chat/completions status=200 duration=1.234s','request rejected: target=Qwen SC117 status=503 reason=backend is unavailable duration=0.001s','state transition: llm_ready -> unloading_llm','state transition: reloading_llm -> error','chat request queued: sequence=1 waiting_chats=2','TabbyAPI reload failed: insufficient_vram','TabbyAPI load error: insufficient_vram; waiting for load stream to end','ComfyUI completed status=403 duration=0.100s','Prism completed status=302 duration=0.0s','Ollama failed: field=value','already \x1b[31mcolored\x1b[0m error=synthetic']
for msg in messages:
 for level in (10,20,30,40,50):
  for color in (False,True):
   rec=logging.LogRecord('st_proxy.service',level,'synthetic',1,msg,(),None)
   f=CatppuccinFormatter(color=color);f.formatTime=lambda *args: ('\x1b[38;2;127;132;156m2026-01-01 00:00:00,000\x1b[0m' if color else '2026-01-01 00:00:00,000')
   add(dict(op='log',timestamp='2026-01-01 00:00:00,000',level=level,name=rec.name,message=msg,color=color),f.format(rec))
for body in ['starting the engine','loading the experts','filling the GPU\'s expert cache (4486 experts, 7.29 GiB of VRAM) ...','experts loaded','almost ready ...','still starting (34 s) - please wait ...','the engine is running again','reading the prompt: 512 of 14,057 tokens, 2 s so far','thinking','answering: 9 of max 1500 tokens, 43.4 tok/s, 2 s','done: 38 tokens in 29 s (51.9 tok/s) (stop, cancel=False), expert cache 39.1% hit','done: 38 tokens in 29 s (stop, cancel=True)','done: 0 tokens in 1 s (error, cancel=False)','done (errors)','error: synthetic failure','\x1b[31merror\x1b[0m']:
 for ending in ('','\n','\r\n'):
  line='[strata] '+body+ending;add(dict(op='strata',line=line),color_strata_line(line))
for tty in (True,False):
 for nc in (None,'','0','1'):
  for term in (None,'xterm-256color','dumb'):
   add(dict(op='colors',tty=tty,no_color=nc,term=term),tty and not nc and term!='dumb')
from st_proxy.workflows import WorkflowCatalog,WorkflowError
catalog=WorkflowCatalog(str(a.reference/'workflows'),67108864)
for wid,w in catalog.workflows.items():
 add(dict(op='workflow',directory=str(root/'proxy/workflows'),id=wid),w.describe())
 for params in ({},{"prompt":"synthetic"},{"prompt":"synthetic","mode":"quick draft","nsfw":True},{"seed":9007199254740991},{"seed":9007199254740992},{"seed":True},{"max_side":257},{"prompt":""},{"prompt":17},{"unknown":1},{"nsfw":1},{"mode":"missing"}):
  try: value=w.normalize(params)
  except WorkflowError as e:value={"error":str(e)}
  add(dict(op='workflow',directory=str(root/'proxy/workflows'),id=wid,parameters=params),value)
import dataclasses,unittest.mock,io,contextlib,itertools
config_cases= [([],{}),(["--model-profiles",""],{})]
for kind in ('koboldcpp','ollama','prism','tabbyapi','all','sc117','orca','sc117_128k','orca_128k'):config_cases.append((["--llm-backend",kind],{}))
for key,value in [('KOBOLD_ROUTER_MODE','true'),('RESTORE_LLM_ON_IDLE','yes'),('STRICT_COMFY_ROUTES','1'),('ALLOW_UNKNOWN_COMFY_ROUTES','0'),('KOBOLD_URL','http://localhost:5002/base/'),('LLM_URL','http://localhost:5002/base/'),('MCP_TOKEN','synthetic')]:config_cases.append(([],{'ST_PROXY_'+key:value}))
for flag in ('chat-port','image-port','mcp-port','max-queued-images','tabby-max-seq-len','max-workflow-body-bytes','connect-timeout','request-timeout','poll-interval'):
 for value in ('-1','0','1','65536'):config_cases.append((["--"+flag,value],{}))
for args,env in config_cases:
 args=['--mcp-workflows-dir','/synthetic/workflows',*args]
 with unittest.mock.patch.dict(os.environ,env,clear=True),contextlib.redirect_stderr(io.StringIO()):
  try:
   value=dataclasses.asdict(config_from_args(build_parser().parse_args(args)));value.pop('test_mode');value.pop('test_registry');value.update(comfy_enabled=True,coordinator_socket=None,imagine_url='http://127.0.0.1:8080',imagine_access_file=None)
  except Exception as e:value={'error':str(e)}
 add(dict(op='config',args=args,env=env),value)
from st_proxy.llm.prism_diagnostics import PrismDiagnostics
for sse in (True,False):
 for complete,connected in itertools.product((True,False),repeat=2):
  event={'choices':[{'delta':{'reasoning_content':'synthetic å'},'message':{'reasoning_content':'synthetic å'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':20,'completion_tokens_details':{'reasoning_tokens':9}}}
  data=('data: '+json.dumps(event)+'\r\n\r\ndata: [DONE]\n\n') if sse else json.dumps(event)
  for cut in (1,17,len(data)):
   chunks=[data[i:i+cut] for i in range(0,len(data),cut)]
   body=json.dumps({'max_tokens':100,'reasoning_effort':'low','chat_template_kwargs':{'enable_thinking':True}})
   obs=PrismDiagnostics(body.encode());ctype='text/event-stream' if sse else 'application/json';obs.start(200,ctype)
   for chunk in chunks:obs.feed(chunk.encode())
   with unittest.mock.patch('st_proxy.llm.prism_diagnostics.LOG') as logger:
    obs.finish(upstream_complete=complete,client_connected=connected);fmt,*fields=logger.info.call_args.args;expected_line=fmt%tuple(fields)
   add(dict(op='prism',body=body,status=200,content_type=ctype,chunks=chunks,complete=complete,connected=connected),expected_line)
from st_proxy.llm.tabbyapi import adapt_chat_request
from st_proxy.model_profiles import ModelProfiles
for effort in ('minimal','low','medium','high','none','off',' LOW ',None,7):
 for extra in ({},{'template_vars':None},{'enable_thinking':False},{'reasoning':None},{'thinking_budget_tokens':10},{'reasoning_budget_tokens':None},{'reasoning_budget_tokens':-1},{'reasoning':{'max_tokens':0}},{'template_vars':{'enable_thinking':None}}):
  for max_tokens in (101,'101',101.0,0,-1,None):
   payload={'messages':[{'content':'synthetic å😀\x7f'}],'reasoning_effort':effort,'max_tokens':max_tokens,**extra};body=json.dumps(payload,ensure_ascii=False,indent=1)
   add(dict(op='tabby',body=body),adapt_chat_request(body.encode()).decode())
for engine in ('tabbyapi','prism','sc117','orca','sc117_128k','orca_128k'):
 profile={'id':'synthetic','models':[engine+'/synthetic'],'defaults':{'temperature':.8,'repetition_penalty':1.1,'reasoning_budget_tokens':70},'thinking':{'temperature':.6},'non_thinking':{'temperature':1.0}}
 profiles=ModelProfiles();profiles._add(profile,set())
 for control in ({},{'rep_pen':None},{'repeat_penalty':1.2},{'temperature':None},{'thinking':{}},{'reasoning_effort':'low'},{'reasoning_effort':'none'},{'reasoning_effort':'off'},{'reasoning_effort':'xhigh'},{'enable_thinking':True},{'enable_thinking':False},{'reasoning':None},{'reasoning':{'effort':'low'}},{'chat_template_kwargs':{'enable_thinking':False}},{'chat_template_kwargs':{'enable_thinking':True}},{'template_vars':None},{'template_vars':{'enable_thinking':True},'chat_template_kwargs':False}):
  add(dict(op='profiles',profiles={'version':1,'profiles':[profile]},engine=engine,model='synthetic',payload=control),profiles.apply(control,engine,'synthetic'))
result=subprocess.run([str(root/'target/debug/examples/oracle')],input=''.join(json.dumps(c)+'\n' for c in cases),text=True,capture_output=True,check=True)
actual=[json.loads(line) for line in result.stdout.splitlines()];assert len(actual)==len(expected)
errors=[(i,c,e,r) for i,(c,e,r) in enumerate(zip(cases,expected,actual)) if e!=r]
for i,c,e,r in errors[:10]:print(json.dumps(dict(case=i,input=c,expected=e,actual=r),ensure_ascii=False))
assert not errors,f'{len(errors)} differential mismatches'
print(f'PASS: {len(cases)} Python/Rust route, log and color comparisons (exact bytes, no normalization)')
