#!/usr/bin/env python3
"""Differential HTTP/WS, lifecycle and disconnect tests on disposable fake servers."""
import argparse,asyncio,json,os,pathlib,re,sys,logging,difflib
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--logs',action='store_true');p.add_argument('--backends',nargs='+',default=['koboldcpp','tabbyapi','prism','sc117','orca','sc117_128k','orca_128k']);a=p.parse_args();sys.dont_write_bytecode=True
sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
from aiohttp import ClientSession
from tests.support import MockKobold,MockComfy,DynamicServer,wait_until,MockOllama
from tests.strata_support import MockStrata
from tests.tabby_support import MockTabby
from tests.test_prism_backend import MockPrism
from st_proxy.config import BrokerConfig,TestEndpointRegistry
from st_proxy.service import BrokerService
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(implementation,kind):
 llm=MockKobold() if kind=='koboldcpp' else MockTabby() if kind=='tabbyapi' else MockPrism() if kind=='prism' else MockOllama() if kind=='ollama' else MockStrata();comfy=MockComfy();servers=[DynamicServer(llm.app()),DynamicServer(comfy.app())];origins=[await s.start() for s in servers];process=None;service=None;logs=[]
 class Capture(logging.Handler):
  def emit(self,record):logs.append(f'{record.levelname} {record.name}: {record.getMessage()}')
 capture=Capture();logger=logging.getLogger('st_proxy');logger.setLevel(logging.DEBUG);logger.addHandler(capture)
 try:
  if implementation=='python':
   registry=TestEndpointRegistry();[registry.approve_disposable(u) for u in origins]
   config=BrokerConfig.for_test(llm_url=origins[0],comfy_url=origins[1],llm_backend=kind,registry=registry,poll_interval=.01,cleanup_timeout=10,**({"prism_model":llm.model} if kind=="prism" else {}))
   service=BrokerService(config);await service.start();chat=f'http://127.0.0.1:{service.chat_port}';image=f'http://127.0.0.1:{service.image_port}'
  else:
   env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')};env['NO_COLOR']='1'
   process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),'--chat-port','0','--image-port','0','--llm-backend',kind,'--llm-url',origins[0],'--comfy-url',origins[1],'--poll-interval','0.01','--cleanup-timeout','10','--unload-timeout','5','--reload-timeout','5','--log-level','DEBUG',*(['--prism-model',llm.model] if kind=='prism' else []),stderr=asyncio.subprocess.PIPE,env=env)
   async with asyncio.timeout(10):
    while True:
     line=await process.stderr.readline();assert line,'Rust exited before readiness';logs.append(line.decode());m=re.search(r'broker ready: chat=(\S+) image=(\S+)',line.decode())
     if m:chat,image=m.groups();break
   async def drain():
    async for line in process.stderr:logs.append(line.decode())
   drain_task=asyncio.create_task(drain())
  outcomes=[]
  async with ClientSession() as client:
   async def request(base,method,path,**kw):
    async with client.request(method,base+path,**kw) as r:
     raw=await r.read()
     try:body=json.loads(raw)
     except ValueError:body=raw.decode()
     value=[r.status,dict((k,r.headers[k]) for k in ('Content-Type','Retry-After') if k in r.headers),body];outcomes.append(value);return value
   await request(chat,'GET','/v1/models');assert not (llm.lifecycle_calls if kind=='ollama' else llm.admin_calls if kind=='koboldcpp' else (llm.loads if kind=='tabbyapi' else llm.operations))
   for method,path in [('GET','/custom/meta'),('POST','/upload/image'),('POST','/custom/new'),('POST','/free'),('POST','/helto_director/prompt_optimizer/optimize'),('POST','/interrupt')]:await request(image,method,path,data=b'synthetic')
   # Passive gateway forwarding preserves protocol and payloads.
   async with client.ws_connect(image+'/ws?clientId=synthetic',headers={'Origin':image}) as ws:
    await ws.send_str('synthetic');outcomes.append(['ws',str((await ws.receive()).data)])
   body={'model':'synthetic-model:latest','messages':[]} if kind=='ollama' else {'prompt':'synthetic'} if kind=='koboldcpp' else {'model':llm.model,'messages':[{'role':'user','content':'synthetic'}]}
   endpoint='/api/chat' if kind=='ollama' else '/api/v1/generate' if kind=='koboldcpp' else '/v1/chat/completions'
   await request(chat,'POST',endpoint,json=body)
   # The response to /prompt does not end the GPU reservation.
   comfy.auto_complete=False
   item=await request(image,'POST','/api/prompt',json={'prompt':{}});prompt_id=item[2]['prompt_id']
   queued=asyncio.create_task(request(chat,'POST',endpoint,json=body))
   await asyncio.sleep(.1);assert not queued.done()
   if kind!='ollama':await request(chat,'GET','/v1/models')
   comfy.completion[prompt_id].set();await queued
   if kind!='ollama':
    # Client disconnect: upstream must finish before an image takes the GPU.
    llm.hold_chat=True
    stream_path='/api/extra/generate/stream' if kind=='koboldcpp' else endpoint
    response=await client.post(chat+stream_path,json={**body,'stream':True});await response.content.readuntil(b'\n\n');response.close()
    pending=asyncio.create_task(request(image,'POST','/prompt',json={'prompt':{}}));await asyncio.sleep(.15);assert len(comfy.prompt_calls)==1
    llm.chat_release.set();await pending
    comfy.completion[comfy.prompt_calls[-1]].set()
   async with client.get(chat+'/broker/status') as r:status=await r.json()
   await asyncio.sleep(.1)
   outcomes.append({'backend_operations':llm.lifecycle_calls if kind=='ollama' else llm.admin_calls if kind=='koboldcpp' else {'loads':llm.loads,'unloads':llm.unloads} if kind=='tabbyapi' else llm.operations,'comfy_prompts':comfy.prompt_calls,'free_calls':comfy.free_calls})
 finally:
  if service:await service.stop()
  if process:
   process.terminate();await asyncio.wait_for(process.wait(),10);await drain_task
  for s in servers:await s.close()
  logger.removeHandler(capture)
 if a.logs:
  normalized=[]
  for line in logs:
   line=re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ','',line.strip())
   if 'st_proxy.cli:' in line:continue
   line=re.sub(r'(?<=duration=)[0-9.]+s','<duration>',line)
   for value,label in [(origins[0],'<llm>'),(origins[1],'<comfy>'),(chat,'<chat>'),(image,'<image>')]:line=line.replace(value,label)
   normalized.append(line)
  # Concurrent request completion and queue wakeup are independent tasks in
  # both runtimes. Compare exact order within each causal stream; keep every
  # event and the complete coordinator/lifecycle order.
  streams={}
  for line in normalized:
   if 'st_proxy.service: request ' in line:
    match=re.search(r'target=(.*?) method=(\S+) path=(\S+)',line)
    key='request:'+':'.join(match.groups()) if match else 'service'
   else:key=line.split(': ',1)[0].split(' ',1)[-1]
   streams.setdefault(key,[]).append(line)
  outcomes.append({'logs':streams})
 return outcomes
async def main():
 for kind in a.backends:
  python=await run('python',kind);rust=await run('rust',kind)
  if python!=rust:
   for index,(p,r)in enumerate(zip(python,rust)):
    if p!=r:
     if isinstance(p,dict) and 'logs' in p:
      for key in p['logs'].keys()|r['logs'].keys():
       if p['logs'].get(key)!=r['logs'].get(key):print(key+'\n'+''.join(difflib.unified_diff([x+'\n' for x in p['logs'].get(key,[])],[x+'\n' for x in r['logs'].get(key,[])],fromfile='Python',tofile='Rust')))
     else:print(json.dumps({'backend':kind,'case':index,'python':p,'rust':r}))
   raise AssertionError(f'{kind}: mismatch')
  print(f'PASS: {kind}: {len(rust)} matching HTTP/WS results and backend operation traces; 202 lifetime and disconnect drain')
asyncio.run(main())
