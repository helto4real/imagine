#!/usr/bin/env python3
"""Recovery after failed/unconfirmed GPU handoffs against disposable servers."""
import argparse, asyncio, json, os, pathlib, re, sys
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args()
sys.dont_write_bytecode=True;sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
from aiohttp import ClientSession
from tests.support import DynamicServer, MockComfy
from tests.strata_support import MockStrata
from st_proxy.config import BrokerConfig, TestEndpointRegistry
from st_proxy.service import BrokerService
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(kind,scenario):
 native=MockStrata();comfy=MockComfy();servers=[DynamicServer(native.app()),DynamicServer(comfy.app())]
 origins=[await s.start() for s in servers];service=process=drainer=None;result=[]
 settings=dict(poll_interval=.01,cleanup_timeout=10,reload_timeout=.2,unload_timeout=.2,image_timeout=.2,comfy_poll_failure_limit=3)
 try:
  if kind=='python':
   registry=TestEndpointRegistry();[registry.approve_disposable(u) for u in origins]
   config=BrokerConfig.for_test(llm_url=origins[0],comfy_url=origins[1],llm_backend='sc117',registry=registry,**settings)
   service=BrokerService(config);await service.start();chat=f'http://127.0.0.1:{service.chat_port}';image=f'http://127.0.0.1:{service.image_port}'
  else:
   args=['--chat-port','0','--image-port','0','--llm-backend','sc117','--llm-url',origins[0],'--comfy-url',origins[1]]
   for k,v in settings.items():args+=['--'+k.replace('_','-'),str(v)]
   process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),*args,env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')},stderr=asyncio.subprocess.PIPE)
   async with asyncio.timeout(10):
    while True:
     line=await process.stderr.readline();assert line,'startup failed';m=re.search(r'broker ready: chat=(\S+) image=(\S+)',line.decode())
     if m:chat,image=m.groups();break
   async def drain():
    async for _ in process.stderr:pass
   drainer=asyncio.create_task(drain())
  async with ClientSession() as client:
   async def call(base,path):
    async with client.post(base+path,json={}) as r:result.append([r.status,await r.json()])
   async def status():
    async with client.get(chat+'/broker/status') as r:v=await r.json()
    result.append({k:v[k] for k in ('state','healthy','gpu_owner','active_chats','waiting_chats','waiting_images','queued_workflow_bytes','chat_available')})
   if scenario.startswith('load'):
    if scenario=='load_http':native.fail_action='load'
    elif scenario=='load_false':native.false_confirmation=True
    else:native.confirm.clear()
    await call(chat,'/v1/chat/completions');await status();assert not native.payloads
    # Failure must not transfer ownership to Comfy until unload is confirmed.
    native.fail_action=None;native.false_confirmation=False;native.confirm.set()
    await call(chat,'/v1/chat/completions');await status()
   elif scenario=='unload':
    await call(chat,'/v1/chat/completions');native.fail_action='unload'
    await call(image,'/prompt');await status();assert not comfy.prompt_calls
    native.fail_action=None;await call(image,'/prompt');await asyncio.sleep(.1);await status()
    await call(chat,'/v1/chat/completions');await status()
   else:
    comfy.auto_complete=False;await call(image,'/prompt');await asyncio.sleep(.4);await status()
    await call(chat,'/v1/chat/completions');await status()
   result.append({'operations':native.operations,'prompts':comfy.prompt_calls,'free':comfy.free_calls,'loaded':native.loaded})
  return result
 finally:
  native.confirm.set();native.chat_release.set()
  if service:await service.stop()
  if process:process.terminate();await asyncio.wait_for(process.wait(),10);await drainer
  for server in servers:await server.close()
async def main():
 for scenario in ('load_http','load_false','load_timeout','unload','image_timeout'):
  async with asyncio.timeout(90):left=await run('python',scenario);right=await run('rust',scenario)
  errors=[{'case':i,'python':p,'rust':r} for i,(p,r) in enumerate(zip(left,right)) if p!=r]
  for error in errors:print(json.dumps(error),flush=True)
  assert len(left)==len(right) and not errors,scenario
  print(f'PASS: {scenario}: responses, ownership, recovery and lifecycle ordering',flush=True)
asyncio.run(main())
