#!/usr/bin/env python3
"""Tabby VRAM retries and Kobold router failure recovery, disposable fixtures only."""
import argparse,asyncio,json,os,pathlib,re,sys
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--backends',nargs='+',default=['tabbyapi','koboldcpp'],choices=['tabbyapi','koboldcpp']);a=p.parse_args();sys.dont_write_bytecode=True;sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
from tests.test_tabbyapi_backend import TabbyTestCase
from tests.test_kobold_router import RouterBrokerTests
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(kind,backend):
 fixture=TabbyTestCase() if backend=='tabbyapi' else RouterBrokerTests();await fixture.asyncSetUp();process=drainer=None;out=[]
 chat=f'http://127.0.0.1:{fixture.service.chat_port}';image=f'http://127.0.0.1:{fixture.service.image_port}'
 try:
  if kind=='rust':
   await fixture.service.stop();c=fixture.config;args=['--chat-port','0','--image-port','0']
   for field in ('llm_backend','llm_url','comfy_url','tabby_model','kobold_model_cache','connect_timeout','request_timeout','image_timeout','chat_drain_timeout','unload_timeout','reload_timeout','cleanup_timeout','poll_interval','comfy_poll_failure_limit'):
    if getattr(c,field) is not None:args+=['--'+field.replace('_','-'),str(getattr(c,field))]
   if c.kobold_router_mode:args+=['--kobold-router-mode']
   process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),*args,env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')},stderr=asyncio.subprocess.PIPE)
   async with asyncio.timeout(10):
    while True:
     line=await process.stderr.readline();assert line;m=re.search(r'broker ready: chat=(\S+) image=(\S+)',line.decode())
     if m:chat,image=m.groups();break
   async def drain():
    async for _ in process.stderr:pass
   drainer=asyncio.create_task(drain())
  async def call(base,path,body):
   async with fixture.client.post(base+path,json=body) as r:out.append([r.status,await r.json()])
  async def status():
   async with fixture.client.get(chat+'/broker/status') as r:v=await r.json()
   out.append({k:v[k] for k in ('state','healthy','gpu_owner','active_chats','waiting_chats','waiting_images','last_error')})
  if backend=='tabbyapi':
   await call(image,'/prompt',{'prompt':{}});await asyncio.sleep(.1)
   fixture.tabby.load_error=True;fixture.tabby.load_error_message='CUDA out of memory. synthetic-private-details'
   for _ in range(2):
    await call(chat,'/v1/chat/completions',{});await status()
    count=len(fixture.tabby.loads)
    async with fixture.client.get(chat+'/health') as r:assert r.status==200;await r.read()
    assert len(fixture.tabby.loads)==count,'passive metadata retried a load'
   fixture.tabby.load_error=False
   await call(chat,'/v1/chat/completions',{});await status()
   out.append({'loads':fixture.tabby.loads,'unloads':fixture.tabby.unloads})
  else:
   fixture.router.fail_load=True
   await call(chat,'/api/v1/generate',{'model':'b.kcpps','prompt':'synthetic'});await asyncio.sleep(.05);await status()
   fixture.router.unload_failures=1
   await call(image,'/prompt',{'prompt':{}});assert not fixture.comfy.prompt_calls
   fixture.router.fail_load=False
   await call(chat,'/api/v1/generate',{'model':'b.kcpps','prompt':'synthetic'})
   await call(image,'/prompt',{'prompt':{}});await asyncio.sleep(.1);await status()
   out.append({'admin':fixture.router.admin_calls,'selected':fixture.router.selected})
  return out
 finally:
  if process:process.terminate();await asyncio.wait_for(process.wait(),10);await drainer
  await fixture.asyncTearDown()
async def main():
 for backend in a.backends:
  async with asyncio.timeout(120):left=await run('python',backend);right=await run('rust',backend)
  for i,(l,r) in enumerate(zip(left,right)):
   if l!=r:print(json.dumps({'case':i,'python':l,'rust':r}),flush=True)
  assert left==right,backend
  print('PASS:',backend,'request-driven recovery, metadata isolation, ownership and lifecycle trace',flush=True)
asyncio.run(main())
