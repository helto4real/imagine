#!/usr/bin/env python3
"""Multi-backend differential test using the reference's GPU ownership guards."""
import argparse,asyncio,json,os,pathlib,re,sys
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args();sys.dont_write_bytecode=True
sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
from tests.test_multi_backend import MultiBackendTests
from aiohttp import ClientSession
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(kind):
 fixture=MultiBackendTests();await fixture.asyncSetUp();process=None;result=[]
 try:
  if kind=='rust':
   await fixture.service.stop();c=fixture.config;env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')};args=['--chat-port','0','--image-port','0','--llm-backend','all']
   for field in ('llm_url','tabby_url','prism_url','sc117_url','orca_url','sc117_128k_url','orca_128k_url','comfy_url','prism_model','poll_interval'):args+=['--'+field.replace('_','-'),str(getattr(c,field))]
   process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),*args,stderr=asyncio.subprocess.PIPE,env=env)
   async with asyncio.timeout(10):
    while True:
     line=await process.stderr.readline();assert line,line;m=re.search(r'broker ready: chat=(\S+) image=(\S+)',line.decode())
     if m:fixture.url,fixture.image_url=m.groups();break
   async def drain():
    async for _ in process.stderr:pass
   drainer=asyncio.create_task(drain())
  async def request(method,path,**kw):
   async with fixture.client.request(method,fixture.url+path,**kw) as r:
    text=await r.text()
    try:value=json.loads(text)
    except ValueError:value=text
    result.append([r.status,r.headers.get('content-type'),value]);return value
  await request('GET','/v1/models');assert not any(method=='POST' for _,method,_ in fixture.events)
  for model in (None,17,'missing','unknown/model','sc117/missing'):
   await request('POST','/v1/chat/completions',json={'model':model,'messages':[]})
  fixture.prism.hold_chat=True
  stream=await fixture.client.post(fixture.url+'/v1/chat/completions',json={'model':fixture.models['prism'],'stream':True})
  await stream.content.readuntil(b'\n\n')
  async with ClientSession() as other:
   queued=asyncio.create_task(other.post(fixture.url+'/v1/chat/completions',json={'model':fixture.models['tabbyapi']}))
   await asyncio.sleep(.2);queued.cancel();await asyncio.gather(queued,return_exceptions=True)
  await asyncio.sleep(.2);fixture.prism.hold_chat=False;fixture.prism.chat_release.set();await stream.read();stream.close()
  await request('POST','/v1/chat/completions',json={'model':fixture.models['prism']})
  assert not fixture.tabby.loads,'disconnected queued request switched models'
  for engine,model in fixture.models.items():
   await request('POST','/v1/chat/completions',json={'model':model,'messages':[{'role':'user','content':'synthetic å😀'}],'reasoning_effort':'low','max_tokens':101})
   # Native metadata never loads an additional model.
   await request('GET','/v1/models')
  result.append(['image',await fixture.image()])
  for engine in ('sc117','tabbyapi','koboldcpp','prism'):
   await request('POST','/v1/chat/completions',json={'model':fixture.models[engine],'messages':[]})
  for path in ('/models/load','/api/admin/reload_config','/v1/model/load'):
   await request('POST',path,json={})
  # Poll frequency depends on runtime scheduling; lifecycle POSTs and their order do not.
  result.append({'lifecycle_posts':[x for x in fixture.events if x[1]=='POST']})
  result.append({'tabby_wire_bodies':[b.decode() for b in fixture.tabby.payloads], 'prism_wire_bodies':[b.decode() for b in fixture.prism.chat_payloads]})
  return result
 finally:
  if process:process.terminate();await asyncio.wait_for(process.wait(),10);await drainer
  await fixture.asyncTearDown()
async def main():
 async with asyncio.timeout(180):
  python=await run('python');rust=await run('rust');errors=[{'case':i,'python':p,'rust':r} for i,(p,r) in enumerate(zip(python,rust)) if p!=r]
  for error in errors:print(json.dumps(error,ensure_ascii=False))
  assert len(python)==len(rust) and not errors,f'{len(errors)} multi-backend differences'
  print(f'PASS: {len(python)} multi-backend results and lifecycle traces; all seven adapters guarded against overlapping GPU ownership')
asyncio.run(main())
