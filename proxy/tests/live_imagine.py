#!/usr/bin/env python3
"""Opt-in real Imagine GPU test, with disposable fake LLM/Comfy control servers.
No active service is replaced. Only synthetic prompts and this run's own assets.
Build release binaries first. Real external backend smoke is a separate test.
"""
import argparse,asyncio,json,os,pathlib,shutil,socket,sys,tempfile,time,yaml
from aiohttp import web,ClientSession,ClientTimeout
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--cases',nargs='+',choices=['standard','quick-draft','lora','upscale'],default=['standard','quick-draft','lora','upscale']);p.add_argument('--reference-image',type=pathlib.Path);a=p.parse_args()
if not a.run:p.exit(message='Pass --run to use the GPU.\n')
ROOT=pathlib.Path(__file__).resolve().parents[2]
def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
async def main():
 (ROOT/'.imagine/test-runs').mkdir(parents=True,exist_ok=True)
 out=pathlib.Path(tempfile.mkdtemp(prefix='profile-mcp-live-',dir=ROOT/'.imagine/test-runs'));profiles=out/'profiles';shutil.copytree(ROOT/'profiles',profiles)
 for path in profiles.glob('*.yaml'):
  profile=yaml.safe_load(path.read_text())
  profile['defaults']['prompt']='A blue ceramic teapot on a wooden table, studio photograph.'
  profile['defaults']['negative_prompt']=''
  path.write_text(yaml.safe_dump(profile,sort_keys=False))
 app=web.Application();app.router.add_get('/health',lambda r:web.json_response({'service':'strata','status':'ok','loaded':False,'model':'synthetic'}));app.router.add_get('/v1/models',lambda r:web.json_response({'data':[{'id':'synthetic'}]}));app.router.add_post('/free',lambda r:web.json_response({}))
 runner=web.AppRunner(app);await runner.setup();fakeport=port();await web.TCPSite(runner,'127.0.0.1',fakeport).start()
 ip,cp,pp,mp=[port() for _ in range(4)];base=f'http://127.0.0.1:{ip}';mcp=f'http://127.0.0.1:{mp}/mcp';sock=out/'gpu.sock';data=out/'data';env={k:v for k,v in os.environ.items() if not k.startswith(('ST_PROXY_','IMAGINE_COORDINATOR_'))};env['IMAGINE_COORDINATOR_SOCKET']=str(sock)
 log=(out/'server.log').open('wb');proxylog=(out/'proxy.log').open('wb');processes=[];evidence={'real_gpu':'Imagine','fake_backends':['sc117','ComfyUI'],'cases':[]}
 try:
  proc=await asyncio.create_subprocess_exec(str(ROOT/'target/release/imagine-server'),'--project',str(ROOT),'--profiles',str(profiles),'--data',str(data),'--bind',f'127.0.0.1:{ip}',env=env,stdout=log,stderr=log);processes.append(proc)
  async with ClientSession(timeout=ClientTimeout(total=900)) as client:
   for _ in range(200):
    try:
     async with client.get(base+'/v1/session') as r:
      if r.status==200:break
    except OSError:pass
    await asyncio.sleep(.1)
   else:raise RuntimeError('Imagine did not start')
   proxy=await asyncio.create_subprocess_exec(str(ROOT/'target/release/st-proxy'),'--chat-port',str(cp),'--image-port',str(pp),'--llm-backend','sc117','--llm-url',f'http://127.0.0.1:{fakeport}','--comfy-url',f'http://127.0.0.1:{fakeport}','--coordinator-socket',str(sock),'--imagine-url',base,'--mcp-port',str(mp),'--mcp-input-dir',str(out),'--mcp-output-dir',str(out/'output'),'--poll-interval','0.1',env=env,stdout=proxylog,stderr=proxylog);processes.append(proxy)
   for _ in range(200):
    try:
     async with client.get(f'http://127.0.0.1:{cp}/broker/status') as r:
      if r.status==200:break
    except OSError:pass
    await asyncio.sleep(.1)
   else:raise RuntimeError('proxy did not start')
   async def call(name,args):
    async with client.post(mcp,headers={'Accept':'application/json, text/event-stream'},json={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':args}}) as r:
     text=await r.text();assert r.status==200,(r.status,text);event=json.loads(next(line[6:] for line in text.splitlines() if line.startswith('data: ')));return event['result']['structuredContent']
   references=[]
   if a.reference_image:
    shutil.copyfile(a.reference_image,out/'selected-reference.png');references.append(str(out/'selected-reference.png'))
   for label,profile,params in [('standard','qwen',{'prompt':'A blue ceramic teapot on a wooden table, studio photograph.','width':256,'height':256,'seed':42}),('quick-draft','quick-draft',{'prompt':'A blue ceramic teapot on a wooden table, studio photograph.','width':256,'height':256,'seed':9007199254740991}),('lora','quick-draft',{'prompt':'A blue ceramic teapot on a wooden table, studio photograph.','width':256,'height':256,'seed':42,'nsfw':True}),('upscale','image_upscale',{'max_side':512,'seed':42})]:
    if label not in a.cases:continue
    args={'job_id':label,'profile':profile,'parameters':params}
    if profile=='image_upscale':
     registered=await call('register_image',{'profile':profile,'path':references[0]});assert 'image_id' in registered,registered;args['references']=[registered['image_id']]
    started=time.monotonic();result=await call('generate',args);record={'case':label,'seconds':round(time.monotonic()-started,3),'result':result};evidence['cases'].append(record);(out/'evidence.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(record),flush=True)
    assert result['status']=='succeeded',result
    assert result['parameters']['resolved_profile']['id']==profile
    assert 'prompt' not in result['parameters'] and 'negative_prompt' not in result['parameters']
    references.append(result['images'][0]['path']);assert await call('generate',args)==result
   async with client.post(base+'/v1/runtime/yield') as r:assert (await r.json())['released']
 finally:
  for proc in reversed(processes):
   if proc.returncode is None:proc.terminate()
   try:await asyncio.wait_for(proc.wait(),30)
   except asyncio.TimeoutError:proc.kill();await proc.wait()
  await runner.cleanup();log.close();proxylog.close();(out/'evidence.json').write_text(json.dumps(evidence,indent=2));print('evidence:',out,flush=True)
asyncio.run(main())
