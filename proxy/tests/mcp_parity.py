#!/usr/bin/env python3
"""Exact MCP metadata/error and HTTP transport comparisons; synthetic inputs only."""
import argparse,asyncio,json,os,pathlib,re,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args();sys.dont_write_bytecode=True;sys.path.insert(0,str(a.reference/'src'))
from aiohttp import ClientSession
from st_proxy.config import BrokerConfig
from st_proxy.workflows import WorkflowCatalog
from st_proxy.media_jobs import MediaJobs
from st_proxy.mcp_server import McpListener
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(kind,root):
 process=None;listener=None;result=[]
 async with ClientSession() as client:
  try:
   if kind=='python':
    config=BrokerConfig(mcp_output_dir=str(root/'output'),mcp_input_dir=str(root/'input'),mcp_workflows_dir=str(ROOT/'proxy/workflows'))
    jobs=MediaJobs(config,WorkflowCatalog(config.mcp_workflows_dir,config.max_workflow_body_bytes),None,client);await jobs.start();listener=McpListener(jobs,0);await listener.start();url=f'http://127.0.0.1:{listener.port}/mcp'
   else:
    creds=root/'access.json';creds.write_text(json.dumps({'admin_token':'synthetic'}));env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')}
    process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),'--chat-port','0','--image-port','0','--mcp-port','0','--mcp-output-dir',str(root/'output'),'--mcp-input-dir',str(root/'input'),'--mcp-workflows-dir',str(ROOT/'proxy/workflows'),'--coordinator-socket',str(root/'gpu.sock'),'--imagine-access-file',str(creds),stderr=asyncio.subprocess.PIPE,env=env)
    async with asyncio.timeout(10):
     while True:
      line=await process.stderr.readline();assert line,'Rust startup failed';m=re.search(r'MCP ready: (\S+)',line.decode())
      if m:url=m[1];break
    async def drain():
     async for _ in process.stderr:pass
    drain_task=asyncio.create_task(drain())
   async def request(label,method='POST',headers=None,**kw):
    h={'Accept':'application/json, text/event-stream',**(headers or {})}
    async with client.request(method,url,headers=h,**kw) as r:
     if method=='GET' and r.status==200:
      async with asyncio.timeout(20):ping=await r.content.readline()
      assert ping.startswith(b': ping - ') and ping.endswith(b'+00:00\r\n'),ping
      text='<UTC keepalive>'
     else:text=await r.text()
     if method!='GET' and r.headers.get('content-type','').startswith('text/event-stream'):
      body=[json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
     else:
      try:body=json.loads(text)
      except ValueError:body=text
     result.append({'case':label,'status':r.status,'headers':{k:r.headers[k] for k in ('Content-Type','Allow','WWW-Authenticate') if k in r.headers},'body':body})
   async def rpc(method,params=None):await request(method+':'+json.dumps(params),json={'jsonrpc':'2.0','id':1,'method':method,**({'params':params} if params is not None else {})})
   await rpc('initialize',{'protocolVersion':'2025-03-26','capabilities':{},'clientInfo':{'name':'synthetic','version':'1'}})
   await rpc('initialize',{'protocolVersion':'1900-01-01','capabilities':{},'clientInfo':{'name':'synthetic','version':'1'}})
   await rpc('ping');await rpc('tools/list');await rpc('missing')
   for name,args in [('list_workflows',{}),('describe_workflow',{'workflow_id':'image_generation'}),('describe_workflow',{'workflow_id':'missing'}),('get_job',{'job_id':'unknown','wait':False}),('generate',{'job_id':'!','workflow_id':'image_generation','parameters':{}}),('generate',{'job_id':'test','workflow_id':'missing','parameters':{}}),('generate',{'job_id':'test','workflow_id':'image_generation','parameters':{'seed':True}}),('generate_batch',{'job_id':'test','workflow_id':'image_generation','items':[]}),('register_image',{'workflow_id':'qwen_scene','path':'relative.png'}),('missing',{}),('get_job',{}),('describe_workflow',{'workflow_id':2})]:await rpc('tools/call',{'name':name,'arguments':args})
   await request('notification',json={'jsonrpc':'2.0','method':'notifications/initialized'})
   await request('get',method='GET');await request('delete',method='DELETE');await request('put',method='PUT',json={})
   await request('bad-host',headers={'Host':'evil.example'},json={});await request('bad-origin',headers={'Origin':'https://evil.example'},json={})
   await request('accept',headers={'Accept':'application/json'},json={});await request('content-type',data='{}');await request('malformed',headers={'Content-Type':'application/json'},data='{')
   await request('bad-version',json={'jsonrpc':'1.0','id':1,'method':'ping'})
   for method,params in [('server/discover',{}),('ping',{}),('tools/list',{}),('tools/call',{'name':'list_workflows','arguments':{}}),('missing',{}),('initialize',{})]:
    params={**params,'_meta':{'io.modelcontextprotocol/protocolVersion':'2026-07-28','io.modelcontextprotocol/clientCapabilities':{}}}
    await request('modern:'+method,headers={'MCP-Protocol-Version':'2026-07-28','Mcp-Method':method,**({'Mcp-Name':params['name']} if 'name' in params else {})},json={'jsonrpc':'2.0','id':1,'method':method,'params':params})
   await request('modern-get',method='GET',headers={'MCP-Protocol-Version':'2026-07-28'})
   await request('modern-missing-meta',headers={'MCP-Protocol-Version':'2026-07-28'},json={'jsonrpc':'2.0','id':1,'method':'ping'})

  finally:
   if listener:await listener.close()
   if process:
    process.terminate();await asyncio.wait_for(process.wait(),10);await drain_task
 return result
async def main():
 with tempfile.TemporaryDirectory(prefix='st-proxy-mcp-parity-') as directory:
  root=pathlib.Path(directory);p=await run('python',root);r=await run('rust',root)
  errors=[]
  for left,right in zip(p,r):
   if left!=right:errors.append({'python':left,'rust':right})
  for error in errors:print(json.dumps(error,ensure_ascii=False))
  assert not errors,f'{len(errors)}/{len(p)} MCP differences'
  print(f'PASS: {len(p)} MCP metadata, validation, error and transport cases')
asyncio.run(main())
