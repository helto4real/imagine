#!/usr/bin/env python3
"""Raw URL/header and WebSocket protocol/close behavior using an echo backend."""
import argparse,asyncio,json,os,pathlib,re,sys
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args();sys.dont_write_bytecode=True;sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
from aiohttp import web,ClientSession
from yarl import URL
from tests.support import DynamicServer
from st_proxy.config import BrokerConfig,TestEndpointRegistry
from st_proxy.service import BrokerService
ROOT=pathlib.Path(__file__).resolve().parents[2]
async def run(kind):
 held=asyncio.Event();http_inflight=0;ws_opened=0
 async def echo(r):
  nonlocal http_inflight,ws_opened
  if r.path=='/hold':
   http_inflight+=1;await held.wait()
   return web.json_response({'ok':True})
  if r.path=='/ws':
   ws_opened+=1
   ws=web.WebSocketResponse(protocols=['synthetic']);await ws.prepare(r)
   async for msg in ws:
    if msg.type==web.WSMsgType.TEXT:
     if msg.data=='close':await ws.close(code=4100);break
     await ws.send_str(msg.data)
    elif msg.type==web.WSMsgType.BINARY:await ws.send_bytes(msg.data)
   return ws
  return web.json_response({'raw_path':r.raw_path,'origin':r.headers.get('Origin'),'referer':r.headers.get('Referer'),'x-test':r.headers.getall('X-Test',[]),'body':(await r.read()).decode()},headers={'Location':f'http://{r.host}/result%2Fone?a=%2B'})
 app=web.Application();app.router.add_route('*','/{tail:.*}',echo);server=DynamicServer(app);origin=await server.start();service=process=drainer=None;out=[]
 try:
  if kind=='python':
   registry=TestEndpointRegistry();registry.approve_disposable(origin)
   service=BrokerService(BrokerConfig.for_test(llm_backend='sc117',llm_url=origin,comfy_url=origin,registry=registry));await service.start();base=f'http://127.0.0.1:{service.image_port}'
  else:
   process=await asyncio.create_subprocess_exec(str(ROOT/'target/debug/st-proxy'),'--chat-port','0','--image-port','0','--llm-backend','sc117','--llm-url',origin,'--comfy-url',origin,env={k:v for k,v in os.environ.items() if not k.startswith('ST_PROXY_')},stderr=asyncio.subprocess.PIPE)
   async with asyncio.timeout(10):
    while True:
     line=await process.stderr.readline();assert line;m=re.search(r'broker ready: chat=(\S+) image=(\S+)',line.decode())
     if m:base=m[2];break
   async def drain():
    async for _ in process.stderr:pass
   drainer=asyncio.create_task(drain())
  async with ClientSession() as client:
   for method,path in [('GET','/custom/a%2Fb/?x=%2B&x=two+words'),('POST','/fr%65e'),('POST','/helto_director%2Fprompt_optimizer%2Foptimize'),('POST','/custom/test/')]:
    async with client.request(method,URL(base+path,encoded=True),data=b'synthetic',headers=[('Origin',base),('Referer',base+'/page?x=1'),('X-Test','one'),('X-Test','two')]) as r:
     result={'status':r.status,'location':r.headers.get('Location'),'body':await r.json()}
     out.append(json.loads(json.dumps(result).replace(origin,'<upstream>').replace(base,'<proxy>')))
   pending=[asyncio.create_task(client.get(base+'/hold')) for _ in range(65)]
   try:
    async with asyncio.timeout(3):
     while http_inflight<64:await asyncio.sleep(.01)
    await asyncio.sleep(.1);out.append({'http_connections':http_inflight});assert http_inflight==64
   finally:held.set()
   for response in await asyncio.gather(*pending):await response.read();response.close()
   async with client.ws_connect(base+'/ws',protocols=['synthetic']) as ws:
    out.append({'protocol':ws.protocol});await ws.send_bytes(b'\x00\xff');out.append({'binary':(await ws.receive()).data.hex()})
    async with client.get(base+'/broker/status') as r:out.append({'active_websockets':(await r.json())['active_websockets']})
    await ws.send_str('close');await ws.receive();out.append({'close_code':ws.close_code})
   baseline=ws_opened;sockets=[];waiting=None
   try:
    for _ in range(32):sockets.append(await client.ws_connect(base+'/ws',protocols=['synthetic']))
    waiting=asyncio.create_task(client.ws_connect(base+'/ws',protocols=['synthetic']))
    await asyncio.sleep(.1);out.append({'websocket_connections':ws_opened-baseline});assert ws_opened-baseline==32 and not waiting.done()
    await sockets.pop().close();sockets.append(await asyncio.wait_for(waiting,3))
   finally:
    for socket in sockets:await socket.close()
    if waiting and not waiting.done():waiting.cancel();await asyncio.gather(waiting,return_exceptions=True)

  return out
 finally:
  if service:await service.stop()
  if process:process.terminate();await asyncio.wait_for(process.wait(),10);await drainer
  await server.close()
async def main():
 left=await run('python');right=await run('rust')
 for i,(l,r) in enumerate(zip(left,right)):
  if l!=r:print(json.dumps({'case':i,'python':l,'rust':r}))
 assert left==right
 print(f'PASS: {len(left)} raw URL, encoded route policy, duplicate/header rewrite, WebSocket binary/subprotocol/count/close cases')
asyncio.run(main())
