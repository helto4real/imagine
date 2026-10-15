#!/usr/bin/env python3
"""Opt-in smoke of an explicitly selected running stack; exercises real GPU ownership.
Uses a synthetic chat/prompt and the caller's explicitly supplied synthetic Comfy graph.
Does not start, stop or replace services. The broker may load/unload their models.
"""
import argparse, asyncio, json, pathlib, time
from aiohttp import ClientSession, ClientTimeout
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',action='store_true');p.add_argument('--chat-url',required=True);p.add_argument('--image-url',required=True)
p.add_argument('--imagine-url',required=True);p.add_argument('--access-file',type=pathlib.Path,required=True)
p.add_argument('--model',required=True,help='Exact selected model ID (backend/model in all mode)')
p.add_argument('--backend',choices=['koboldcpp','tabbyapi','ollama','prism','sc117','orca','sc117_128k','orca_128k','all'],required=True)
p.add_argument('--comfy-workflow',type=pathlib.Path,required=True,help='Explicit synthetic API-format graph')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
if not a.run:p.exit(message='Pass --run after reserving the GPU and these services for this smoke test.\n')
async def main():
 auth={'Authorization':'Bearer '+json.loads(a.access_file.read_text())['admin_token']}
 evidence={'backend':a.backend,'model':a.model,'real_services':True,'checks':[]}
 async with ClientSession(timeout=ClientTimeout(total=1800)) as client:
  async def request(base,path,payload=None,headers=None):
   async with client.request('POST' if payload is not None else 'GET',base+path,json=payload,headers=headers) as r:
    body=await r.json();assert r.status<400,(path,r.status,body);return body
  async def chat():
   if a.backend=='ollama':path='/api/chat';body={'model':a.model,'messages':[{'role':'user','content':'Reply with only the word blue.'}],'stream':False,'options':{'num_predict':8}}
   else:path='/v1/chat/completions';body={'model':a.model,'messages':[{'role':'user','content':'Reply with only the word blue.'}],'max_tokens':8,'stream':False}
   await request(a.chat_url,path,body);evidence['checks'].append('chat completed')
  await chat()
  direct=await request(a.imagine_url,'/v1/jobs',{'model':'qwen-image-2.1-int8','operation':'text_to_image','prompt':'A blue ceramic teapot on a white table, studio photograph.','seed':42},auth)
  # Submit Comfy and LLM while direct Imagine work is pending or active.
  image_task=asyncio.create_task(request(a.image_url,'/prompt',{'prompt':json.loads(a.comfy_workflow.read_text())}))
  chat_task=asyncio.create_task(chat())
  prompt=await image_task
  started=time.monotonic()
  while True:
   job=await request(a.imagine_url,'/v1/jobs/'+direct['id'],headers=auth)
   history=await request(a.image_url,'/history/'+str(prompt['prompt_id']))
   entry=history.get(str(prompt['prompt_id']),{});status=entry.get('status',{})
   if job['status'] in ('failed','cancelled'):raise AssertionError(job)
   if status.get('status_str') in ('error','failed','cancelled'):raise AssertionError(entry)
   if job['status']=='succeeded' and status.get('completed'):break
   assert time.monotonic()-started<1800,'generation deadline'
   await asyncio.sleep(.5)
  await chat_task
  evidence['checks']+=['direct Imagine completed','Comfy workflow completed','queued chat restored backend']
  evidence['broker_status']=await request(a.chat_url,'/broker/status')
 a.output.write_text(json.dumps(evidence,indent=2)+'\n');print('PASS:',a.output)
asyncio.run(main())
