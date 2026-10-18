#!/usr/bin/env python3
"""Opt-in trained HTTP/queue/worker check in a NEW isolated project only."""
import argparse,hashlib,io,json,os,shutil,socket,subprocess,sys,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['own-root','model-root','manifest','binary','site','control']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--expected-image',type=Path)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
control=json.loads(a.control.read_text());a.own_root.mkdir(parents=True,exist_ok=False)
model='moss-image-sft-int8'if control['variant']=='int8_convrot'and 'moss' in control['checkpoints']['dit']else 'qwen-image-2.1-turbo'if control['variant']=='turbo_bf16'else'qwen-image-2.1-int8'
modelroot=a.own_root/'models'
for relative in list(control['checkpoints'].values())+[s['path']for s in control.get('lora',[])]:
 path=modelroot/relative;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(a.model_root/relative)
profiles=a.own_root/'profiles';profiles.mkdir();shutil.copyfile(project/'profiles/moss.yaml',profiles/'moss.yaml')
with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
base=f'http://127.0.0.1:{port}'
def request(path,value=None,raw=None):
 data=raw if raw is not None else json.dumps(value).encode()if value is not None else None
 headers={'Content-Type':'image/png'if raw is not None else'application/json'}
 with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=15)as response:
  payload=response.read();return json.loads(payload)if 'json'in response.headers.get('Content-Type','')else payload
# Only the explicitly selected site is used for test image codecs; no Torch import.
sys.path.insert(0,str(a.site));from PIL import Image
refs=[]
for spec in control.get('references',[]):
 _,h,w,c=spec['shape'];assert c==4
 pixels=bytes((i*spec['byte_multiplier']+spec['byte_offset'])%256 for i in range(h*w*c))
 image=io.BytesIO();Image.frombytes('RGBA',(w,h),pixels).save(image,format='PNG');refs.append(image.getvalue())
options=dict(diffusion_model=control['checkpoints']['dit'],text_encoder=control['checkpoints']['text'],vae=control['checkpoints']['vae'],loras=[dict(path=s['path'].removeprefix('loras/'),strength=s['strength_model'])for s in control.get('lora',[])],zero_negative_conditioning=control.get('zero_negative_conditioning',False))
body=dict(model=model,operation='image_edit'if refs else'text_to_image',prompt=control['synthetic_prompt'],negative_prompt=control.get('negative_prompt',''),width=control['width'],height=control['height'],seed=control['seed'],sampling=dict(steps=control['steps'],cfg_scale=control['cfg'],sampler=control['sampler'],scheduler=control['scheduler'],custom_sigmas=control.get('custom_sigmas',[])),model_options={'moss_image':options}if model=='moss-image-sft-int8'else{'qwen_image_2_1':{'generation':options}},inputs={'reference_images':[]})
env=dict(os.environ,QWEN_MODEL_ROOT=str(modelroot));env.pop('IMAGINE_COORDINATOR_SOCKET',None)
with(a.own_root/'server.log').open('w')as log:
 pid=None
 server=subprocess.Popen([str(a.binary),'--project',str(a.own_root),'--data',str(a.own_root/'data'),'--rust-runtime',str(a.manifest),'--bind',f'127.0.0.1:{port}'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
 try:
  deadline=time.monotonic()+60
  while True:
   assert server.poll()is None,'Isolated server exited'
   try:startup=request('/v1/runtime');break
   except OSError:
    if time.monotonic()>deadline:raise
    time.sleep(.1)
  assert startup['engine']['backend']=='rust'and startup['engine']['pid']is None
  catalog=request('/v1/model-library');selected=next(m for m in catalog['models']if m['model']==model);assert selected['available'],selected.get('problem')
  for image in refs:body['inputs']['reference_images'].append(request('/v1/assets',raw=image)['id'])
  # Exercise actual discovery/component selection, not just explicit file paths.
  selected_options=body.pop('model_options');body['model_selection']={'diffusion':selected['id'],'loras':[]}
  for spec in control.get('lora',[]):
   file=next(f for f in catalog['files']if f['path']==spec['path']);body['model_selection']['loras'].append({'id':file['id'],'strength':spec['strength_model']})
  request('/v1/jobs/validate',body)
  start=time.monotonic();job=request('/v1/jobs',body);deadline=start+1800
  while True:
   job=request('/v1/jobs/'+job['id'])
   if job['status']in ['succeeded','failed','cancelled']:break
   assert time.monotonic()<deadline,'Isolated job timeout';time.sleep(.2)
  assert job['status']=='succeeded',job.get('error')
  wall=time.monotonic()-start;image=request('/v1/assets/'+job['output']['assets'][0]['id']);(a.own_root/'image.png').write_bytes(image)
  runtime=request('/v1/runtime');pid=runtime['engine']['pid'];maps=Path(f'/proc/{pid}/maps').read_text();assert 'libpython'not in maps
  pixels=Image.open(io.BytesIO(image)).convert('RGBA').tobytes()
  report=dict(schema=1,backend=runtime['engine']['backend'],model=model,worker_binary_sha256=json.loads(a.manifest.read_text())['files'][json.loads(a.manifest.read_text())['binary']],control_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),server_binary_sha256=hashlib.file_digest(a.binary.open('rb'),'sha256').hexdigest(),wall_seconds=wall,startup_did_not_launch_worker=True,worker_maps_contain_python=False,pixel_sha256=hashlib.sha256(pixels).hexdigest(),width=control['width'],height=control['height'],library_model_selectable=True,progress=runtime['engine'].get('progress'))
  if a.expected_image:
   expected=Image.open(a.expected_image).convert('RGBA');actual=Image.open(io.BytesIO(image)).convert('RGBA');report['pixels_exact']=expected.size==actual.size and expected.tobytes()==actual.tobytes()
  (a.own_root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
  if a.expected_image:assert report['pixels_exact'],'Pixel mismatch against explicit original control'
 finally:
  server.terminate()
  try:
   assert server.wait(timeout=30)==0
   if pid is not None:assert not Path(f'/proc/{pid}').exists(),'Worker survived server shutdown'
   assert not any(p.is_file()for p in (a.own_root/'data/media').rglob('*')),'Session media survived shutdown'
  except subprocess.TimeoutExpired:server.kill();server.wait();raise
