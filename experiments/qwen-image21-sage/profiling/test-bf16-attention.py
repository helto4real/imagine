"""Synthetic BF16 chain: fused vs unfused and direct vs expanded Sage inputs."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');p.add_argument('--sanitizer',choices=['memcheck','racecheck']);a=p.parse_args()
here=Path(__file__).resolve().parent;root=here.parent
out=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-bf16-attention-check-'),dir=root/'runs'));print(out,flush=True)
binary=root/'runs/bf16-attention-build/check-bf16-attention';shutil.copy2(binary,out/'check-bf16-attention')
for name in ['check-bf16-attention.cpp','test-bf16-attention.py']:shutil.copyfile(here/name,out/name)
base={**os.environ,'SD_QWEN21_BF16_ATTENTION':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_FUSED_RMS_ROPE':'1','GGML_CUDA_CONVROT_THREADS':'512'}
for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','GGML_CUDA_DISABLE_FUSION']:base.pop(k,None)
if a.small:base['BF16_TEST_SMALL']='1'
results=[];hashes={}
with tempfile.TemporaryDirectory(dir=out,prefix='synthetic-') as temp:
 inp=Path(temp);rng=np.random.default_rng(42);n=17 if a.small else 5600
 def write(name,data):data.tofile(inp/name);hashes[name]=hashlib.sha256(data.tobytes()).hexdigest()
 write('x.f32',rng.integers(-125,126,(n,4096),dtype=np.int16).astype(np.float32)/64)
 for i in range(4):write(f'w{i}.i8',rng.integers(-127,128,(4096,4096),dtype=np.int8))
 write('scale.f32',np.full(4096,1/4096,dtype=np.float32))
 write('norm.f32',rng.integers(50,150,128,dtype=np.int16).astype(np.float32)/100)
 angles=np.arange(n*64,dtype=np.float32).reshape(n,64)*np.float32(.0001)
 pe=np.empty((n,64,2,2),np.float32);pe[:,:,0,0]=np.cos(angles);pe[:,:,0,1]=-np.sin(angles);pe[:,:,1,0]=np.sin(angles);pe[:,:,1,1]=np.cos(angles);write('pe.f32',pe)
 for prefix in ([7] if a.small else [230,57]):
  v=rng.integers(-125,126,(prefix,32,128),dtype=np.int16).astype(np.float32)/64
  write(f'pv-{prefix}.f32',v);write(f'pk-{prefix}.f32',v.transpose(1,0,2).copy())
  reference=None
  for label,variant in ([('fused',2)] if a.sanitizer else [('fused',2),('unfused',2),('expanded-sage',3),('profile',2)]):
   dest=out/f'{prefix}-{label}';dest.mkdir();env=base.copy()
   if label=='unfused':env['GGML_CUDA_DISABLE_FUSION']='1'
   if label=='profile':env.update(LD_PRELOAD=str(root/'runs/activity-profiler/activity.so'),QWEN_ACTIVITY_TRACE=str(dest/'activity.tsv'))
   command=[str(out/'check-bf16-attention'),str(inp),str(dest),str(prefix),str(variant),'1' if a.small else '3']
   if a.sanitizer:command=['/opt/cuda/bin/compute-sanitizer','--tool',a.sanitizer,'--error-exitcode','99',*command]
   (dest/'command.json').write_text(json.dumps(command,indent=2)+'\n')
   print(prefix,label,flush=True)
   with (dest/'process.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   vals=np.fromfile(dest/'y.f32',dtype=np.float32);assert vals.size==n*4096 and np.isfinite(vals).all()
   digest=hashlib.sha256(vals.tobytes()).hexdigest()
   if reference is None:reference=vals.copy()
   diff=int(np.count_nonzero(vals.view(np.uint32)!=reference.view(np.uint32)))
   item=dict(run=dest.name,sha256=digest,values=vals.size,bit_differences=diff,max_abs=float(np.max(np.abs(vals-reference))))
   results.append(item);(out/'validation.json').write_text(json.dumps(results,indent=2)+'\n');print(item,flush=True)
   assert diff==0,'BF16 chain differs from matched-precision reference'
   (dest/'y.f32').unlink()
(out/'input-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('PASS',out,flush=True)
