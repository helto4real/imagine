"""Matched synthetic data, sequential engines, warm event timings and CUPTI."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
here=Path(__file__).resolve().parent
root=here.parent
out=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-attention-chain-'),dir=root/'runs'))
print(out,flush=True)
binary=root/'runs/attention-chain-build/attention-chain'
shutil.copy2(binary,out/'attention-chain')
tracer=root/'runs/activity-profiler/activity.so'
shutil.copy2(tracer,out/'activity.so')
for name in ['attention-chain.cpp','attention-stream.cu','kitchen-attention-chain.py','compare-attention-chain.py']:
    shutil.copyfile(here/name,out/name)
env={**os.environ,'GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_FUSED_RMS_ROPE':'1','GGML_CUDA_CONVROT_THREADS':'512','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(out/'cuda-cache')}
for key in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE']:env.pop(key,None)
summary=[]
hashes={}
with (out/'gpu.csv').open('w') as log:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=log)
    try:
        with tempfile.TemporaryDirectory(prefix='synthetic-',dir=out) as tmp:
            inp=Path(tmp)
            rng=np.random.default_rng(42)
            def write(name,value):
                value.tofile(inp/name);hashes[name]=hashlib.sha256(value.tobytes()).hexdigest()
            write('x.f32',rng.integers(-125,126,(5600,4096),dtype=np.int16).astype(np.float32)/64)
            for i in range(4):write(f'w{i}.i8',rng.integers(-127,128,(4096,4096),dtype=np.int8))
            write('scale.f32',np.full(4096,1/4096,dtype=np.float32))
            write('norm.f32',np.ones(128,dtype=np.float32))
            angle=np.arange(5600*64,dtype=np.float32).reshape(5600,64)*np.float32(.0001)
            pe=np.empty((5600,64,2,2),np.float32)
            pe[:,:,0,0]=np.cos(angle);pe[:,:,0,1]=-np.sin(angle);pe[:,:,1,0]=np.sin(angle);pe[:,:,1,1]=np.cos(angle)
            write('pe.f32',pe)
            for prefix in [230,57]:
                v=rng.integers(-125,126,(prefix,32,128),dtype=np.int16).astype(np.float32)/64
                write(f'pv-{prefix}.f32',v);write(f'pk-{prefix}.f32',v.transpose(1,0,2).copy())
            for traced in ([False] if a.smoke else [False,True]):
                for prefix in ([230] if a.smoke else [230,57]):
                    order=['native','direct','kitchen'] if prefix==230 else ['kitchen','direct','native']
                    for variant in order:
                        dest=out/f'{prefix}-{variant}-{"trace" if traced else "timing"}'
                        dest.mkdir()
                        iterations=2 if a.smoke else 3 if traced else 30
                        if variant=='kitchen':cmd=['/path/to/ComfyUI/.venv/bin/python',str(here/'kitchen-attention-chain.py'),str(inp),str(dest),str(prefix),str(iterations)]
                        else:cmd=[str(out/'attention-chain'),str(inp),str(dest),str(prefix),str(int(variant=='direct')),str(iterations)]
                        current=env.copy()
                        if traced:current.update(LD_PRELOAD=str(out/'activity.so'),QWEN_ACTIVITY_TRACE=str(dest/'activity.tsv'))
                        (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
                        print(dest.name,flush=True)
                        with (dest/'process.log').open('w') as f:subprocess.run(cmd,env=current,stdout=f,stderr=subprocess.STDOUT,check=True)
                        data=json.loads((dest/'result.json').read_text())
                        img=np.fromfile(dest/'y.f32',dtype=np.float32);assert img.size==5600*4096 and np.isfinite(img).all()
                        item=dict(run=dest.name,prefix=prefix,variant=variant,traced=traced,output_sha256=hashlib.sha256(img.tobytes()).hexdigest(),mean_ms={key:statistics.mean(b['cuda_ms'] for b in data['batches'] if b['part']==key) for key in {b['part'] for b in data['batches']}})
                        del img
                        summary.append(item)
                        if variant!='kitchen':
                            item['v_sha256']=hashlib.sha256((dest/'v.f16').read_bytes()).hexdigest()
                            peers=[s for s in summary if s['prefix']==prefix and s['traced']==traced and s['variant']!='kitchen']
                            if len(peers)==2:
                                assert peers[0]['output_sha256']==peers[1]['output_sha256'],'native A/B output changed'
                                assert peers[0]['v_sha256']==peers[1]['v_sha256'],'native A/B cast changed'
                            (dest/'v.f16').unlink()
                        (dest/'y.f32').unlink()
                        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            (out/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,environment={k:v for k,v in env.items() if k.startswith('GGML_')},binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),synthetic=True,tensor_values_persisted=False),indent=2)+'\n')
    finally:monitor.terminate();monitor.wait()
print('DONE',out,flush=True)
