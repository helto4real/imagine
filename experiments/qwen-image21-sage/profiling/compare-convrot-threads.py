"""Check complete outputs against the old binary and benchmark ConvRot thread counts."""
import csv
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

here = Path(__file__).resolve().parent
experiment = here.parent
build = experiment / 'runs/convrot-threads-build'
run = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-convrot-threads-'), dir=experiment/'runs'))
for name in ['linear', 'linear-before']:
    shutil.copy2(build/name, run/name)
shutil.copyfile(__file__, run/'compare-convrot-threads.py')
shutil.copyfile(here/'linear.cpp', run/'linear.cpp')
shutil.copyfile(experiment/'runs/activity-profiler/activity.so', run/'activity.so')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rotate(x):
    x = x.copy() * np.float32(1/16)
    for stride in [1, 4, 16, 64]:
        v = x.reshape(-1, 256//(4*stride), 4, stride)
        a,b,c,d = [v[:,:,i,:].copy() for i in range(4)]
        v[:,:,0,:] = a+b+c-d
        v[:,:,1,:] = a+b-c+d
        v[:,:,2,:] = a-b+c+d
        v[:,:,3,:] = -a+b+c+d
    return x

summary = {'binary_sha256': {n:digest(run/n) for n in ['linear','linear-before']},
           'method':'20 warmups; 7 synchronized batches of 30 calls per mode; separate five-call CUPTI trace',
           'stress_rows':['zero','small 1e-20','large 1e10','impulse','alternating sign','quantization half steps','subnormal 1e-38','outlier','near half steps'],
           'cases': []}
cases = [(f'rows{m}',m,4096,4096) for m in [1,3,16,57,230,511,512,513,5600,5657,5830]]
cases += [('gate-up',5600,4096,24576), ('down-control',5600,12288,4096)]
env_base = {**os.environ, 'GGML_CUDA_FUSED_INT8_GATE_UP':'1', 'GGML_CUDA_FUSED_INT8_PROJECTION':'1', 'QWEN_LINEAR_VALIDATE_ALL':'1'}
assert not any(env_base.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_LINEAR_QUANT_ONLY'])
print(run,flush=True)
with (run/'gpu.csv').open('w') as gpu, (run/'gpu-monitor.log').open('w') as errors:
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=gpu,stderr=errors)
    try:
        for case_index,(label,m,k,n) in enumerate(cases):
            case = {'name':label,'m':m,'k':k,'n':n,'variants':{}}
            summary['cases'].append(case)
            with tempfile.TemporaryDirectory(dir=run,prefix='input-') as tmp:
                inp = Path(tmp)
                rng = np.random.default_rng(20261013)
                x = rng.normal(0,1,(m,k)).astype(np.float32)
                x[0] = 0
                if m>1: x[1] *= np.float32(1e-20)
                if m>2: x[2] *= np.float32(1e10)
                if m>3: x[3]=0; x[3,k//2]=1
                if m>4: x[4]=np.tile(np.array([-1,1],np.float32),k//2)
                if m>5:
                    target = np.resize(np.array([127,-127,.5,-.5,1.5,-1.5,63.5,-63.5,126.5,-126.5],np.float32),k)
                    x[5] = rotate(target)
                if m>6: x[6] *= np.float32(1e-38)
                if m>7: x[7,k//3] = 1e4
                if m>8:
                    target[2:] = np.nextafter(target[2:], np.float32(np.inf))
                    x[8] = rotate(target)
                w = rng.integers(-127,128,(n,k),dtype=np.int8)
                scales = rng.uniform(.0001,.03,n).astype(np.float32)
                for name,values in [('x.f32',x),('w.i8',w),('scale.f32',scales)]: values.tofile(inp/name)
                case['input_sha256'] = {p.name:digest(p) for p in inp.iterdir()}
                del x,w,scales
                order = ['original'] + ([1024,512,256] if case_index%2==0 else [256,512,1024])
                for threads in order:
                    key=str(threads)
                    dest=run/f'{label}-{key}';dest.mkdir()
                    env={**env_base,'GGML_CUDA_CONVROT_THREADS':key}
                    binary=run/('linear-before' if threads=='original' else 'linear')
                    command=[str(binary),str(inp),str(dest),str(m),str(k),str(n),'7','30']
                    (dest/'command.json').write_text(json.dumps({'argv':command,'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','QWEN_'))}},indent=2)+'\n')
                    with (dest/'process.log').open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    batches=json.loads((dest/'result.json').read_text())['batches']
                    times=[(b['end_ns']-b['start_ns'])/1e6/b['iterations'] for b in batches]
                    hashes={name:digest(dest/name) for name in ['y.f32','q.i8','scales.f32']}
                    for name in ['y.f32','scales.f32']:
                        data=np.memmap(dest/name,dtype=np.float32,mode='r');assert np.isfinite(data).all();del data
                    case['variants'][key]={'chain_median_ms':statistics.median(times),'chain_batches_ms':times,'output_sha256':hashes}
                    if threads!='original':
                        assert hashes==case['variants']['original']['output_sha256'],f'Numerical mismatch: {label}/{key}'
                    for name in hashes: (dest/name).unlink()
                    if threads=='original': continue
                    quant=dest/'quant';quant.mkdir()
                    command[2]=str(quant)
                    with (quant/'process.log').open('w') as log:
                        subprocess.run(command,env={**env,'QWEN_LINEAR_QUANT_ONLY':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
                    batches=json.loads((quant/'result.json').read_text())['batches']
                    times=[(b['end_ns']-b['start_ns'])/1e6/b['iterations'] for b in batches]
                    case['variants'][key].update(quant_median_ms=statistics.median(times),quant_batches_ms=times)
                    for name in ['q.i8','scales.f32']:
                        assert digest(quant/name)==hashes[name]
                        (quant/name).unlink()
                    trace=dest/'trace';trace.mkdir()
                    command[2]=str(trace);command[-2:]=['1','5']
                    with (trace/'process.log').open('w') as log:
                        subprocess.run(command,env={**env,'LD_PRELOAD':str(run/'activity.so'),'QWEN_ACTIVITY_TRACE':str(trace/'activity.tsv')},stdout=log,stderr=subprocess.STDOUT,check=True)
                    bounds=json.loads((trace/'result.json').read_text())['batches'][0]
                    lines=(trace/'activity.tsv').read_text().splitlines();assert lines[-1]=='# dropped_records=0'
                    hits=[]
                    for row in csv.DictReader((l for l in lines if not l.startswith('#')),delimiter='\t'):
                        start,end=int(row['start_ns']),int(row['end_ns'])
                        assert not any(start<b<end for b in [bounds['start_ns'],bounds['end_ns']])
                        if bounds['start_ns']<=start<=end<=bounds['end_ns'] and 'quantize_rowwise_i8_convrot_cuda' in row['name']:
                            hits.append(row)
                    actual=threads if k==4096 else 1024
                    assert len(hits)==5 and all(row['block']==f'{actual},1,1' and f'ILi{actual}EE' in row['name'] for row in hits),hits
                    case['variants'][key]['convrot_gpu_ms']=sum(int(h['end_ns'])-int(h['start_ns']) for h in hits)/5e6
                    case['variants'][key]['verified_kernel']=hits[0]['name']
                    for name in hashes: (trace/name).unlink()
                case['all_outputs_bit_identical']=True
                case['output_values_checked']=m*n
            (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(label,{v:{k:round(r[k],4) for k in ['chain_median_ms','quant_median_ms','convrot_gpu_ms'] if k in r} for v,r in case['variants'].items()},'bit-identical',flush=True)
    finally:
        monitor.terminate();monitor.wait()
print('DONE',run,flush=True)
