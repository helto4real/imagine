"""Compare shape-targeted CUTLASS choices in the actual ConvRot + INT8 Linear graph."""
import argparse,csv,datetime,hashlib,json,os,re,shutil,statistics,subprocess,tempfile
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
experiment=here.parent
build=experiment/'runs/gate-up-tuning-build'
configs=['128x256x64-s3','128x128x64-s3','64x256x64-s3','64x128x64-s3','128x256x64-s2','128x256x64-s4']
parser=argparse.ArgumentParser()
parser.add_argument('--configs',nargs='+',choices=configs)
parser.add_argument('--rows',nargs='+',type=int,default=[5600,5657,5830])
parser.add_argument('--reverse',action='store_true')
parser.add_argument('--controls',action='store_true')
args=parser.parse_args();assert min(args.rows)>0
selected=args.configs or configs
run=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-gate-up-configs-'),dir=experiment/'runs'))
for name in ['linear','linear-before','int8-gate-up.cu']:shutil.copy2(build/name,run/name)
for name in ['linear.cpp','compare-gate-up-configs.py']:shutil.copy2(here/name,run/name)
shutil.copy2(experiment/'runs/activity-profiler/activity.so',run/'activity.so')
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
summary={'binary_sha256':{n:digest(run/n) for n in ['linear','linear-before']},'method':'20 warmups then seven synchronized batches of 30 calls; separate five-call GPU trace; all output values checked','cases':[]}
base={**os.environ,'GGML_CUDA_FUSED_INT8_GATE_UP':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_CONVROT_THREADS':'512','GGML_CUDA_FUSED_SWIGLU_CONVROT':'1','QWEN_LINEAR_VALIDATE_ALL':'1'}
assert not any(base.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE','QWEN_LINEAR_QUANT_ONLY'])
cases=[(f'rows{m}',m,4096,24576) for m in args.rows]
if args.controls:cases += [('projection-control',5600,4096,4096),('down-control',5600,12288,4096)]
print(run,flush=True)
with (run/'gpu.csv').open('w') as gpu:
 monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=gpu)
 try:
  for index,(label,m,k,n) in enumerate(cases):
   case={'name':label,'m':m,'k':k,'n':n,'variants':{}};summary['cases'].append(case)
   with tempfile.TemporaryDirectory(dir=run,prefix='inputs-') as tmp:
    inp=Path(tmp);rng=np.random.default_rng(20261013)
    x=rng.normal(0,1,(m,k)).astype(np.float32)
    if m>1:x[0]=0
    if m>3:x[1]*=1e-20;x[2]*=1e10;x[3,k//3]=1e4
    w=rng.integers(-127,128,(n,k),dtype=np.int8);scales=rng.uniform(.0001,.03,n).astype(np.float32)
    for name,values in [('x.f32',x),('w.i8',w),('scale.f32',scales)]:values.tofile(inp/name)
    case['input_sha256']={p.name:digest(p) for p in inp.iterdir()};del x,w,scales
    order=selected[index%len(selected):]+selected[:index%len(selected)]
    if args.reverse:order=order[::-1]
    for config in ['original']+order:
     dest=run/f'{label}-{config}';dest.mkdir()
     env={**base,'GGML_CUDA_GATE_UP_CONFIG':config}
     cmd=[str(run/('linear-before' if config=='original' else 'linear')),str(inp),str(dest),str(m),str(k),str(n),'7','30']
     (dest/'command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','QWEN_'))}},indent=2)+'\n')
     with (dest/'process.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
     batches=json.loads((dest/'result.json').read_text())['batches'];times=[(b['end_ns']-b['start_ns'])/1e6/b['iterations'] for b in batches]
     names=['y.f32','q.i8','scales.f32'];hashes={n:digest(dest/n) for n in names}
     for name in ['y.f32','scales.f32']:
      data=np.memmap(dest/name,dtype=np.float32,mode='r');assert np.isfinite(data).all();del data
     item={'median_ms':statistics.median(times),'batch_means_ms':times,'output_sha256':hashes};case['variants'][config]=item
     if config!='original':assert hashes==case['variants']['original']['output_sha256'],f'{label}/{config}: numerical mismatch'
     for name in names:(dest/name).unlink()
     if config!='original':
      trace=dest/'trace';trace.mkdir();cmd[2]=str(trace);cmd[-2:]=['1','5']
      with (trace/'process.log').open('w') as log:subprocess.run(cmd,env={**env,'LD_PRELOAD':str(run/'activity.so'),'QWEN_ACTIVITY_TRACE':str(trace/'activity.tsv')},stdout=log,stderr=subprocess.STDOUT,check=True)
      bounds=json.loads((trace/'result.json').read_text())['batches'][0]
      lines=(trace/'activity.tsv').read_text().splitlines();assert lines[-1]=='# dropped_records=0'
      rows=list(csv.DictReader((l for l in lines if not l.startswith('#')),delimiter='\t'))
      assert not any(int(r['start_ns'])<b<int(r['end_ns']) for r in rows for b in [bounds['start_ns'],bounds['end_ns']])
      rows=[r for r in rows if bounds['start_ns']<=int(r['start_ns'])<=int(r['end_ns'])<=bounds['end_ns']]
      fused=[r for r in rows if 'DefaultGemmWithVisitor' in r['name']]
      expected=k==4096
      assert len(fused)==(5 if expected else 0),f'{label}/{config}: unexpected fallback'
      if expected:
       actual=config if n==24576 else '128x256x64-s3'
       tm,tn,st=map(int,re.fullmatch(r'(\d+)x(\d+)x64-s(\d+)',actual).groups())
       demangled=subprocess.run(['c++filt',fused[0]['name']],capture_output=True,text=True,check=True).stdout.strip()
       assert f'cutlass::arch::Sm80, cutlass::gemm::GemmShape<{tm}, {tn}, 64>' in demangled,demangled
       assert f'GemmIdentityThreadblockSwizzle<1>, {st},' in demangled,demangled
       assert all(r['grid']==f'{(m+tm-1)//tm},{(n+tn-1)//tn},1' and r['block']==f'{tm//64*(tn//64)*32},1,1' for r in fused)
       item['verified_kernel']=demangled
      item['fused_gemm_ms']=sum(int(r['end_ns'])-int(r['start_ns']) for r in fused)/5e6
      item['gpu_activity_ms']=sum(int(r['end_ns'])-int(r['start_ns']) for r in rows)/5e6
      for name in names:
       assert digest(trace/name)==hashes[name]
       (trace/name).unlink()
     (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
     print(label,config,round(item['median_ms'],4),'ms; bit-identical',flush=True)
    case['all_outputs_bit_identical']=True;case['output_values_checked']=m*n
   (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 finally:monitor.terminate();monitor.wait()
print('DONE',run,flush=True)
