"""Run sequential, matched-data native/Kitchen microbenchmarks and traced repeats."""
import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--case', choices=['all', 'projection', 'gate_up', 'down'], default='all')
parser.add_argument('--timing-only', action='store_true')
parser.add_argument('--reverse-order', action='store_true')
args = parser.parse_args()
experiment = here.parent
run = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+'linear-', dir=experiment/'runs'))
print(run, flush=True)
python = '/path/to/ComfyUI/.venv/bin/python'
binary = experiment/'runs/linear-benchmark/linear'
tracer = experiment/'runs/activity-profiler/activity.so'
for source in ('linear.cpp', 'kitchen-linear.py', 'compare-linear.py', 'activity.cpp'):
    shutil.copyfile(here/source, run/source)
shutil.copy2(binary, run/'linear')
shutil.copyfile(tracer, run/'activity.so')
manifest = {'cases':[], 'timing':'synchronized host time per batch, excluding initialization, transfer and warmup; separate traced repeats'}
variants = ['native', 'f32', 'f32-bf16', 'bf16', 'fallback-f32']
if args.reverse_order:
    variants.reverse()
with (run/'gpu.csv').open('w') as gpu:
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=gpu)
    try:
        for label, k, n in [('projection',4096,4096),('gate_up',4096,24576),('down',12288,4096)]:
            if args.case not in ('all', label):
                continue
            m=5600
            case = {'name':label, 'm':m, 'k':k, 'n':n, 'input_hashes':{}}
            manifest['cases'].append(case)
            with tempfile.TemporaryDirectory(dir=run, prefix='synthetic-') as tmp:
                inp=Path(tmp)
                random=np.random.default_rng(42)
                # All activation values are exactly representable in BF16 and F32.
                x=random.integers(-125,126,size=(m,k),dtype=np.int16).astype(np.float32)/64
                w=random.integers(-127,128,size=(n,k),dtype=np.int8)
                s=np.full(n,1/128,dtype=np.float32)
                for name,data in [('x.f32',x),('w.i8',w),('scale.f32',s)]:
                    data.tofile(inp/name)
                    case['input_hashes'][name]=hashlib.sha256(data.tobytes()).hexdigest()
                del x,w,s
                for traced in ((False,) if args.timing_only else (False, True)):
                    for variant in variants:
                        dest=run/f'{label}-{variant}-{"trace" if traced else "timing"}'
                        dest.mkdir()
                        batches,iterations=(1,5) if traced else (7,30)
                        if variant=='native':
                            command=[str(run/'linear'),str(inp),str(dest),str(m),str(k),str(n),str(batches),str(iterations)]
                        else:
                            command=[python,str(here/'kitchen-linear.py'),str(inp),str(dest),str(m),str(k),str(n),variant,str(batches),str(iterations)]
                        env=os.environ.copy()
                        if traced:
                            env['LD_PRELOAD']=str(run/'activity.so')
                            env['QWEN_ACTIVITY_TRACE']=str(dest/'activity.tsv')
                        (dest/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                        print(label,variant,'trace' if traced else 'timing',flush=True)
                        with (dest/'process.log').open('w') as log:
                            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    finally:
        monitor.terminate()
        monitor.wait()
print('DONE',run,flush=True)
