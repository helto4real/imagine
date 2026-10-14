"""A/B optional native INT8 fusion with full-output numerical checks."""
import argparse
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
parser = argparse.ArgumentParser()
parser.add_argument('--projection', action='store_true', help='Test 4096 projections with gate/up fusion held on')
parser.add_argument('--rows', nargs='+', type=int, help='Focused projection row counts for kernel crossover checks')
args = parser.parse_args()
if args.rows and (not args.projection or min(args.rows) <= 0):
    parser.error('--rows requires --projection and positive counts')
experiment = here.parent
root = experiment.parents[1]
run = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+'fused-', dir=experiment/'runs'))
print(run, flush=True)
shutil.copy2(experiment/'runs/linear-benchmark/linear', run/'linear')
for name in ('compare-fused.py', 'linear.cpp', 'activity.cpp'):
    shutil.copyfile(here/name, run/name)
for name in ('int8-gate-up.cu', 'int8-gate-up.cuh'):
    shutil.copyfile(root/'stable-diffusion.cpp/ggml/src/ggml-cuda'/name, run/name)
shutil.copyfile(experiment/'runs/activity-profiler/activity.so', run/'activity.so')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

summary = {'binary_sha256': digest(run/'linear'), 'projection_experiment':args.projection, 'cases': []}
cases = [('single',1,4096,24576), ('prefix57',57,4096,24576),
         ('prefix230',230,4096,24576), ('steady',5600,4096,24576),
         ('positive',5830,4096,24576), ('negative',5657,4096,24576),
         ('projection-fallback',5600,4096,4096)]
if args.projection:
    cases = [(label,m,4096,4096) for label,m in
             [('single',1),('tail3',3),('prefix57',57),('prefix230',230),('below-floor',511),('at-floor',512),
              ('steady',5600),('positive',5830),('negative',5657)]]
    cases += [('down-fallback',5600,12288,4096), ('gate-up-control',5600,4096,24576)]
    if args.rows:
        cases = [(f'rows{m}',m,4096,4096) for m in args.rows]
with (run/'gpu.csv').open('w') as gpu:
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu', '--format=csv', '-l', '1'], stdout=gpu)
    try:
        for label, m, k, n in cases:
            case = {'name': label, 'm': m, 'k': k, 'n': n, 'variants': {}}
            summary['cases'].append(case)
            with tempfile.TemporaryDirectory(dir=run, prefix='input-') as tmp:
                inp = Path(tmp)
                rng = np.random.default_rng(20261013)
                # Exercise ordinary F32 values and non-power-of-two channel scales.
                x = rng.normal(0, 1, (m,k)).astype(np.float32)
                x[0] = 0  # Quantizer's zero-row behavior.
                w = rng.integers(-127,128,(n,k),dtype=np.int8)
                scales = rng.uniform(.0001,.03,n).astype(np.float32)
                for name, values in [('x.f32',x), ('w.i8',w), ('scale.f32',scales)]:
                    values.tofile(inp/name)
                case['input_sha256'] = {p.name: digest(p) for p in inp.iterdir()}
                del x, w, scales
                outputs = {}
                for variant, enabled in [('fallback','0'), ('fused','1')]:
                    dest = run/f'{label}-{variant}'
                    dest.mkdir()
                    gate_up = '1' if args.projection else enabled
                    projection = enabled if args.projection else '0'
                    env = {**os.environ, 'GGML_CUDA_FUSED_INT8_GATE_UP':gate_up,
                           'GGML_CUDA_FUSED_INT8_PROJECTION':projection, 'QWEN_LINEAR_VALIDATE_ALL':'1'}
                    command = [str(run/'linear'),str(inp),str(dest),str(m),str(k),str(n),'7','30']
                    (dest/'command.json').write_text(json.dumps({'argv':command, 'gate_up_fusion':gate_up,
                        'projection_fusion':projection, 'validate_all':True},indent=2)+'\n')
                    with (dest/'process.log').open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    result = json.loads((dest/'result.json').read_text())
                    timings = [(b['end_ns']-b['start_ns'])/1e6/b['iterations'] for b in result['batches']]
                    outputs[variant] = {name:digest(dest/name) for name in ('y.f32','q.i8','scales.f32')}
                    case['variants'][variant] = {'median_ms':statistics.median(timings),'batch_means_ms':timings,'output_sha256':outputs[variant]}
                    vals = np.memmap(dest/'y.f32',dtype=np.float32,mode='r')
                    assert np.isfinite(vals).all()
                    del vals
                    for name in outputs[variant]:
                        (dest/name).unlink()
                    # Separate profiled calls prove dispatch, including tails and fallback.
                    trace = dest/'trace'
                    trace.mkdir()
                    env.pop('QWEN_LINEAR_VALIDATE_ALL')
                    env.update(LD_PRELOAD=str(run/'activity.so'), QWEN_ACTIVITY_TRACE=str(trace/'activity.tsv'))
                    command[2] = str(trace)
                    command[-2:] = ['1','5']
                    with (trace/'process.log').open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    bounds = json.loads((trace/'result.json').read_text())['batches'][0]
                    lines = (trace/'activity.tsv').read_text().splitlines()
                    assert lines[-1] == '# dropped_records=0'
                    groups = {}
                    for row in csv.DictReader((line for line in lines if not line.startswith('#')), delimiter='\t'):
                        start,end = int(row['start_ns']),int(row['end_ns'])
                        assert not any(start < b < end for b in (bounds['start_ns'],bounds['end_ns']))
                        if not bounds['start_ns'] <= start <= end <= bounds['end_ns']: continue
                        name = row['name']
                        category = ('fused_gemm' if 'DefaultGemmWithVisitor' in name else
                                    'dequant' if 'dequantize_i32' in name else
                                    'gemm' if 'gemm' in name else
                                    'convrot' if 'convrot' in name else row['kind'])
                        group = groups.setdefault(category, {'count':0,'ms_per_call':0})
                        group['count'] += 1
                        group['ms_per_call'] += (end-start)/1e6/5
                    expected_fused = k == 4096 and ((gate_up == '1' and n == 24576) or (projection == '1' and n == 4096 and m >= 512))
                    assert groups.get('fused_gemm',{}).get('count',0) == (5 if expected_fused else 0), groups
                    assert groups.get('dequant',{}).get('count',0) == (0 if expected_fused else 5), groups
                    case['variants'][variant]['trace'] = groups
                assert outputs['fallback'] == outputs['fused'], f'Full-output mismatch: {label}'
                case['bit_identical_all_outputs'] = True
                case['output_values_checked'] = m*n
            (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(label, {v:round(r['median_ms'],4) for v,r in case['variants'].items()}, 'all outputs bit-identical',flush=True)
    finally:
        monitor.terminate()
        monitor.wait()
print('DONE',run,flush=True)
