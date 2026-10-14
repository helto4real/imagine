"""Summarize matched Linear timings, GPU stages, dispatch and sampled numerics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
a=parser.parse_args()
run=a.run.resolve()
manifest=json.loads((run/'manifest.json').read_text())
summary={}
kernel_rows=[]
for case in manifest['cases']:
    label=case['name']
    native=run/f'{label}-native-timing'
    reference={name:np.fromfile(native/file,dtype=dtype) for name,file,dtype in
               [('q','q.i8',np.int8),('scales','scales.f32',np.float32),('y','y.f32',np.float32)]}
    summary[label]={}
    for variant in ['native','f32','f32-bf16','bf16','fallback-f32']:
        dest=run/f'{label}-{variant}-timing'
        result=json.loads((dest/'result.json').read_text())
        ms=[(b['end_ns']-b['start_ns'])/1e6/b['iterations'] for b in result['batches']]
        metrics={}
        for name,file,dtype in [('q','q.i8',np.int8),('scales','scales.f32',np.float32),('y','y.f32',np.float32)]:
            values=np.fromfile(dest/file,dtype=dtype)
            assert len(values)==len(reference[name]) and np.isfinite(values).all()
            difference=values.astype(np.float64)-reference[name].astype(np.float64)
            metrics[name]={'sampled_values':len(values),'bit_identical':values.tobytes()==reference[name].tobytes(),
                           'unequal_fraction':float(np.mean(values!=reference[name])),
                           'max_abs_error':float(np.abs(difference).max()),
                           'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(reference[name].astype(np.float64)))}
        if variant in ('native','f32','fallback-f32'):
            assert all(v['bit_identical'] for v in metrics.values()), 'F32 numerical mismatch'
        item={'median_ms':statistics.median(ms),'min_ms':min(ms),'max_ms':max(ms),
              'batch_means_ms':ms,'numerics':metrics,'dispatch':result.get('dispatch')}
        trace=run/f'{label}-{variant}-trace'
        if trace.exists():
            bounds=json.loads((trace/'result.json').read_text())['batches'][0]
            lines=(trace/'activity.tsv').read_text().splitlines()
            assert lines[-1]=='# dropped_records=0'
            groups=defaultdict(float)
            symbols=defaultdict(lambda:[0,0.0])
            crossings=0
            for row in csv.DictReader((x for x in lines if not x.startswith('#')),delimiter='\t'):
                start,end=int(row['start_ns']),int(row['end_ns'])
                assert end>=start>0
                crossings+=sum(start<b<end for b in (bounds['start_ns'],bounds['end_ns']))
                if not (bounds['start_ns']<=start and end<=bounds['end_ns']):continue
                name=row['name']
                if row['kind']!='kernel':group=row['kind']
                elif 'dequantize' in name:group='separate_dequantization'
                elif 'convrot' in name:group='convrot_quantization'
                elif 'DefaultGemmWithVisitor' in name:group='gemm_with_fused_output'
                elif 'gemm' in name:group='gemm'
                else:group='other_kernel'
                duration=(end-start)/1e6/bounds['iterations']
                groups[group]+=duration
                symbols[name][0]+=1;symbols[name][1]+=duration
            assert crossings==0
            item['profile_gpu_ms_per_call']=dict(groups)
            item['dropped_records']=0
            item['boundary_crossing_events']=crossings
            names=list(symbols)
            demangled=subprocess.run(['c++filt'],input='\n'.join(names)+'\n',text=True,capture_output=True,check=True).stdout.splitlines()
            for name,readable in zip(names,demangled):
                kernel_rows.append([label,variant,symbols[name][0],symbols[name][1],readable])
        summary[label][variant]=item
(run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (run/'kernels.csv').open('w') as file:
    writer=csv.writer(file);writer.writerow(['case','variant','launches','gpu_ms_per_call','kernel'])
    writer.writerows(kernel_rows)
for label,variants in summary.items():
    print(label)
    for variant,item in variants.items():
        print(f"  {variant:14s} {item['median_ms']:.4f} ms ({item['min_ms']:.4f}..{item['max_ms']:.4f})",item.get('profile_gpu_ms_per_call',{}))
