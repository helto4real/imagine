#!/usr/bin/env python3
"""Source-driven synthetic trace fixtures for the frozen Triton policy.

Mock only clocks/device operations or the kernel launch. Execute the original
do_bench/Autotuner methods unchanged. No real user/runtime data is accessed.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.site))
import triton.testing as testing
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources
from triton import knobs
from comfy_kitchen.backends.triton.quantization import _int8_matmul_dequant_kernel as original
from comfy_kitchen.backends.triton.quantization import _int8_matmul_dequant_per_row_kernel as per_row

assert [c.all_kwargs() for c in original.configs]==[c.all_kwargs() for c in per_row.configs]

trace=[]
times=[1.,3.,2.,5.,4.,9.,6.,7.,8.,10.]
class Event:
    next_id=0
    def __init__(self,enable_timing):
        assert enable_timing
        self.id=Event.next_id
        Event.next_id+=1
        trace.append(['event',self.id])
    def record(self):trace.append(['record',self.id])
    def elapsed_time(self,other):
        trace.append(['elapsed',self.id,other.id])
        return 50. if self.id==0 else times[self.id-2]
def sync():trace.append(['sync'])
def cache():trace.append(['cache',256*1024*1024]);return object()
def clear(cache):trace.append(['clear'])
testing.runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(
    get_device_interface=lambda:SimpleNamespace(synchronize=sync,Event=Event),
    get_empty_cache_for_benchmark=cache,clear_cache=clear)))
summary=testing.do_bench(lambda:trace.append(['launch']),quantiles=(0.5,0.2,0.8))
benchmark={'trace':trace,'quantiles':summary}

trace=[]
timings=[[4.,1.,2.],[2.,3.,5.],[2.,1.,8.],[2.,1.,7.],[2.,1.,7.],None]
current=[None]
def config_index(kwargs):
    return next(i for i,c in enumerate(original.configs) if all(kwargs[k]==v for k,v in c.kwargs.items()))
def fake_kernel():pass
def run(**kwargs):
    index=config_index(kwargs)
    current[0]=index
    trace.append(['launch',index])
    if index==5:raise OutOfResources(100,50,'shared memory')
    return index
fake_kernel.run=run
def bench(fn,quantiles):
    assert quantiles==(0.5,0.2,0.8)
    fn()
    index=current[0]
    trace.append(['benchmark',index])
    return timings[index]
knobs.autotuning.cache=False
tuner=Autotuner(fake_kernel,['a','b','c','sa','sb','bias','m','n','k','stride_am'],original.configs,
    ['m','n','k'],None,None,do_bench=bench,cache_results=False)
def call(dtype='torch.float32',stride=128):
    return tuner.run(a=SimpleNamespace(dtype='torch.int8'),b=SimpleNamespace(dtype='torch.int8'),
        c=SimpleNamespace(dtype=dtype),sa=SimpleNamespace(dtype='torch.float32'),sb=SimpleNamespace(dtype='torch.float32'),
        bias=SimpleNamespace(dtype=dtype),m=16,n=128,k=128,stride_am=stride)
winners=[call(),call(stride=256),call(dtype='torch.float16')]
selection={'timings':timings,'winners':winners,'trace':trace,'cache_entries':len(tuner.cache)}
result={'schema':1,'inputs':'synthetic timing/device traces only','benchmark':benchmark,'selection':selection}
result['kernels_with_identical_policy']=[original.base_fn.__name__,per_row.base_fn.__name__]
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('Generated exact original benchmark trace, quantiles and cold/warm/dtype-cache selection fixtures.')
