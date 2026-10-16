"""Compare native merge against installed Comfy/Kitchen using synthetic inputs only."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy-root', type=Path, required=True)
p.add_argument('--run', action='store_true')
a = p.parse_args()
source = ROOT/'.imagine/lora-merge/source/src/model/adapter/lora_int8_merge.cu'
library = ROOT/'.imagine/lora-merge/test-merge.so'
subprocess.run(['/opt/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','-arch=sm_89',
                '-Xcompiler=-fPIC','-shared',str(source),'-lcublas','-o',str(library)],check=True)
if not a.run:
    print('Built synthetic merge comparison library; GPU not used.')
    raise SystemExit(0)
free, util = map(float, subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                  '--format=csv,noheader,nounits'],text=True).strip().split(','))
assert free >= 13000 and util < 20, 'GPU is not idle; no service stopped'
sys.path.insert(0,str(a.comfy_root))
sys.argv=[sys.argv[0]]
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
# All generated tensors below are synthetic. Suppress ordinary library logs.
report_fd=os.dup(1)
with open(os.devnull,'w') as sink:
    os.dup2(sink.fileno(),1);os.dup2(sink.fileno(),2)
def report(item):os.write(report_fd,(json.dumps(item)+'\n').encode())
try:
    import torch
    import numpy as np
    import comfy.options
    comfy.options.enable_args_parsing(False)
    import comfy.lora
    import comfy.utils
    from comfy.weight_adapter.lora import LoRAAdapter
    from comfy_kitchen.tensor import QuantizedTensor,TensorWiseINT8Layout
    torch.backends.cuda.matmul.allow_tf32=False
    class Patch(ctypes.Structure):
        _fields_=[('row_offset',ctypes.c_int),('rows',ctypes.c_int),('rank',ctypes.c_int),
                  ('strength',ctypes.c_float),('up',ctypes.c_void_p),('down',ctypes.c_void_p)]
    lib=ctypes.CDLL(str(library))
    fn=lib.sd_lora_int8_merge_cuda
    fn.restype=ctypes.c_bool
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_int,
                 ctypes.POINTER(Patch),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_bool]
    cases=[('positive',512,137,32,256,[(0,137,1.)]),
           ('negative',512,137,32,256,[(0,137,-.5)]),
           ('zero',512,137,32,256,[(0,137,0.)]),
           ('small',512,137,32,256,[(0,137,.01)]),
           ('ordered',512,137,32,256,[(0,137,1.),(0,137,-.25),(0,137,.5)]),
           ('gate_halves',6144,256,32,256,[(0,128,1.),(128,128,-.75)]),
           ('down_width',24576,128,32,256,[(0,128,1.)]),
           ('rank256',512,128,256,256,[(0,128,1.)])]
    results=[]
    for name,width,rows,rank,group,specs in cases:
        # Dyadic synthetic operands avoid confounding quantizer parity with
        # cuBLAS algorithm-dependent FP32 accumulation at different tile shapes.
        q=((torch.arange(rows*width,dtype=torch.int32).reshape(rows,width)*17+3)%127-63).to(torch.int8).cuda()
        scales=((torch.arange(rows,dtype=torch.float32)%7+1)/4096).reshape(rows,1).cuda()
        params=TensorWiseINT8Layout.Params(scale=scales,orig_dtype=torch.bfloat16,
            orig_shape=(rows,width),is_weight=True,convrot=True,convrot_groupsize=group)
        original=QuantizedTensor(q,'TensorWiseINT8Layout',params)
        reference=original.to(dtype=torch.float16).dequantize()
        key='diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight'
        seed=comfy.utils.string_to_seed(key)
        arrays=[]; native=[]; comfy_patches=[]
        for i,(offset,n,strength) in enumerate(specs):
            up=(((np.arange(n*rank,dtype=np.int32)*3+i)%17-8)/128).astype(np.float32).reshape(n,rank)
            down=(((np.arange(rank*width,dtype=np.int32)*7+i)%19-9)/128).astype(np.float32).reshape(rank,width)
            arrays.extend([up,down])
            native.append(Patch(offset,n,rank,strength,up.ctypes.data,down.ctypes.data))
            adapter=LoRAAdapter(set(),(torch.from_numpy(up).cuda().half(),torch.from_numpy(down).cuda().half(),rank,None,None,None))
            comfy_patches.append((strength,adapter,1.,(0,offset,n),None))
        reference=comfy.lora.calculate_weight(comfy_patches,reference,key)
        expected=original.requantize_from_float(reference,scale='recalculate',stochastic_rounding=seed,inplace_ops=True)
        q_host=q.cpu().numpy().copy();s_host=scales.cpu().numpy().reshape(-1).copy()
        native_array=(Patch*len(native))(*native)
        assert fn(width,rows,group,seed,len(native),native_array,q_host.ctypes.data,s_host.ctypes.data,False)
        eq=expected._qdata.cpu().numpy();es=expected._params.scale.cpu().numpy().reshape(-1)
        item=dict(case=name,elements=int(q_host.size),different_int8=int(np.count_nonzero(q_host!=eq)),
                  max_scale_error=float(np.max(np.abs(s_host-es))))
        results.append(item);report(item)
        assert np.array_equal(q_host,eq),name
        assert np.array_equal(s_host,es),name
        # Resident GPU weights use the same whole-matrix path without a CPU roundtrip.
        device_q=q.clone();device_s=scales.cpu().numpy().reshape(-1).copy()
        assert fn(width,rows,group,seed,len(native),native_array,device_q.data_ptr(),device_s.ctypes.data,True)
        assert np.array_equal(device_q.cpu().numpy(),eq) and np.array_equal(device_s,es)
        # A second preparation from originals must not accumulate the previous patch.
        again_q=q.cpu().numpy().copy();again_s=scales.cpu().numpy().reshape(-1).copy()
        assert fn(width,rows,group,seed,len(native),native_array,again_q.ctypes.data,again_s.ctypes.data,False)
        assert np.array_equal(q_host,again_q) and np.array_equal(s_host,again_s)
        assert not fn(width,rows,0,seed,len(native),native_array,again_q.ctypes.data,again_s.ctypes.data,False)
        assert not fn(width,rows,64,seed,len(native),native_array,again_q.ctypes.data,again_s.ctypes.data,False)
    path=ROOT/'native/lora-merge/synthetic-results.json'
    path.write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
    report(dict(passed=True,cases=len(results)))
except BaseException as exc:
    import traceback
    # Synthetic-only test failures contain no user runtime data.
    report(dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    raise SystemExit(1)
