#!/usr/bin/env python3
"""Inspect owned wrapper DLPack metadata, stopping before native GEMM launch."""
import argparse,ctypes,hashlib,json,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[2]/'design/reference-local/source-manifest.json').read_text());hashes={}
for e in manifest:
    if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and Path(e['path']).suffix=='.py':
        h=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert h==e['sha256'];hashes[e['path']]=h
sys.path.insert(0,str(a.site));import torch,comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.backends import cuda
class Device(ctypes.Structure):_fields_=[('type',ctypes.c_int),('id',ctypes.c_int)]
class Dtype(ctypes.Structure):_fields_=[('code',ctypes.c_uint8),('bits',ctypes.c_uint8),('lanes',ctypes.c_uint16)]
class Tensor(ctypes.Structure):_fields_=[('data',ctypes.c_void_p),('device',Device),('ndim',ctypes.c_int),('dtype',Dtype),('shape',ctypes.POINTER(ctypes.c_int64)),('strides',ctypes.POINTER(ctypes.c_int64)),('offset',ctypes.c_uint64)]
class Managed(ctypes.Structure):_fields_=[('tensor',Tensor),('context',ctypes.c_void_p),('deleter',ctypes.c_void_p)]
get=ctypes.pythonapi.PyCapsule_GetPointer;get.restype=ctypes.c_void_p;get.argtypes=[ctypes.py_object,ctypes.c_char_p]
def inspect(capsule):
    t=ctypes.cast(get(capsule,b'dltensor'),ctypes.POINTER(Managed)).contents.tensor
    return dict(null_data=t.data is None,device=[t.device.type,t.device.id],dtype=[t.dtype.code,t.dtype.bits,t.dtype.lanes],shape=[t.shape[i]for i in range(t.ndim)],strides=[t.strides[i]for i in range(t.ndim)]if t.strides else None,byte_offset=t.offset)
records=[]
for dt in [torch.float16,torch.bfloat16]:
    b=torch.arange(7,device='cuda:0',dtype=torch.float32).mul(.03123).to(dt)
    with ck.use_backend('eager'):q=QuantizedTensor.from_float(b,'TensorWiseINT8Layout')
    vector=cuda._gemm_vector_arg(q,q.device,dt)
    wrapper=cuda._wrap_for_dlpack(vector)
    metadata=inspect(wrapper)
    assert vector is q and metadata['null_data']and metadata['shape']==[7]and metadata['dtype'][1]==16
    # Ordinary control uses its actual non-null allocation and same logical shape.
    control=inspect(cuda._wrap_for_dlpack(b));assert not control['null_data']and control['shape']==[7]
    records.append(dict(original_dtype=str(dt),original_vector_returns_same_quantized_object=True,wrapper_python_data_ptr_nonzero=q.data_ptr()!=0,physical_payload_dtype=str(q._qdata.dtype),quantized_dlpack=metadata,ordinary_dlpack=control))
report=dict(result='metadata proved',inspection_only=True,native_gemm_called=False,unknown_or_unwritten_payload_read=False,source_hashes=hashes,cases=records)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='source_hashes'},indent=2))
