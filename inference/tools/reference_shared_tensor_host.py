#!/usr/bin/env python3
"""Original Tensor.item/tolist on bounded initialized tensors and meta headers."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import re,struct,warnings
header=a.site/'torch/include/torch/headeronly/core/ScalarType.h'
codes={name:int(code)for name,code in re.findall(r'_\([^\n]+,\s*(\w+)\)\s*/\*\s*(\d+)\s*\*/',header.read_text())}
types=dict(bool='Bool',uint8='Byte',int8='Char',int16='Short',int32='Int',int64='Long',float16='Half',bfloat16='BFloat16',float32='Float',float64='Double',complex32='ComplexHalf',complex64='ComplexFloat',complex128='ComplexDouble',float8_e5m2='Float8_e5m2',float8_e4m3fn='Float8_e4m3fn',float8_e5m2fnuz='Float8_e5m2fnuz',float8_e4m3fnuz='Float8_e4m3fnuz',uint16='UInt16',uint32='UInt32',uint64='UInt64',float8_e8m0fnu='Float8_e8m0fnu')
def encode_host(value):
 if isinstance(value,list):return dict(type='list',items=[encode_host(item)for item in value])
 if isinstance(value,bool):return dict(type='bool',value=value)
 if isinstance(value,int):return dict(type='int',value=str(value))
 if isinstance(value,float):return dict(type='float',bits=struct.pack('>d',value).hex())
 if isinstance(value,complex):return dict(type='complex',real=struct.pack('>d',value.real).hex(),imaginary=struct.pack('>d',value.imag).hex())
 raise AssertionError(type(value))
def capture(desc,value):
 results={}
 for operation in ['item','tolist']:
  try:results[operation]=dict(output=encode_host(getattr(value,operation)()))
  except Exception as error:
   reject_resource_failure(error);results[operation]=dict(error=type(error).__name__,message=str(error))
 return dict(input=desc,**results)
cases=[]
with warnings.catch_warnings():
 warnings.simplefilter('ignore',UserWarning)
 for device in ['cpu','cuda:0']:
  for name,label in types.items():
   dtype=getattr(torch,name);code=codes[label]
   for shape in [[],[0],[1],[2],[2,1],[2,3]]:
    n=math.prod(shape)
    base=torch.arange(n,dtype=torch.float64).reshape(shape)*.25+.25
    if dtype.is_complex:base=torch.complex(base,base*.5)
    if name=='bool':base=base>.5
    if name=='uint64'and n==1:base=torch.tensor(2**64-1,dtype=dtype).reshape(shape)
    value=base.to(dtype)
    raw=value.reshape(-1).view(torch.uint8).tolist()
    desc=dict(kind='dense',dtype=name,code=code,shape=shape,bytes=raw,device=device,layout='plain')
    for layout in ['plain','transpose']if len(shape)==2 else ['plain']:
     value_on_device=value.to(device)
     if layout=='transpose':value_on_device=value_on_device.transpose(0,1)
     cases.append(capture(dict(desc,layout=layout),value_on_device))
 for dtype in [torch.float32,torch.int64]:
  for shape in [[],[0],[1],[2]]:
   cases.append(capture(dict(kind='meta',dtype=str(dtype).split('.')[-1],shape=shape,device='meta'),torch.empty(shape,dtype=dtype,device='meta')))
 for name in ['qint8','quint8','qint32']:
  for shape in [[],[0],[1],[2],[2,3]]:
   base=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)*.25+.25
   value=torch.quantize_per_tensor(base,.125,0,getattr(torch,name))
   cases.append(capture(dict(kind='quantized',dtype=name,shape=shape,device='cpu'),value))
 for layout in ['neg','conj']:
  base=torch.complex(torch.tensor([.25,.75]),torch.tensor([.125,.375]))if layout=='conj'else torch.tensor([.25,.75])
  value=base.conj()if layout=='conj'else torch._neg_view(base)
  raw=base.reshape(-1).view(torch.uint8).tolist()
  cases.append(capture(dict(kind='dense',dtype='complex64'if layout=='conj'else'float32',code=9 if layout=='conj'else 6,shape=[2],bytes=raw,device='cpu',layout=layout),value))
sources={str(header):hashlib.sha256(header.read_bytes()).hexdigest()}
for name in ['tensor_list.cpp','python_scalars.h']:
 source=project/'comfyui/moss-20261018/sources/pytorch/torch/csrc/utils'/name
 sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,torch_version=torch.__version__,trained_model_weights=0,max_tensor_elements=6,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original item/tolist histories; maximum six initialized elements')
