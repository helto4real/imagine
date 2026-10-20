#!/usr/bin/env python3
"""CPU-only own-data feasibility probe; not checkpoint parity acceptance."""
import argparse, collections, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
base=torch.arange(12,dtype=torch.float32).reshape(3,4)*.125-.5
weights=collections.OrderedDict(weight=base,bias=base[0])
weights._metadata=collections.OrderedDict([('',{'version':1})])
inputs=[('plain',{'weight':base}),('state-dict',{'state_dict':{'weight':base},'epoch':7}),
 ('single-wrapper',{'model':{'weight':base}}),('metadata',{'weight':base,'flag':True,'name':'own','epoch':7,'scale':.125,'nothing':None}),
 ('ordered',weights),('aliases',{'a':base,'same':base,'view':base.T,'slice':base[1:]}),
 ('parameter',{'weight':torch.nn.Parameter(base.clone())}),('size',{'weight':base,'shape':torch.Size([3,4])}),
 ('device',{'weight':base,'device':torch.device('cpu')}),('dtype',{'weight':base,'dtype':torch.bfloat16}),
 ('set',{'weight':base,'set':{1,2}}),('complex',{'weight':base,'value':complex(.5,-.25)}),
 ('quantized',{'weight':torch.quantize_per_tensor(base,.125,0,torch.qint8)}),
 ('sparse',{'weight':base.to_sparse()}),('uint64',{'weight':torch.tensor([2**64-1],dtype=torch.uint64)})]
cases=[]
with tempfile.TemporaryDirectory(prefix='imagine-stage3-own-torch-archives-')as own:
 for name,value in inputs:
  for protocol in [2,4]:
   path=Path(own)/f'{name}-{protocol}.pt';torch.save(value,path,pickle_protocol=protocol)
   try:
    reference=torch.load(path,map_location='cpu',weights_only=True)
    original=dict(result='loaded',type=type(reference).__name__)
   except Exception as error:original=dict(result='failed',error=type(error).__name__,message=str(error))
   result=subprocess.run([str(a.binary.resolve()),str(path)],capture_output=True,text=True)
   cases.append(dict(name=name,protocol=protocol,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference=original,native=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [Path(__file__),Path(__file__).with_name('probe_torch_archive.cpp'),a.site/'torch/serialization.py',a.site/'torch/_weights_only_unpickler.py']}
a.output.write_text(json.dumps(dict(schema=1,scope='CPU-only own-data feasibility, no exact tensor or complete loader acceptance',stage3_registry_complete=False,trained_model_weights=0,max_tensor_elements=12,sources=sources,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),cases=cases),indent=2)+'\n')
print('Recorded',len(cases),'own archive-reader feasibility cases; no trained weights or CUDA')
