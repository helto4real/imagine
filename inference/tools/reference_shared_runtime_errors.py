#!/usr/bin/env python3
"""Original default Torch C++ RuntimeError display, CPU only, tiny own tensors."""
import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
assert os.environ.get('TORCH_SHOW_CPP_STACKTRACES','0')=='0','This capture is the locked stack-disabled profile'
cases=[]
for name,call in [('view',lambda:torch.ones(2,3).view(5)),('mm',lambda:torch.ones(2,3).mm(torch.ones(4,2))),('min',lambda:torch.zeros(0).min()),('truth',lambda:torch.is_nonzero(torch.ones(2)))]:
 try:call()
 except Exception as error:cases.append(dict(name=name,error=type(error).__name__,message=str(error)))
 else:raise AssertionError(name)
source=a.site/'torch/include/torch/csrc/Exceptions.h'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_access=False,cpp_stacktraces_enabled=False,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),cases=cases),indent=2)+'\n');print('Captured',len(cases),'original complete RuntimeError messages')
