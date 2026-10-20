#!/usr/bin/env python3
"""Original wrapper registration helpers, with explicit model effects only."""
import argparse,ast,hashlib,json
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Callable
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={}
ns=dict(ModelPatcher=object,Callable=Callable,torch=NS(Tensor=object))
for relative,names in [('comfy/patcher_extension.py',['WrappersMP']),('comfy/context_windows.py',['_prepare_sampling_wrapper','_sampler_sample_wrapper','create_prepare_sampling_wrapper','create_sampler_sample_wrapper'])]:
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,(ast.ClassDef,ast.FunctionDef))and node.name in names:exec(compile(ast.Module([node],type_ignores=[]),str(path),'exec'),ns)
ns['comfy']=NS(patcher_extension=NS(WrappersMP=ns['WrappersMP']))
cases=[]
for count in range(1,5):
 for sequence,mode,models in product(product(['prepare','sampler'],repeat=count),['normal','error-first','error-second'],[1,2]):
  events=[];functions=[]
  class Model:
   def __init__(self,i):self.i=i
   def add_wrapper_with_key(self,kind,key,fn):
    name='_prepare_sampling_wrapper'if kind=='prepare_sampling'else'_sampler_sample_wrapper'
    events.append(dict(model=self.i,kind=kind,key=key,name=name,same_canonical=fn is ns[name],same_previous=[fn is v for v in functions]));functions.append(fn)
    if mode=='error-first'and len(events)==1 or mode=='error-second'and len(events)==2:raise ValueError('own wrapper registration failure')
  owners=[Model(i)for i in range(models)]
  try:
   returns=[ns['create_prepare_sampling_wrapper'if action=='prepare'else'create_sampler_sample_wrapper'](owners[i%models])for i,action in enumerate(sequence)];outcome=dict(ok=True,returns_none=[v is None for v in returns])
  except Exception as e:outcome=dict(error=type(e).__name__,message=str(e))
  cases.append(dict(params=dict(sequence=sequence,mode=mode,models=models),events=events,**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,torch_python_imported=False,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original helper registrations, canonical function identities and partial failures; no Torch or tensors',flush=True)
