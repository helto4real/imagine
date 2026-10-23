#!/usr/bin/env python3
"""Execute only the original pure estimators with shape stubs; no Torch/CUDA/imports."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
COMFY=Path('/home/thhel/git/ComfyUI')
paths=[COMFY/'comfy'/name for name in ('model_base.py','sampler_helpers.py','supported_models.py')]
base,helpers,supported=[ast.parse(p.read_text())for p in paths]
baseclass=next(n for n in base.body if isinstance(n,ast.ClassDef)and n.name=='BaseModel')
h3class=next(n for n in base.body if isinstance(n,ast.ClassDef)and n.name=='MiniMaxH3')
assert not any(isinstance(n,ast.FunctionDef)and n.name in ('memory_required','extra_conds_shapes')for n in h3class.body)
h3config=next(n for n in supported.body if isinstance(n,ast.ClassDef)and n.name=='MiniMaxH3')
factor=next(ast.literal_eval(n.value)for n in h3config.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='memory_usage_factor'for t in n.targets))
assert factor==0.114
functions=[next(n for n in baseclass.body if isinstance(n,ast.FunctionDef)and n.name==name)for name in ('memory_required','extra_conds_shapes')]
functions.append(next(n for n in helpers.body if isinstance(n,ast.FunctionDef)and n.name=='estimate_memory'))
management=types.SimpleNamespace(xformers_enabled=lambda:False,pytorch_attention_flash_attention=lambda:True,dtype_size=lambda dtype:2)
namespace=dict(math=math,collections=collections,comfy=types.SimpleNamespace(model_management=management))
exec(compile(ast.Module(body=functions,type_ignores=[]),'<selected original pure memory estimators>','exec'),namespace)
class Model:
    memory_usage_factor=factor
    memory_usage_factor_conds=()
    memory_required=namespace['memory_required']
    extra_conds_shapes=namespace['extra_conds_shapes']
    def get_dtype_inference(self):return 'bf16'

records=[]
for width,height,frames,vt,at in [(384,256,39,12,65),(768,512,124,37,207),(512,768,124,37,207)]:
    elements=24*vt*(height//16)*(width//16)+32*2*at
    model=types.SimpleNamespace(model=Model())
    memory,minimum=namespace['estimate_memory'](model,[1,1,elements],{'positive':[{}]})
    assert memory==2*minimum
    reserve=400*1024**2;floor=.8*1024**3+reserve
    old=max(floor,memory+reserve);new=max(floor,minimum+reserve)
    records.append(dict(width=width,height=height,frames=frames,packed_elements=elements,requested=memory,minimum=minimum,old_minimum_with_reserve=old,new_minimum_with_reserve=new))
assert records[0]['old_minimum_with_reserve']==records[0]['new_minimum_with_reserve']
assert abs(records[1]['new_minimum_with_reserve']/1024**3-3.4570825)<1e-10
result=dict(result='passed',scope='Original pure estimator, synthetic shapes; no model or GPU execution',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},cases=records)
out=ROOT/'comfyui/h3-memory-20261023/source-estimator.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print('PASS: original H3 estimator, three packed AV geometries; no Torch/CUDA imported')
