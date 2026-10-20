#!/usr/bin/env python3
"""Original constant conditioning over shape metadata; no Tensor/model allocation."""
import argparse,ast,hashlib,json,math,logging,sys
from itertools import product
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.site));import torch
environment=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'environment.json').read_text())['torch_source_version']
assert torch.__version__==environment['__version__']and torch.version.git_version==environment['git_version'],'Original Torch reference changed'
source=a.comfy/'comfy/conds.py';manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());expected=next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/conds.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
tree=ast.parse(source.read_text());ns=dict(torch=torch,math=math,logging=logging);body=[v for v in tree.body if isinstance(v,(ast.FunctionDef,ast.ClassDef))and v.name in ['is_equal','CONDRegular','CONDConstant']];assert len(body)==3;exec(compile(ast.Module(body,type_ignores=[]),str(source),'exec'),ns)
context_source=a.comfy/'comfy/context_windows.py';context_sha=next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/context_windows.py');assert hashlib.sha256(context_source.read_bytes()).hexdigest()==context_sha
context_cls=next(v for v in ast.parse(context_source.read_text()).body if isinstance(v,ast.ClassDef)and v.name=='IndexListContextHandler');guide=next(v for v in context_cls.body if isinstance(v,ast.FunctionDef)and v.name=='_get_guide_entries');exec(compile(ast.Module([guide],type_ignores=[]),str(context_source),'exec'),ns)
def shape(kind,dims):return torch.Size(dims)if kind=='size'else tuple(dims)if kind=='tuple'else list(dims)
def payload(p):
 if p['root']=='bare':return shape(p['kind'],p['dims'])
 values=[shape(p['kind'],p['dims'])]+([shape(p['kind'],[3,4])]if p['count']==2 else[])if p['count']else[]
 return values if p['root']=='list'else tuple(values)
def describe(value):
 if isinstance(value,(list,tuple)):return dict(type=type(value).__name__,items=[describe(v)for v in value])
 return dict(type='scalar',value=value)
forms=[]
for root in ['bare','list','tuple']:
 for count in [1]if root=='bare'else[0,1,2]:
  for kind,dims in [('size',[])]if count==0 else product(['size','tuple','list'],[[],[1],[1,2],[1,0,-2]]):forms.append(dict(root=root,count=count,kind=kind,dims=dims))
cases=[]
for left,right in product(forms,forms):
 x=payload(left);y=payload(right);c=ns['CONDConstant'](x);d=ns['CONDConstant'](y);processed=c.process_cond(4);copied=c._copy_with(c.cond)
 cases.append(dict(left=left,right=right,can_concat=c.can_concat(d),same_objects=x is y,left_value=describe(x),right_value=describe(y),left_repr=repr(x),process_same_payload=processed.cond is x,process_same_wrapper=processed is c,copy_same_payload=copied.cond is x,concat_same_payload=c.concat([d])is x,size=c.size()))
 found=ns['_get_guide_entries']([[{'model_conds':{'guide_attention_entries':c}}]])
 cases[-1].update(truth=bool(x),guide_same_payload=found is x,guide_is_none=found is None)
representations=[dict(dims=v,value=describe(torch.Size(v)),repr=repr(torch.Size(v)))for v in [[],[0],[1],[-1],[1,2,3],[2**63-1],[-2**63],[-2**63,0,2**63-1]]]
for case in representations:
 try:getattr(torch.Size(case['dims']),'cond')
 except AttributeError as e:case['attribute_error']=str(e)
a.output.write_text(json.dumps(dict(schema=1,source_sha256=expected,context_source_sha256=context_sha,torch_version=torch.__version__,torch_git=torch.version.git_version,trained_model_weights=0,tensors_created=0,cases=cases,representations=representations),separators=(',',':'))+'\n');print('Captured',len(cases),'constant metadata pairs and',len(representations),'Size representations; no fixture tensors or models',flush=True)
