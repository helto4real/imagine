#!/usr/bin/env python3
"""Unmodified context metadata methods, own bounded values, no Torch/models."""
import argparse,ast,hashlib,json
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.comfy/'comfy/context_windows.py'
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
expected=next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/context_windows.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Original context source changed'
tree=ast.parse(source.read_text());cls=next(v for v in tree.body if isinstance(v,ast.ClassDef)and v.name=='IndexListContextHandler')
names=['_get_latent_shapes','_get_guide_entries','_get_keyframe_idxs'];ns={}
for f in cls.body:
 if isinstance(f,ast.FunctionDef)and f.name in names:
  exec(compile(ast.Module([f],type_ignores=[]),str(source),'exec'),ns)
class Cond:
 def __init__(self,value):self.cond=value
def encode(v):
 if v is None or isinstance(v,(bool,int,float,str)):return dict(type='scalar',value=v)
 if isinstance(v,dict):return dict(type='dict',items=[[k,encode(v)]for k,v in v.items()])
 return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
def run(method,profile,style,layout):
 calls=[];payload=['own',3];alternate=['alternate',7]
 class Descriptor:
  @property
  def cond(self):
   i=len(calls);calls.append(i)
   if style=='missing':raise AttributeError('own missing cond')
   if style=='error':raise ValueError('own descriptor failure')
   if style=='error-second'and i==1:raise ValueError('own second descriptor failure')
   if style=='none':return None
   if style=='false':return False
   if style=='empty':return []
   if style=='changing':return payload if i%2==0 else alternate
   return payload
 value=Descriptor()if style!='plain'else 3
 key={'_get_latent_shapes':'latent_shapes','_get_guide_entries':'guide_attention_entries','_get_keyframe_idxs':'keyframe_idxs'}[method]
 model={key:value}
 if profile.startswith('bad-model-'):
  model={'none':None,'int':3,'empty-list':[],'list':['latent_shapes'],'tuple':('latent_shapes',),'text':'own latent_shapes','empty-text':'own','bool':False}[profile[len('bad-model-'):]]
 entry={'model_conds':model}
 rows=[[entry,{'model_conds':{key:Cond(alternate)}}]]
 if profile=='skip-none':rows=[None,*rows]
 elif profile=='first-empty':rows=[[{},*rows[0]]]
 elif profile=='missing':rows=[[{}]]
 elif profile=='empty':rows=[]
 elif profile=='none-value':model[key]=None
 if layout=='tuple':rows=tuple(tuple(v)if v is not None else None for v in rows)
 try:result=ns[method](rows);out=dict(output=encode(result),same_payload=result is payload,same_alternate=result is alternate)
 except Exception as e:out=dict(error=type(e).__name__,message=str(e))
 return dict(params=dict(method=method,profile=profile,style=style,layout=layout),calls=calls,**out)
cases=[run(*v)for v in product(names,['normal','skip-none','first-empty','missing','empty','none-value'],['normal','missing','error','error-second','none','false','empty','changing','plain'],['list','tuple'])]
cases += [run(*v)for v in product(names,['bad-model-'+v for v in ['none','int','empty-list','list','tuple','text','empty-text','bool']],['normal'],['list','tuple'])]
a.output.write_text(json.dumps(dict(schema=1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'metadata cases, no Torch import or tensors')
