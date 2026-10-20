#!/usr/bin/env python3
"""Whole unmodified context preparation wrapper; bounded own shapes, no Torch."""
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
tree=ast.parse(source.read_text());f=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='_prepare_sampling_wrapper');ns=dict(torch=NS(Tensor=object));exec(compile(ast.Module([f],type_ignores=[]),str(source),'exec'),ns);fn=ns[f.name]
def encode(v):
 if v is None or isinstance(v,(bool,int,float,str)):return dict(type='scalar',value=v)
 if isinstance(v,dict):return dict(type='dict',items=[[k,encode(v)]for k,v in v.items()])
 return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
shapes={'standard':[1,4,16,2,2],'packed':[1,1,256],'short':[2,3,8],'empty':[],'scalar':3,'none':None,'text':'own','dict':{'a':1,'b':3},'bool-packed':[1,True,9],'float':[1,3.5,8.25],'wide':[1,2**63,3]}
def run(shape_name,layout,dim,length,options,call):
 shape=shapes[shape_name];shape=shape.copy()if isinstance(shape,(list,dict))else shape
 if layout=='tuple'and isinstance(shape,list):shape=tuple(shape)
 model=['model'];conds=['conds'];token=['return'];events=[]
 opts={}if options=='empty'else{'context_handler':None}if options=='none-handler'else{'context_handler':3}if options=='bad-handler'else{'context_handler':NS(dim=dim,context_length=length)}
 args=[model,shape,conds];kwargs={'model_options':opts,'own':'extra'}
 if options=='missing':kwargs.pop('model_options')
 if options=='none-options':kwargs['model_options']=None
 if options=='bad-options':kwargs['model_options']=3
 if call=='keywords':args=[];kwargs.update(model=model,noise_shape=shape,conds=conds)
 if call=='mixed':args=[model];kwargs.update(noise_shape=shape,conds=conds)
 if call=='extra':args+=['extra',7]
 if call=='duplicate':kwargs['model']=model
 if call=='receiver':kwargs['executor']=3
 if call=='missing-all':args=[]
 if call=='missing-shape':args=[model];kwargs['conds']=conds
 if call=='missing-conds':args=[model,shape]
 def executor(*args,**kwargs):
  events.append(dict(shape=encode(args[1]),same_shape=args[1]is shape,same_model=args[0]is model,same_conds=args[2]is conds,extra=encode(args[3:]),keys=list(kwargs),same_options=kwargs.get('model_options')is opts,own=kwargs.get('own')))
  return token
 try:out=fn(executor,*args,**kwargs);result=dict(same_return=out is token)
 except Exception as e:result=dict(error=type(e).__name__,message=str(e))
 return dict(params=dict(shape=shape_name,layout=layout,dim=dim,length=length,options=options,call=call),input=encode(shape),events=events,post=encode(shape),**result)
cases=[run(*v)for v in product(shapes,['list','tuple'],[-8,-1,0,1,2,7],[-1,1,4,12],['handler'],['positional','keywords'])]
cases += [run(*v)for v in product(['standard','packed','scalar'],['list','tuple'],[2],[4],['empty','none-handler','bad-handler','missing','none-options','bad-options'],['positional','mixed','extra','duplicate','receiver','missing-all','missing-shape','missing-conds'])]
a.output.write_text(json.dumps(dict(schema=1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole preparation-wrapper cases without Torch or model weights')
