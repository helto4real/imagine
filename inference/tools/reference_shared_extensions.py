#!/usr/bin/env python3
"""Static-source callback, wrapper and nested option alias oracle; no tensors."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Callable
from reference_paths import reference_dir

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
path='comfy/patcher_extension.py'
expected=next(e['sha256'] for e in manifest if e['root']=='comfy' and e['path']==path)
source=a.comfy/path
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
ns=dict(Callable=Callable)
exec(compile(ast.parse(source.read_text()),str(source),'exec'),ns)

def record(call):
 try:return dict(output=call())
 except Exception as e:return dict(error=type(e).__name__,message=str(e))

wrappers=[]
for class_obj in [None,'own-instance']:
 for idx in [0,1,2,3]:
  trace=[]
  def original(x,offset=0):
   trace.append(['original',x,offset])
   return x*3+offset
  def first(ex,x,offset=0):
   trace.append(['before-first',ex.idx,ex.class_obj,x])
   out=ex(x+1,offset=offset+2)
   trace.append(['after-first',out])
   return out+10
  def second(ex,x,offset=0):
   trace.append(['before-second',ex.idx,ex.class_obj,x])
   out=ex(x+2,offset=offset)+ex(x+3,offset=offset)
   trace.append(['after-second',out])
   return out+20
  ex=ns['WrapperExecutor'](original,class_obj,[first,second],idx)
  out=record(lambda:ex.execute(4,offset=5))
  wrappers.append(dict(class_obj=class_obj,idx=idx,trace=trace,**out))
last=ns['WrapperExecutor'].new_executor(lambda:None,[])
next_error=record(last)
registries=[]
for variant in ['callback','wrapper']:
 for model_options in [False,True]:
  options={}
  add=ns['add_'+variant+'_with_key']
  all_values=ns['get_all_'+variant+'s']
  keyed=ns['get_'+variant+'s_with_key']
  for key,value in [(None,1),('b',2),('a',3),('b',4),(None,5)]:add('own-type',key,value,options,is_model_options=model_options)
  registries.append(dict(variant=variant,model_options=model_options,all=all_values('own-type',options,is_model_options=model_options),none=keyed('own-type',None,options,is_model_options=model_options),b=keyed('own-type','b',options,is_model_options=model_options),missing=keyed('missing',None,options,is_model_options=model_options)))

options=[]
for copy_left in [False,True]:
 left=dict(nested=dict(values=[1,2],keep='own'),list=[[3,4]],scalar=1)
 right=dict(nested=dict(values=[5],new=6),list=[[7]],scalar=9,added=True)
 before=json.loads(json.dumps(left))
 child=left['list'][0]
 out=ns['merge_nested_dicts'](left,right,copy_dict1=copy_left)
 options.append(dict(copy_left=copy_left,left_before=before,right=right,output=out,left_after=left,root_same=out is left,nested_same=out['nested'] is left['nested'],list_same=out['list'] is left['list'],child_same=out['list'][0] is child))
copied_input=dict(nested=dict(values=[[1,2]]),tuple=(1,2))
copied=ns['copy_nested_dicts'](copied_input)
copy_aliases=dict(root_same=copied is copied_input,nested_same=copied['nested'] is copied_input['nested'],list_same=copied['nested']['values'] is copied_input['nested']['values'],child_same=copied['nested']['values'][0] is copied_input['nested']['values'][0],tuple_same=copied['tuple'] is copied_input['tuple'])
collisions=[]
for left,right in [(dict(x=3),dict(x=[1])),(dict(x=[1]),dict(x=dict(y=2)))]:
 collisions.append(dict(left=left,right=right,**record(lambda:ns['merge_nested_dicts'](left,right))))
a.output.write_text(json.dumps(dict(schema=1,source_sha256=expected,callbacks=[getattr(ns['CallbacksMP'],k)for k in vars(ns['CallbacksMP'])if k.startswith('ON_')],wrappers_registry=[getattr(ns['WrappersMP'],k)for k in vars(ns['WrappersMP'])if k.isupper()],wrappers=wrappers,next_error=next_error,registries=registries,options=options,copy_aliases=copy_aliases,collisions=collisions),indent=2)+'\n')
print('Captured original wrapper stacks, registration order, option mutations, aliases and collision errors; no model imports')
