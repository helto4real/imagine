#!/usr/bin/env python3
"""Original COND wrapper assignments/copies, own weak objects; no Torch import."""
import argparse,ast,hashlib,json,sys,weakref
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());source=a.comfy/'comfy/conds.py';expected=next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/conds.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
names=['CONDRegular','CONDNoiseShape','CONDCrossAttn','CONDConstant','CONDList'];tree=ast.parse(source.read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name in names],type_ignores=[]),str(source),'exec'),ns)
class Token:
 def __init__(self,label):self.label=label
def encode(v):
 if isinstance(v,Token):return['token',v.label]
 if isinstance(v,list):return['list',[encode(x)for x in v]]
 if isinstance(v,dict):return['dict',[[k,encode(x)]for k,x in v.items()]]
 raise AssertionError(type(v))
cases=[]
for name in names:
 for style in ['token','list','dict']:
  refs=[]
  def token():
   value=Token(len(refs));refs.append(weakref.ref(value));return value
  def payload():return token()if style=='token'else[token(),token()]if style=='list'else{'inner':token(),'other':token()}
  holders={'a':ns[name](payload())};holders['b']=holders['a'];holders['c']=holders['a']._copy_with(holders['a'].cond)
  def snapshot():return dict(holders=[[k,type(v).__name__,encode(v.cond)]for k,v in holders.items()],same_wrappers=[[a is b for b in holders.values()]for a in holders.values()],same_payloads=[[a.cond is b.cond for b in holders.values()]for a in holders.values()],live=[ref()is not None for ref in refs])
  initial=snapshot();events=[]
  for op,target,source_name in [('set','b',None),('copy','d','a'),('drop','a',None),('set','c',None),('alias','e','d'),('mutate','d',None),('set','e',None),('drop','b',None),('drop','d',None),('drop','c',None),('drop','e',None)]:
   if op=='set':holders[target].cond=payload()
   elif op=='copy':holders[target]=holders[source_name]._copy_with(holders[source_name].cond)
   elif op=='alias':holders[target]=holders[source_name]
   elif op=='drop':del holders[target]
   elif style=='token':holders[target].cond.label+=1000
   elif style=='list':holders[target].cond[0]=token()
   else:holders[target].cond['inner']=token()
   events.append(dict(action=dict(op=op,target=target,source=source_name),post=snapshot()))
  assert not any(ref()is not None for ref in refs)
  cases.append(dict(name=name,style=style,initial=initial,events=events))
reentrant=[]
for name in names:
 events=[];state={}
 class Finalizer:
  def __del__(self):
   events.append(['before',state['cond'].cond]);state['cond'].cond='from-finalizer';events.append(['after',state['cond'].cond])
 state['cond']=ns[name](Finalizer());alias=state['cond'];alias.cond='replacement'
 reentrant.append(dict(name=name,events=events,output=alias.cond));del alias;state.clear()
a.output.write_text(json.dumps(dict(schema=1,source_sha256=expected,trained_model_weights=0,torch_python_imported=False,cases=cases,reentrant=reentrant),separators=(',',':'))+'\n')
print('Captured',len(cases),'COND owner histories and',len(reentrant),'reentrant assignments; no tensors or Torch import')
