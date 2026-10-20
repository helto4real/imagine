#!/usr/bin/env python3
"""Original patcher callback/wrapper list ownership; standard library only."""
import argparse,ast,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Callable
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];relative='comfy/model_patcher.py';path=a.comfy/relative
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(n['sha256']for n in manifest if n['root']=='comfy'and n['path']==relative)
tree=ast.parse(path.read_text());model=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ModelPatcher')
names=['add_callback','add_callback_with_key','remove_callbacks_with_key','get_callbacks','get_all_callbacks','add_wrapper','add_wrapper_with_key','remove_wrappers_with_key','get_wrappers','get_all_wrappers']
methods=[n for n in model.body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(methods)==len(names)
Subject=type('OriginalRegistry',(),{});ns=dict(Callable=Callable)
exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),ns)
for name in names:setattr(Subject,name,ns[name])
clone=next(n for n in model.body if isinstance(n,ast.FunctionDef)and n.name=='clone')
loops=[n for n in clone.body if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Attribute)and n.iter.func.attr=='items'and isinstance(n.iter.func.value,ast.Attribute)and n.iter.func.value.attr in ['callbacks','wrappers']]
assert len(loops)==2
clone_code=compile(ast.Module(body=loops,type_ignores=[]),str(path),'exec')
class Token:
 def __init__(self,index):self.index=index
cases=[]
for category in ['callbacks','wrappers']:
 for seed in range(32):
  rng=random.Random(137+seed);tokens=[Token(i)for i in range(8)];roots={};handles={};steps=[]
  roots['original']=Subject();roots['original'].callbacks={};roots['original'].wrappers={}
  stem='callback'if category=='callbacks'else'wrapper'
  get='get_callbacks'if category=='callbacks'else'get_wrappers';all_name='get_all_'+category;remove='remove_'+category+'_with_key'
  def snapshot():
   return dict(roots={root:{kind:[t.index for t in getattr(value,all_name)(kind)]for kind in ['first','second','missing']}for root,value in roots.items()},handles={key:[t.index for t in value]for key,value in handles.items()},aliases=[[a is b for b in handles.values()]for a in handles.values()])
  def act(op,**kw):
   root=roots[kw.get('root','original')]
   if op=='add':getattr(root,'add_'+stem+'_with_key')(kw['kind'],kw['key'],tokens[kw['token']])
   elif op=='plain_add':getattr(root,'add_'+stem)(kw['kind'],tokens[kw['token']])
   elif op=='get':handles[kw['handle']]=getattr(root,get)(kw['kind'],kw['key'])
   elif op=='all':handles[kw['handle']]=getattr(root,all_name)(kw['kind'])
   elif op=='append':handles[kw['handle']].append(tokens[kw['token']])
   elif op=='pop':
    if handles[kw['handle']]:handles[kw['handle']].pop(0)
   elif op=='remove':getattr(root,remove)(kw['kind'],kw['key'])
   elif op=='clone':
    n=Subject();n.callbacks={};n.wrappers={};exec(clone_code,{},dict(self=root,n=n));roots['clone']=n
   steps.append(dict(op=op,**kw,expected=snapshot()))
  act('plain_add',kind='first',token=0);act('add',kind='first',key='a',token=1)
  act('get',kind='first',key=None,handle='live');act('get',kind='first',key=None,handle='same');act('all',kind='first',handle='flat')
  act('get',kind='missing',key='a',handle='absent1');act('get',kind='missing',key='a',handle='absent2');act('append',handle='absent1',token=2)
  act('clone');act('get',root='clone',kind='first',key=None,handle='cloned');act('append',handle='live',token=3)
  act('remove',kind='first',key=None);act('append',handle='live',token=4);act('plain_add',kind='first',token=5)
  for index in range(40):
   root=rng.choice(['original','clone']);kind=rng.choice(['first','second','missing']);key=rng.choice([None,'a','b']);token=rng.randrange(8)
   op=rng.choice(['add','remove','get','all','append','pop'])
   if op in ['append','pop']:act(op,handle=rng.choice(list(handles)),**({'token':token}if op=='append'else{}))
   elif op=='add':act(op,root=root,kind=kind,key=key,token=token)
   elif op=='get':act(op,root=root,kind=kind,key=key,handle='random'+str(index))
   elif op=='all':act(op,root=root,kind=kind,handle='random'+str(index))
   else:act(op,root=root,kind=kind,key=key)
  cases.append(dict(category=category,seed=seed,steps=steps))
a.output.write_text(json.dumps(dict(schema=1,sources={relative:digest},scope='unmodified getter/add/remove methods and the two clone registry loops only; not complete patcher clone',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'registry histories,',sum(len(c['steps'])for c in cases),'observations; no Torch or model weights',flush=True)
