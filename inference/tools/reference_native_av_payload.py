#!/usr/bin/env python3
"""Original contiguous H3 payload assembly block; opaque synthetic owners only.

Does not execute BaseModel.extra_conds, learned text projection, mask pooling or
PackedLayout. Those are explicitly separate caller gates.
"""
import argparse,ast,hashlib,itertools,json,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.comfy/'comfy/model_base.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MiniMaxH3');func=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='extra_conds')
start=next(i for i,n in enumerate(func.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='payload'for t in n.targets));stop=next(i for i,n in enumerate(func.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='denoise_mask'for t in n.targets));audio=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='audio_scale')
body=func.body[start:stop]+[ast.Return(ast.Name(id='payload',ctx=ast.Load()))]
wrapper=ast.FunctionDef(name='payload',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='kwargs')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[]);ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[audio,wrapper],type_ignores=[])),str(source),'exec'),ns)
owners={name:object()for name in ['tags','v1','v2','a1','a2']}
def decode(v):
 if isinstance(v,dict):
  if list(v)==['owner']:return owners[v['owner']]
  return{k:decode(x)for k,x in v.items()}
 if isinstance(v,list):return[decode(x)for x in v]
 return v
def encode(v):
 for name,obj in owners.items():
  if v is obj:return dict(owner=name)
 if isinstance(v,dict):return{k:encode(x)for k,x in v.items()}
 if isinstance(v,list):return[encode(x)for x in v]
 return v
filled_k=[dict(latent={'owner':'v1'},audio_latent=None),dict(audio_latent={'owner':'a1'}),dict(latent=None),{}]
filled_r=[dict(latent=None),dict(latent={'owner':'v2'},audio_latent={'owner':'a2'}),dict(audio_latent=None),{}]
cases=[]
for tag,k,r,aug,shape,seed in itertools.product([False,True],range(4),range(4),range(3),[None,[(1,2)],[(1,2),(1,3)]],[0,None]):
 spec={}
 if tag:spec['minimax_token_tags']={'owner':'tags'}
 for name,mode,filled in [('minimax_keyframes',k,filled_k),('minimax_refs',r,filled_r)]:
  if mode:spec[name]=None if mode==1 else([]if mode==2 else filled)
 if aug:spec.update(minimax_visual_cond_noise_aug=None if aug==1 else 0.0,minimax_audio_cond_noise_aug=None if aug==1 else False)
 spec['seed']=seed;kwargs=decode(spec)
 model=types.SimpleNamespace(latent_shapes=shape,model_sampling=types.SimpleNamespace(audio_scale=4.0));model.audio_scale=lambda:ns['audio_scale'](model)
 output=ns['payload'](model,kwargs)
 cases.append(dict(input=spec,latent_shapes=shape,output=encode(output),keys=list(output),forwarded={name:output[target]is kwargs[name]for name,target in [('minimax_token_tags','text_token_tags'),('minimax_keyframes','keyframes'),('minimax_refs','refs')]if name in kwargs and kwargs[name]is not None}))
a.output.write_text(json.dumps(dict(schema=1,sources={'comfy/model_base.py':hashlib.sha256(source.read_bytes()).hexdigest()},scope='contiguous payload assembly and actual audio_scale; not extra_conds learned projection, masks or layout',trained_weight_tensors=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original payload histories with opaque owner identity')
