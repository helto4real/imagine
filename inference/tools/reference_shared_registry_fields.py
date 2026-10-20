#!/usr/bin/env python3
"""Original patcher registries over live instance fields; own metadata only."""
import argparse,ast,hashlib,json,random,sys
from pathlib import Path
from typing import Callable
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/model_patcher.py';digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/model_patcher.py')
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ModelPatcher')
names=['add_callback','add_callback_with_key','remove_callbacks_with_key','get_callbacks','get_all_callbacks','add_wrapper','add_wrapper_with_key','remove_wrappers_with_key','get_wrappers','get_all_wrappers']
ns={'Callable':Callable};exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'),ns)
Subject=type('ModelPatcher',(),{n:ns[n] for n in names})
class Token:
    def __init__(self,index):self.index=index

def decode(wire):
    if isinstance(wire,dict):return tuple(map(decode,wire['tuple']))
    return wire

def snap(value):
    if isinstance(value,Token):return {'token':value.index}
    if isinstance(value,dict):return {'dict':[[snap(k),snap(v)] for k,v in value.items()]}
    if isinstance(value,list):return {'list':list(map(snap,value))}
    if isinstance(value,tuple):return {'tuple':list(map(snap,value))}
    return value

cases=[]
for category in ['callbacks','wrappers']:
    for seed in range(12):
        owner=Subject();owner.callbacks={};owner.wrappers={};handles={};roots=[];tokens=[Token(i) for i in range(8)];steps=[]
        stem='callback' if category=='callbacks' else 'wrapper'
        def act(op,kind='first',key=None,token=0,handle='live'):
            error=None;out=None
            try:
                if op=='add':getattr(owner,'add_'+stem+'_with_key')(decode(kind),decode(key),tokens[token])
                elif op=='plain':getattr(owner,'add_'+stem)(decode(kind),tokens[token])
                elif op=='get':handles[handle]=getattr(owner,'get_'+category)(decode(kind),decode(key))
                elif op=='all':handles[handle]=getattr(owner,'get_all_'+category)(decode(kind))
                elif op=='remove':getattr(owner,'remove_'+category+'_with_key')(decode(kind),decode(key))
                elif op=='append':handles[handle].append(tokens[token])
                elif op=='replace':
                    if hasattr(owner,category):roots.append(getattr(owner,category))
                    setattr(owner,category,{})
                elif op=='delete':delattr(owner,category)
                elif op=='bad-root':setattr(owner,category,7)
                elif op=='bad-list':getattr(owner,category)[decode(kind)]={decode(key):7}
                else:raise AssertionError(op)
            except (AttributeError,TypeError,KeyError,RuntimeError) as exception:error=[type(exception).__name__,str(exception)]
            steps.append(dict(op=op,kind=kind,key=key,token=token,handle=handle,error=error,
                fields=snap(vars(owner)),roots=list(map(snap,roots)),handles={k:snap(v) for k,v in handles.items()},
                aliases=[[x is y for y in handles.values()] for x in handles.values()]))
        pair={'tuple':['raw',None]}
        act('plain');act('add',key=pair,token=1);act('get');act('get',handle='same');act('all',handle='flat')
        act('get',key='missing',handle='miss1');act('get',key='missing',handle='miss2');act('append',handle='miss1',token=2)
        act('replace');act('append',token=3);act('plain',token=4);act('get',handle='new');act('remove');act('append',handle='new',token=5)
        act('add',kind=pair,key=pair,token=6);act('get',kind=pair,key=pair,handle='tuple')
        act('add',key=[],token=7);act('get',key=[]);act('remove',key=[]);act('add',kind=[],key='missing');act('get',kind=[])
        act('bad-list',kind='broken',key='slot');act('add',kind='broken',key='slot');act('all',kind='broken',handle='bad');act('remove',kind='broken',key='slot')
        act('delete');act('plain');act('get');act('all');act('replace')
        act('bad-root');act('plain');act('get');act('all');act('replace')
        rng=random.Random(seed)
        for i in range(40):
            op=rng.choice(['add','get','all','remove','replace','append']);kind=rng.choice(['first','second',pair]);key=rng.choice([None,'a',pair]);token=rng.randrange(8)
            act(op,kind,key,token,handle=rng.choice(list(handles)) if op=='append' else 'random'+str(i))
        cases.append(dict(category=category,seed=seed,steps=steps))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original field-registry histories,',sum(len(c['steps']) for c in cases),'observations; zero tensors/weights')
