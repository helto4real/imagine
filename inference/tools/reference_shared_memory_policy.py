#!/usr/bin/env python3
"""Frozen coordinator policy matrix; metadata only, no Torch or model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
from itertools import product
states=['DISABLED','NO_VRAM','LOW_VRAM','NORMAL_VRAM','HIGH_VRAM','SHARED']
for index,name in enumerate(states):setattr(V,name,index)
def policy(state='NORMAL_VRAM',smart=True,available=True,reserve=400*1024*1024,ratio=0.):
 return dict(state=state,disable_smart_memory=not smart,lowvram_available=available,reserved=reserve,minimum_weight_ratio=ratio)
def apply(cfg):
 namespace.update(vram_state=getattr(V,cfg['state']),DISABLE_SMART_MEMORY=cfg['disable_smart_memory'],lowvram_available=cfg['lowvram_available'],EXTRA_RESERVED_VRAM=cfg['reserved'],MIN_WEIGHT_MEMORY_RATIO=cfg['minimum_weight_ratio'])
records=[]
original_free=namespace['get_free_memory']
def observed_free(device,torch_free_too=False):
 events.append(['free-memory',str(device)])
 return original_free(device,torch_free_too)
namespace['get_free_memory']=observed_free
for state,smart,dynamic,dev,required,torch_unused in product(states,[False,True],[False,True],[None,'cpu','cuda:0'],[0.,999_999_150.,1e30],[0.,249_999_751.]):
 reset();cfg=policy(state,smart);apply(cfg);capacity[0]=1_000_000_000;unused[0]=torch_unused
 add('a',400,400,retained=2,dynamic=dynamic);add('b',500,200,dynamic=dynamic);add('c',300,300,dynamic=dynamic);add('cpu',100,100,device='cpu',dynamic=dynamic)
 initial=snapshot();step=run('free-policy','free',required=required,keep=['b'],device=dev,dynamic=dynamic,pins=123)
 records.append(dict(name='free-policy',policy=cfg,dynamic_models=dynamic,initial=initial,capacity=capacity[0],unused=unused[0],steps=[step]))
for state,available,full,ratio,reserve,dev in product(states,[False,True],[False,True],[0.,.4],[0.,400*1024*1024,1024*1024*1024],['cpu','cuda:0']):
 reset();cfg=policy(state,True,available,reserve,ratio);apply(cfg)
 capacity[0]=1_500_000_000
 owners['a']=Patcher('a',600_000_000,'a',device=dev);owners['b']=Patcher('b',800_000_000,'b')
 step=run('load-policy','load',models=['a','b','a'],full=full,force=True,memory=1_200_000_000,minimum=900_000_000)
 records.append(dict(name='cpu'if dev=='cpu'else'load-policy',policy=cfg,initial=[],capacity=capacity[0],unused=0,steps=[step]))
for state,smart in product(states,[False,True]):
 reset();cfg=policy(state,smart);apply(cfg)
 owners['a']=Patcher('a',600_000_000,'a');owners['b']=Patcher('b',800_000_000,'b')
 steps=[run('first','load',models=['a']),run('reuse','load',models=['a'],memory=9_000_000_000,minimum=8_800_000_000)]
 records.append(dict(name='reuse',policy=cfg,initial=[],capacity=capacity[0],unused=0,steps=steps))
a.output.write_text(json.dumps(dict(schema=1,source=dict(path=str(path),sha256=digest),safety=dict(torch_imported=False,model_weights_allocated=False,logical_retention_counts=True),cases=records),separators=(',',':'))+'\n')
print('Captured',len(records),'original coordinator policy sequences,',sum(len(c['steps'])for c in records),'actions; no tensors',flush=True)
