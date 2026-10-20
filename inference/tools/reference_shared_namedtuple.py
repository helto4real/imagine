#!/usr/bin/env python3
"""Namedtuple source semantics at COND equality and Kitchen metadata boundaries."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from collections import namedtuple
import logging
ns['logging']=logging
definitions('comfy/conds.py',['is_equal'])
from comfy_kitchen.constraints import ParamConstraint,FunctionConstraints,validate_function_call
from comfy_kitchen.backends import cuda
for path in ['comfy_kitchen/constraints.py','comfy_kitchen/backends/cuda/__init__.py']:
 sha=hashlib.sha256((a.site/path).read_bytes()).hexdigest();assert sha==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path);sources[path]=sha
Pair=namedtuple('Pair','a b');OtherPair=namedtuple('Pair','a b');Triple=namedtuple('Triple','a b c');Empty=namedtuple('Empty','')
values=[('pair_a',Pair(1,2)),('pair_a_2',Pair(1,2)),('pair_a_diff',Pair(1,3)),('pair_b',OtherPair(1,2)),('triple',Triple(3,3,3)),('bad_triple',Triple(0,3,3)),('pair_list',Pair([1],2)),('empty',Empty()),('tuple',(1,2)),('list',[1,2])]
def encode(v):
 if isinstance(v,tuple)and hasattr(v,'_fields'):return dict(type='namedtuple',class_id='pair_a'if type(v)is Pair else'pair_b'if type(v)is OtherPair else'triple'if type(v)is Triple else'empty',fields=[[k,encode(x)]for k,x in zip(v._fields,v,strict=True)])
 if isinstance(v,(list,tuple)):return dict(type='list'if isinstance(v,list)else'tuple',values=[encode(x)for x in v])
 return dict(type='scalar',value=v)
constant=FunctionConstraints(params={'x':ParamConstraint(dtypes=frozenset([int]))})
na=cuda._build_constraints()['na3d']
metadata=[]
for key,v in values:
 item=dict(key=key,value=encode(v),typename=type(v).__name__,representation=repr(v),truth=bool(v),length=len(v))
 for name,constraints,kwargs in [('dtype',constant,{'x':v}),('na',na,{'kernel_size':v})]:
  try:
   r=validate_function_call(constraints,kwargs,(8,9));item[name]=dict(success=r.success,parameter=r.failed_param,reason=r.failure_reason)
  except Exception as e:item[name]=dict(error=type(e).__name__,message=str(e))
 metadata.append(item)
equality=[dict(a=ak,b=bk,cond_equal=ns['is_equal'](av,bv))for ak,av in values for bk,bv in values]
a.output.write_text(json.dumps(dict(schema=1,sources=sources,values=metadata,equality=equality),separators=(',',':'))+'\n')
print(f'Captured {len(values)} namedtuple metadata cases and {len(equality)} original COND equalities; tensors/models/GPU allocations: 0')
