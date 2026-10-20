#!/usr/bin/env python3
"""Bypass metadata, shared constructor attributes and original weight movement."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_bypass.py')
exec(compile(loader.read_text().split('\ninputs=[];cases=[]\n',1)[0],str(loader),'exec'))
metadata=[];constructors=[];moves=[]
def reference_alias(a,b):
 # Inference ownership observations cover tensors, containers and None;
 # immutable scalar values are compared numerically/by value, not Python ids.
 if not (a is None or isinstance(a,(torch.Tensor,list,tuple,dict)))or not (b is None or isinstance(b,(torch.Tensor,list,tuple,dict))):return None
 return a is b
classes=[('Conv3dButLinear',[torch.nn.Linear],['Linear']),('Conv3dButConv1d',[torch.nn.Conv1d],['Conv1d']),('Conv2d',[torch.nn.Conv2d],['Conv2d']),('Conv3d',[torch.nn.Conv3d],['Conv3d']),('Mixed',[torch.nn.Conv2d,torch.nn.Conv1d],['Conv2d','Conv1d']),('CustomCONV3D',[],[]),('CustomConv2d',[],[]),('CustomConv1d',[],[]),('CustomConvolution',[],[]),('Plain',[],[])]
attributes=[{},dict(stride=2,padding=1,dilation=2,groups=2,kernel_size=3,in_channels=4,out_channels=8),dict(stride=[1,2],padding=(0,1),dilation=True,kernel_size=None,in_channels=None,out_channels=False)]
def module(name,bases,attrs):
 cls=type(name,tuple(bases or [torch.nn.Module]),{})
 result=object.__new__(cls);torch.nn.Module.__init__(result)
 for k,v in attrs.items():setattr(result,k,v)
 return result
for name,bases,instances in classes:
 for attrs in attributes:
  m=module(name,bases,attrs);info=ns['get_module_type_info'](m)
  keys=['stride','padding','dilation','kernel_size']
  metadata.append(dict(name=name,instances=instances,attrs=encode(attrs),output=encode(info),aliases=[[info[k]is attrs.get(n,object())for n in keys]for k in keys],sharing=[[info[k]is info[n]for n in keys]for k in keys]))
adapter=NS(weights=(),untouched=['own']);previous=None
original_info=ns['get_module_type_info'];observed=[]
def get_info(m):
 info=original_info(m);observed.append(info);return info
ns['get_module_type_info']=get_info
for name,bases,instances in classes:
 m=module(name,bases,attributes[1]);old_kwargs=getattr(adapter,'kw_dict',None)
 hook=ns['BypassForwardHook'](m,adapter,-.5)
 info=observed[-1]
 constructors.append(dict(name=name,instances=instances,attrs=encode(attributes[1]),multiplier=-.5,post=encode(vars(adapter)),new_kwargs=adapter.kw_dict is not old_kwargs,kw_aliases={k:reference_alias(adapter.kw_dict[k],info[k])for k in adapter.kw_dict},kernel_same=adapter.kernel_size is info['kernel_size']))
ns['get_module_type_info']=original_info
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
forms=['missing','none','tensor','list','tuple','empty_list','empty_tuple','dict','scalar','nested','quantized','tuple_quantized']
for source in ['cpu','cuda:0']:
 for target in ['cpu','cuda:0']:
  for module_kind in ['missing','none','torch.float32','torch.bfloat16','torch.float64','torch.float8_e4m3fn','torch.int8','torch.float16','quantized_int8_bfloat16']:
   for form in forms:
    t=data((3,2),source,torch.float32,3).t();aux=data((2,3),source,torch.float64,5);shared=dict(keep=[t])
    raw=(torch.arange(128,device=source)%31-15).to(torch.int8).reshape(2,64)
    q=QuantizedTensor(raw,'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.tensor(.03125,device=source),orig_dtype=torch.bfloat16,orig_shape=(2,64)))
    encoded_q=encode(q)
    weights=None if form in ['missing','none']else t if form=='tensor'else [t,t,None,shared,'own',aux]if form=='list'else (t,t,None,shared,'own',aux)if form=='tuple'else []if form=='empty_list'else ()if form=='empty_tuple'else shared if form=='dict'else .5 if form=='scalar'else [[t],(t,),shared]
    if form=='quantized':weights=q
    if form=='tuple_quantized':weights=(q,q,t,shared)
    adapter=NS()
    if form!='missing':adapter.weights=weights
    m=NS(forward=lambda *args,**kwargs:None)
    if module_kind!='missing':m.weight=None if module_kind=='none'else q if module_kind=='quantized_int8_bfloat16'else torch.empty((0,),device='meta',dtype=getattr(torch,module_kind.split('.')[-1]))
    hook=ns['BypassForwardHook'](m,adapter,.75)
    ns['comfy'].model_management=NS(get_torch_device=lambda:torch.device(target))
    old_items=list(weights)if isinstance(weights,(list,tuple))else []
    outcome=result(hook.inject)
    after=getattr(adapter,'weights',None);new_items=list(after)if isinstance(after,(list,tuple))else []
    root_same=after is weights if isinstance(weights,(torch.Tensor,list,tuple,dict))or weights is None else None
    moves.append(dict(source=source,target=target,module_kind=module_kind,form=form,tensor=encode(t),aux=encode(aux),quantized=encoded_q,post=encode(after),present=hasattr(adapter,'weights'),root_same=root_same,aliases=[[reference_alias(v,old)for old in old_items]for v in new_items],sharing=[[reference_alias(v,other)for other in new_items]for v in new_items],**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,metadata=metadata,constructors=constructors,moves=moves),separators=(',',':'))+'\n')
print(f'Captured {len(metadata)} metadata, {len(constructors)} shared-constructor and {len(moves)} movement cases; trained weights=0')
