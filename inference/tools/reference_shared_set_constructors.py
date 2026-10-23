#!/usr/bin/env python3
"""Unmodified CPython exact set/frozenset constructors on own scalar containers."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def materialize(d):
 kind,raw=d
 if kind=='int':return int(raw)
 if kind=='str':return ''.join(chr(p) for p in raw)
 if kind=='list':return [materialize(v)for v in raw]
 if kind=='tuple':return tuple(materialize(v)for v in raw)
 if kind=='bytes':return bytes(raw)
 if kind=='bytearray':return bytearray(raw)
 if kind=='set':return set(materialize(v)for v in raw)
 if kind=='frozenset':return frozenset(materialize(v)for v in raw)
 if kind=='dict':return {materialize(k):materialize(v)for k,v in raw}
 if kind=='None':return None
 raise AssertionError(kind)
def encode(v):
 if type(v)is int:return ['int',str(v)]
 if type(v)is str:return ['str',list(map(ord,v))]
 if type(v)is tuple:return ['tuple',[encode(x)for x in v]]
 if type(v)is frozenset:return ['frozenset',[encode(x)for x in v]]
 raise AssertionError(type(v))
def run(d,frozen):
 source=materialize(d)if d is not None else None;ctor=frozenset if frozen else set;result=None;error=None
 try:result=ctor(source)if d is not None else ctor()
 except Exception as e:error=dict(type=type(e).__name__,message=str(e))
 return dict(input=d,frozen=frozen,error=error,result=None if result is None else dict(order=[encode(v)for v in result],length=len(result),capacity=(result.__sizeof__()-ctor().__sizeof__())//16 or 8,same_owner=result is source,hash=hash(result)if frozen else None))
I=lambda n:['int',str(n)];T=lambda s:['str',[ord(x)for x in s]]
inputs=[None,['None',None],I(12),['list',[]],['tuple',[]],T(''),T('abca'),T('ééā😃\ud800'),['bytes',[0,255,0,3]],['bytearray',[1,1,7]],['list',[['list',[]]]],['tuple',[['dict',[]]]]]
for n in [0,1,4,5,18,20,75]:
 vals=[I(i*17)for i in range(n)]
 for kind in ['list','tuple','set','frozenset']:inputs.append([kind,vals])
 inputs.append(['dict',[[v,I(1)]for v in vals]])
inputs += [['dict',[[T('a'),I(1)],[T('b'),I(2)],[T('a'),I(3)]]],['list',[['frozenset',[]],['frozenset',[I(2)]],['tuple',[I(3),T('a')]]]]]
cases=[run(d,f)for d in inputs for f in [False,True]]
errors=[]
for ctor in [set,frozenset]:
 try:ctor([],[])
 except Exception as e:errors.append(dict(type=type(e).__name__,message=str(e)))
events=[]
class Key:
 def __init__(self,i):self.i=i
 def __hash__(self):events.append(['hash',self.i]);return 7
 def __eq__(self,other):events.append(['equal',self.i,other.i]);return self.i==other.i
source={Key(i):i for i in range(6)};events.clear();result=set(source)
cached_dict=dict(events=events,order=[v.i for v in result],capacity=(result.__sizeof__()-set().__sizeof__())//16 or 8)
classes=[]
for cls in [set,frozenset,complex]:
 try:complex(cls)
 except Exception as e:complex_error=dict(type=type(e).__name__,message=str(e))
 try:set(cls)
 except Exception as e:iterable_error=dict(type=type(e).__name__,message=str(e))
 pointer_hash=((id(cls)>>4)|(id(cls)<<(64-4)))&((1<<64)-1)
 if pointer_hash >= (1<<63):pointer_hash-=1<<64
 if pointer_hash == -1:pointer_hash=-2
 assert hash(cls)==pointer_hash
 classes.append(dict(repr=repr(cls),type=type(cls).__name__,truth=bool(cls),complex_error=complex_error,iterable_error=iterable_error))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,hashseed='0',cases=cases,argument_errors=errors,cached_dict=cached_dict,classes=classes),separators=(',',':'))+'\n');print('Captured',len(cases),'set constructor cases')
