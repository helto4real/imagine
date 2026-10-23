#!/usr/bin/env python3
"""Original frozenset table/hash/equality and mutable-set argument wrappers."""
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3] == (3,12,13)

def view(s):
 size=s.__sizeof__()-frozenset().__sizeof__()
 return dict(order=[str(v) for v in s], length=len(s), capacity=size//16 if size else 8, hash=hash(s), repeated_hash=hash(s), copy_is_self=s.copy() is s)
cases=[]
for count in [0,1,4,5,18,20,75,130]:
 for mode in ['ordinary','negative','collisions']:
  values=[i*17 if mode=='ordinary' else -i-1 if mode=='negative' else i*(2**61-1) for i in range(count)]
  for holes in [False,True]:
   source=set(values)
   removed=values[::3] if holes else []
   for v in removed:source.discard(v)
   frozen=frozenset(source)
   target={frozen}
   copy=frozenset(source)
   lookup=set(source)
   nested=frozenset([frozen,frozenset(),frozenset([913])])
   contained=lookup in target
   wrong={-777}
   error=None
   try:target.remove(wrong)
   except KeyError as e:error=dict(type=type(e).__name__,original=e.args[0] is wrong)
   target.discard(lookup)
   after_discard=len(target)
   target.add(frozen)
   target.remove(lookup)
   cases.append(dict(values=[str(v) for v in values],removed=[str(v) for v in removed],frozen=view(frozen),distinct=copy is not frozen,equal=frozen==copy,mutable_equal=frozen==source,contained=contained,missing_error=error,after_discard=after_discard,after_remove=len(target),nested_hash=hash(nested),nested_length=len(nested)))
events=[]
class Key:
 def __init__(self,i):self.i=i
 def __hash__(self):events.append(['hash',self.i]);return 7
 def __eq__(self,other):events.append(['equal',self.i,other.i]);return self.i==other.i
s=set(Key(i) for i in range(6));events.clear();f=frozenset(s);events_after_copy=list(events);events.clear();h=hash(f);hash(f);cache_events=list(events)
complex_errors=[]
for args in [(frozenset(),),(1,frozenset())]:
 try:complex(*args)
 except Exception as e:complex_errors.append(dict(type=type(e).__name__,message=str(e)))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,cases=cases,cached_hash=dict(value=h,copy_events=events_after_copy,hash_events=cache_events),complex_errors=complex_errors),separators=(',',':'))+'\n')
print('Captured',len(cases),'frozen-set histories')
