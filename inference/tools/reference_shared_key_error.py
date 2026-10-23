#!/usr/bin/env python3
"""CPython numeric/string KeyError args and deferred formatting, own objects."""
import argparse,gc,json,math,struct,sys,traceback,weakref
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def raw(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]
def fbits(n):return struct.pack('<d',n).hex()
def integer(v):
 if not v:return ''
 return v.to_bytes(((~v if v<0 else v).bit_length()+8)//8,'little',signed=True).hex()
def encode(value):
 if value is None:return dict(kind='none')
 if type(value)is bool:return dict(kind='bool',value=value)
 if type(value)is int:return dict(kind='int',bytes=integer(value))
 if type(value)is float:return dict(kind='float',bits=fbits(value))
 if type(value)is complex:return dict(kind='complex',real=fbits(value.real),imaginary=fbits(value.imag))
 if type(value)is str:return dict(kind='str',points=list(map(ord,value)))
 raise TypeError(type(value))
def result(fn):
 try:return dict(value=fn())
 except Exception as e:return dict(error=dict(type=type(e).__name__,message=str(e)))
values=[None,False,True,0,-1,1,2**64,-2**64,2**1024,-2**1024]
patterns=[0,1,0x8000000000000000,0x3ff0000000000000,0xbff0000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000000,0xfff8000000000001,0x7ff0000000000001]
state=0xa98b76543210fedc
for _ in range(4096):state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1);patterns.append(state)
values.extend(raw(v)for v in patterns)
for r in patterns[:10]:
 for i in patterns[:10]:values.append(complex(raw(r),raw(i)))
for points in [[],[0],[39],[34],[39,34],[92],[10],[0xd800],[0xdcff],[0x10ffff],[0x85],[0xa0],[0x1f600],[65,0xd800,66]]:values.append(''.join(map(chr,points)))
cases=[]
for value in values:
 e=KeyError(value);cases.append(dict(key=encode(value),representation=repr(value),message=str(e),args_same=e.args is e.args,key_same=e.args[0]is value))
limit_cases=[]
old_limit=sys.get_int_max_str_digits()
try:
 for digits in [639,640,641,699,700,4299,4300,4301,6000]:
  for negative in [False,True]:
   value=(10**digits-1)*(-1 if negative else 1)
   e=KeyError(value)
   history=[]
   for limit in [640,0,4300]:
    sys.set_int_max_str_digits(limit)
    history.append(dict(limit=limit,format=result(lambda:str(e)),class_unchanged=type(e).__name__,args_same=e.args[0]is value))
   limit_cases.append(dict(key=encode(value),history=history))
finally:sys.set_int_max_str_digits(old_limit)
events=[]
class OwnKey:
 fail=False
 def __repr__(self):
  events.append('repr')
  if self.fail:raise ValueError('own repr failed')
  return '<own key>'
key=OwnKey();weak=weakref.ref(key);e=KeyError(key);del key;args=e.args
before=list(events);first=str(e);weak().fail=True;failed=result(lambda:str(e));diagnostic=''.join(traceback.format_exception_only(e));e.args=();alive_with_args=weak()is not None;del args;gc.collect();expired=weak()is None
own=dict(before=before,first=first,failed=failed,diagnostic=diagnostic,events=events,alive_with_exposed_args=alive_with_args,expired_after_release=expired,empty_message=str(e))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,default_digit_limit=old_limit,scope='scalar/Unicode key repr, deferred KeyError str/args and own repr protocol; compound repr and complete exception protocols remain separate',cases=cases,limit_cases=limit_cases,own_protocol=own),separators=(',',':'))+'\n');print('Captured',len(cases),'scalar/Unicode KeyErrors,',len(limit_cases),'mutable-limit histories and deferred repr ownership')
