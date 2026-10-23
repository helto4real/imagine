#!/usr/bin/env python3
"""Own CPython numeric owners/equality/dictionary histories, no Torch or weights."""
import argparse,json,math,struct,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
values=[];definitions=[]
def integer_bytes(data):values.append(int.from_bytes(data,'little',signed=True));definitions.append(dict(kind='int',bytes=data.hex()))
def integer(value):integer_bytes(value.to_bytes(max(1,(value.bit_length()+8)//8),'little',signed=True))
def number(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]
def bits(value):return struct.pack('<d',value).hex()
def floating(value):
 value=number(struct.unpack('<Q',struct.pack('<d',value))[0]);values.append(value);definitions.append(dict(kind='float',bits=bits(value)))
def complex_value(real,imag):
 value=complex(real,imag);values.append(value);definitions.append(dict(kind='complex',real=bits(value.real),imaginary=bits(value.imag)))
for v in [False,True]:values.append(v);definitions.append(dict(kind='bool',value=v))
for v in range(-7,259):integer(v);integer(v)
for data in [b'',b'\0',b'\0'*65,b'\xff',b'\xff'*65,b'\x80',b'\x80\xff',b'\x80\0',b'\x80\0\0']:integer_bytes(data)
for power in [30,31,52,53,54,60,61,62,63,64,65,127,128,255,256,511,512,1022,1023,1024,2048]:
 for sign in [-1,1]:
  for delta in [-1,0,1]:integer(sign*((1<<power)+delta))
for v in [-math.inf,-2.**1023,-2.**128,-2.**64,-2.**63,-2.**61,-2.**53,-6.,-5.,-1.,-0.,0.,1.,2.,3.5,256.,257.,2.**53,2.**61,2.**63,2.**64,2.**128,2.**1023,math.inf]:floating(v);floating(v)
for bits_value in [1,(1<<52)-1,0x7ff0000000000001,0x7ff8000000000000,0xfff8000000000001]:floating(number(bits_value));floating(number(bits_value))
for r in [-math.inf,-2.**64,-1.,-0.,0.,1.,1.5,2.**53,math.inf,math.nan]:
 for i in [-0.,0.,1.,-1.,math.inf,math.nan]:complex_value(r,i)
state=0x12345678
for _ in range(128):
 state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1);floating(number(state))
owner_index={id(v):i for i,v in enumerate(values)}
for index,v in enumerate(values):
 definition=definitions[index];definition.update(hash=hash(v),owner_hash=object.__hash__(v),truth=bool(v),exact_int=type(v)is int,owner_class=owner_index[id(v)])
comparisons=[]
initial_count=len(values)
for i in range(initial_count):
 v=values[i]
 targets=set(range(min(20,len(values))))|{i}|{(i*31+j*97)%len(values)for j in range(32)}
 if type(v)in[int,bool]:
  try:f=float(v)
  except OverflowError:pass
  else:
   floating_index=len(values);values.append(f);definitions.append(dict(kind='float',bits=bits(f),hash=hash(f),owner_hash=object.__hash__(f),truth=bool(f),exact_int=False,owner_class=floating_index));targets.add(floating_index)
 for j in sorted(targets):comparisons.append([i,j,v==values[j]])
# Explicit finite-key table: no process-address-dependent NaN table indices.
finite=[i for i,v in enumerate(values)if not(type(v)is float and math.isnan(v))and not(type(v)is complex and(math.isnan(v.real)or math.isnan(v.imag)))]
d={};history=[]
for index in finite:
 d[values[index]]=index
 if len(history)<64 or index%17==0:
  history.append(dict(index=index,entries=[[owner_index.get(id(k),next(i for i,v in enumerate(values)if v is k)),v]for k,v in d.items()]))
# Distinct NaNs must stay distinct, but the same NaN key is found by identity.
nans=[float('nan'),float('nan')];nan_map={nans[0]:11,nans[1]:22};nan_case=dict(length=len(nan_map),same_lookup=[nan_map[v]for v in nans],equality=nans[0]==nans[0])
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,scope='numeric object payload/cache/equality/key ownership; NaN hashes follow each actual owning object, not cross-process address equality',trained_model_weights=0,cuda_access=False,definitions=definitions,comparisons=comparisons,finite_history=history,nan_identity=nan_case),separators=(',',':'))+'\n');print('Captured',len(definitions),'numeric objects,',len(comparisons),'comparisons and',len(history),'dictionary histories')
