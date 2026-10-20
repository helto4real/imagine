#!/usr/bin/env python3
"""CPython 3.12.13 numeric hashes on bounded own scalar values only."""
import argparse,json,struct,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
assert (sys.hash_info.width,sys.hash_info.modulus,sys.hash_info.inf,sys.hash_info.imag)==(64,(1<<61)-1,314159,1000003)
integer_bytes=[b'',*(bytes([v]) for v in range(256))]
for power in [0,1,7,8,15,16,30,31,60,61,62,63,64,65,127,128,255,256,511,512,1023,1024,2047,2048,4095]:
 for delta in [-2,-1,0,1,2]:
  for sign in [-1,1]:
   value=sign*((1<<power)+delta);integer_bytes.append(value.to_bytes(max(1,(value.bit_length()+8)//8),'little',signed=True))
state=0x89abcdef01234567;mask=(1<<64)-1
def draw():
 global state
 state=(state*6364136223846793005+1442695040888963407)&mask;return state
for length in range(257):integer_bytes.append(bytes(draw()>>56 for _ in range(length)))
integers=[dict(bytes=data.hex(),hash=hash(int.from_bytes(data,'little',signed=True))) for data in integer_bytes]
patterns=[]
for exponent in range(2048):
 for sign in [0,1]:
  for fraction in [0,1,1<<51,(1<<52)-1]:patterns.append((sign<<63)|(exponent<<52)|fraction)
patterns.extend(draw() for _ in range(4096))
def number(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]
floats=[]
for bits in patterns:
 value=number(bits);floats.append(dict(bits=f'{bits:016x}',owner_hash=object.__hash__(value),hash=hash(value)))
special=[0,1,0x8000000000000000,0x3ff0000000000000,0xbff0000000000000,0x43c0000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000000,0xfff8000000000001,0x7ff0000000000001,0xfff0000000000001]
pairs=[(r,i)for r in special for i in special]+[(draw(),draw())for _ in range(1024)]
complex_values=[]
for real,imaginary in pairs:
 value=complex(number(real),number(imaginary));complex_values.append(dict(real=f'{real:016x}',imaginary=f'{imaginary:016x}',owner_hash=object.__hash__(value),hash=hash(value)))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,reference='https://docs.python.org/3.12/library/stdtypes.html#hashing-of-numeric-types',trained_model_weights=0,cuda_access=False,integer_bytes=integers,floats=floats,complex=complex_values),separators=(',',':'))+'\n')
print('Captured',len(integers),'integer,',len(floats),'float and',len(complex_values),'complex hashes from CPython 3.12.13')
