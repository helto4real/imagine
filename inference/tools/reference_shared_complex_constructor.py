#!/usr/bin/env python3
"""Exact builtin complex constructor, own immutable values and malformed strings."""
import argparse,json,struct,sys,itertools
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
def bits(v):return struct.pack('<d',v).hex()
def raw(b):return struct.unpack('<d',struct.pack('<Q',b))[0]
def integer(v):return ''if not v else v.to_bytes(((~v if v<0 else v).bit_length()+8)//8,'little',signed=True).hex()
def spec(v):
 if v is None:return dict(kind='none')
 if type(v)is bool:return dict(kind='bool',value=v)
 if type(v)is int:return dict(kind='int',bytes=integer(v))
 if type(v)is float:return dict(kind='float',bits=bits(v))
 if type(v)is complex:return dict(kind='complex',real=bits(v.real),imaginary=bits(v.imag))
 if type(v)is str:return dict(kind='str',points=list(map(ord,v)))
 if type(v)is bytes:return dict(kind='bytes',data=v.hex())
 if type(v)is bytearray:return dict(kind='bytearray',data=bytes(v).hex())
 if type(v)is list:return dict(kind='list')
 if type(v)is tuple:return dict(kind='tuple')
 if type(v)is dict:return dict(kind='dict')
 raise TypeError(type(v))
values=[None,False,True,-1,0,1,257,2**53+1,2**64-1,2**1023,2**1024,2**1024-2**970-1,2**1024-2**970,2**1024-2**970+1,-2**1024,[],(),{}]
patterns=[0,0x8000000000000000,0x3ff0000000000000,0xbff0000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0xfff8000000000002,0x7ff0000000000003,0xfff0000000000004]
values.extend(raw(b)for b in patterns)
values.extend(complex(raw(r),raw(i))for r,i in itertools.product(patterns,patterns))
strings=['',' ','1','-0','-0j','-j','J','1+j','1-j','(1+2j)',' ( 1+2j ) ','((1))','1 +2j','1+ 2j','1e2+3E-2j','.5-.25j','1.e2','1e','1e+j','1e9999','-1e9999j','Inf','-Infinity','NaN','-NaNj','NAN+nanj','1_000.5+2_0j','1__0j','1e_2','_1','1_','1\0','(1)\0','١٢.٥+٣j','\u2003(１+２j)\u00a0','\x1c1','\x851','1\ud800','1+2J ','1_0\0','1\0_0','1\ud800_0','1_\ud800','1\x7f_é','(j)','( -j )','+nan-nanj']
values.extend(strings)
cases=[]
def capture(args):
 try:
  out=complex(*args);return dict(output=dict(real=bits(out.real),imaginary=bits(out.imag)),same=[out is v for v in args])
 except Exception as e:return dict(error=dict(type=type(e).__name__,message=str(e)))
cases.append(dict(args=[],**capture([])))
for i,v in enumerate(values):cases.append(dict(args=[i],**capture([v])))
for i,v in enumerate(values):
 for j,w in enumerate(values):
  if i<128 and j<128 or i>=128 and j in range(18)or j>=128 and i in range(18):cases.append(dict(args=[i,j],**capture([v,w])))
# All original Unicode decimal/whitespace points, plus ASCII/control and
# underscore boundaries, are single-argument cases rather than a cross-product.
for point in range(0x110000):
 char=chr(point)
 if char.isdecimal():text=char+'.'+char+'j'
 elif char.isspace()or point<256:text=char+'1'+char
 else:continue
 index=len(values);values.append(text);cases.append(dict(args=[index],**capture([text])))
for value in [b'',b'1',bytearray(),bytearray(b'1')]:
 index=len(values);values.append(value)
 for args in [[index],[index,4],[4,index],[index,128],[128,index]]:cases.append(dict(args=args,**capture([values[i]for i in args])))
for count in [3,4,10]:cases.append(dict(args=[4]*count,**capture([values[4]]*count)))
conversions=[]
for i,v in enumerate(values):
 if type(v)in[int,bool,float,complex]:
  try:conversions.append(dict(index=i,output=bits(float(v))))
  except Exception as e:conversions.append(dict(index=i,error=dict(type=type(e).__name__,message=str(e))))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,scope='exact builtin complex after caller positional expansion; custom/subclass protocols and kwargs caller remain separate',definitions=[spec(v)for v in values],cases=cases,conversions=conversions),separators=(',',':'))+'\n');print('Captured',len(values),'builtin inputs,',len(cases),'complex calls and',len(conversions),'builtin float conversions')
