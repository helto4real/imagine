#!/usr/bin/env python3
"""Locked CPython float repr corpus: exponent boundaries and exact midpoint cases."""
import argparse,json,struct,sys,math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
patterns=set()
for exponent in range(2048):
 for sign in [0,1]:
  for fraction in [0,1,(1<<20)-1,1<<20,(1<<51)-1,1<<51,(1<<52)-1]:patterns.add((sign<<63)|(exponent<<52)|fraction)
for power in range(-64,9):
 for n in [1,3,5,7,9,11,15,17,31,33,63,65,127,129,255,257,511,513,1023,1025,(1<<40)-1,(1<<40)+1,(1<<50)-1,(1<<50)+1,(1<<52)-1,(1<<52)+1,(1<<53)-1]:
  for sign in [-1,1]:patterns.add(struct.unpack('<Q',struct.pack('<d',math.ldexp(sign*n,power)))[0])
state=0x789abcdef0123456
for _ in range(32768):state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1);patterns.add(state)
# Dense mantissas at quarter/eighth ULPs uncover shortest-decimal ties.
for _ in range(8192):
 state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
 n=((1<<52)|(state&((1<<52)-1)))|1
 for power in [-2,-3,-4,-5]:
  for sign in [-1,1]:patterns.add(struct.unpack('<Q',struct.pack('<d',math.ldexp(sign*n,power)))[0])
cases=[[f'{b:016x}',repr(struct.unpack('<d',struct.pack('<Q',b))[0])]for b in sorted(patterns)]
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,reference='CPython 3.12.13 Python/dtoa.c mode 0 plus Python/pystrtod.c repr format',cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'exact built-in float representations')
