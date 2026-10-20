#!/usr/bin/env python3
"""CPython 3.12 own byte decode boundaries; no Torch, files or runtime data."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
inputs={b'',b'own',b'\xed\xa0',b'\xed\xa0\x80',b'\xed\xbf\xbf',b'\xef\xbb\xbf',b'\xf4\x8f\xbf\xbf'}
inputs.update(bytes([b])for b in range(256))
for lead in [0xc2,0xdf,0xe0,0xe1,0xed,0xef,0xf0,0xf1,0xf4]:
 for second in range(256):
  for tail in [b'',b'\x80',b'\xbf',b'\x80\x80',b'\xbf\xbf',b'A',b'\x80A']:
   inputs.add(bytes([lead,second])+tail)
state=0x12345678
for i in range(4096):
 raw=[]
 for _ in range(i%7):state=(1664525*state+1013904223)&0xffffffff;raw.append(state>>24)
 inputs.add(bytes(raw))
cases=[]
for raw in sorted(inputs):
 for encoding,mode in [('utf-8','strict'),('utf-8','surrogatepass'),('ascii','strict')]:
  try:output=raw.decode(encoding,mode);result=dict(points=[ord(c)for c in output])
  except UnicodeDecodeError as error:result=dict(error=type(error).__name__,message=str(error),start=error.start,end=error.end,reason=error.reason)
  cases.append(dict(bytes=raw.hex(),encoding=encoding,mode=mode,**result))
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'original Unicode decode histories; no Torch or runtime data')
