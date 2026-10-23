#!/usr/bin/env python3
"""CPython byte owners/caches/key behavior from own buffers, no Torch/weights."""
import argparse,io,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
assert sys.flags.hash_randomization==0
def copied(data):
 stream=io.BytesIO(b'x'+data);stream.read(1);return stream.read(len(data))
def allocated(data):return bytes(memoryview(data))
def result(fn):
 try:return dict(value=fn())
 except Exception as e:return dict(error=dict(type=type(e).__name__,message=str(e)))
data=[b'',*(bytes([i])for i in range(256)),bytes(range(256)),b"'",b'"',b"'\"\\\t\n\r",b'abc',b'storage',b'\xffstorage']
cases=[]
for raw in data:
 first=copied(raw);second=copied(raw);fresh=allocated(raw);fresh2=allocated(raw);mutable=bytearray(raw);mutable2=bytearray(raw)
 d={first:1};d[second]=2;d[fresh]=3
 cases.append(dict(data=raw.hex(),copy_same=first is second,allocated_same=fresh is fresh2,copy_allocated_same=first is fresh,mutable_same=mutable is mutable2,repr=repr(first),mutable_repr=repr(mutable),hash=hash(first),truth=bool(first),bytes_mutable_equal=first==mutable,mutable_equal=mutable==mutable2,text_equal=first==raw.decode('latin1'),mutable_hash=result(lambda:hash(mutable)),dict_length=len(d),dict_key_same=next(iter(d))is first,dict_value=d[first]))
value=bytearray(b'a');d={}
try:d.pop(value)
except KeyError as error:
 before=str(error);value[:]=b"b'\xff";after=str(error);owned=error.args[0]is value
mutation=dict(before=before,after=after,same=owned,data=bytes(value).hex())
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,hashseed=0,trained_model_weights=0,cuda_work=False,scope='nonnull bytes copy vs allocated-buffer owners, raw mutable bytearray payload and builtin keys/repr; constructor coercion and generic buffer exports remain separate',cases=cases,mutation=mutation),separators=(',',':'))+'\n');print('Captured',len(cases),'byte owner/hash/repr histories and deferred mutable key ownership')
