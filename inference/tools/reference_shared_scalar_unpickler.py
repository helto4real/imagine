#!/usr/bin/env python3
"""Original weights-only VM scalar owner graphs on tiny own pickle byte streams."""
import argparse,hashlib,io,json,pickle,struct,sys,warnings
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
from torch._weights_only_unpickler import Unpickler
assert sys.version_info[:3]==(3,12,13)
def nbytes(v):
 if not v:return b''
 n=((~v if v<0 else v).bit_length()+8)//8;return v.to_bytes(n,'little',signed=True)
streams=[]
for v in [None,False,True,-6,-5,0,1,256,257,2**64-1,-2**64,2**1023,2**2047,-0.,float('inf'),float('-inf'),float('nan'),[],(),{},[None,None,True,True,1,1],['A','A','é','é','€','€'],{1:True,1.0:False,2.5:None,(1,2):'key'},[float('nan'),float('nan')]]:
 streams.append(pickle.dumps(v,protocol=2))
for raw in [b'(K\x01K\x01t.',b'(J\x01\x01\0\0J\x01\x01\0\0t.',b'(G'+struct.pack('>d',float('nan'))+b'q\x00h\x00t.',b'\x80\x03N.',b'G\0\0',b'}(]K\x01u.',b'}(K\x01',b'Nq\x00',b'\x80\x02\x8a\x00.',b'(U\x01AU\x01At.',b'}(NK\x01NK\x02u.']:
 streams.append(raw)
# A second load retains memo and proto but resets stack/metastack.
streams.append(pickle.dumps([257,257,-0.],protocol=2)+b'h\x00.')
repeated=streams[-1]
streams=[('utf-8',data,data==repeated)for data in streams]
def short(data):return b'U'+bytes([len(data)])+data
for data in [b'',b'A',b'\0',b'\xff',b'storage',b'xx']:
 streams.extend([('bytes',short(data)+b'.',False),('bytes',b'('+short(data)+short(data)+b't.',False)])
for tag in [b'storage',b'other',b'\xff']:
 streams.append(('bytes',b'('+short(tag)+b'tQ.',False))
streams.append(('bytes',b'X\x07\0\0\0storage\x85Q.',False))
for invalid in [b'\xff', b'\xe2', b'\xed\xa0\x80', b'\xf4\x90\x80\x80']:
 streams.append(('bytes',b']q\x00c'+invalid+b'\nname\n.',False))
 streams.append(('bytes',b'(Ncown\n'+invalid+b'\n.',False))
for data in [b'\x8f.',b'(\x8f\x8ft.',b'\x8fq\x00h\x00\x86.',b'}(\x8fK\x01u.',b'\x8fK\x01a.',b'\x8fK\x01K\x02s.']:
 streams.append(('bytes',data,False))
def graph(roots):
 nodes=[];seen={}
 def ref(v):
  if id(v)in seen:return seen[id(v)]
  i=len(nodes);seen[id(v)]=i;nodes.append(None);kind=type(v).__name__;n=dict(kind=kind)
  if kind=='int':n['bytes']=nbytes(v).hex()
  elif kind=='float':n['bits']=struct.pack('<d',v).hex()
  elif kind=='bool':n['value']=v
  elif kind=='str':n['points']=list(map(ord,v))
  elif kind in ['bytes','bytearray']:n['data']=bytes(v).hex()
  elif kind in ['list','tuple','set']:n['items']=[ref(x)for x in v]
  elif kind=='dict':n['entries']=[[ref(k),ref(x)]for k,x in v.items()]
  elif kind!='NoneType':raise TypeError(kind)
  nodes[i]=n;return i
 result={name:ref(v)for name,v in roots.items()}
 return dict(roots=result,nodes=nodes)
cases=[]
for encoding,data,repeated in streams:
 stream=io.BytesIO(data);vm=Unpickler(stream,encoding=encoding);loads=[]
 for turn in range(2 if repeated else 1):
  with warnings.catch_warnings(record=True)as caught:
   warnings.simplefilter('always');result=None;error=None
   try:result=vm.load()
   except Exception as e:error=dict(type=type(e).__name__,message=str(e))
  roots={}
  if error is None:roots['output']=result
  roots['stack']=vm.stack
  for i,value in enumerate(vm.metastack):roots['mark'+str(i)]=value
  for i,value in vm.memo.items():roots['memo'+str(i)]=value
  loads.append(dict(error=error,position=stream.tell(),proto=vm.proto,warnings=[str(w.message)for w in caught],graph=graph(roots)))
 cases.append(dict(encoding=encoding,bytes=data.hex(),loads=loads))
decoder_identity=[]
for point in [*range(256),256,8364,0xd800,0x10ffff]:
 text=chr(point);data=text.encode('utf-8','surrogatepass')
 first=data.decode('utf-8','surrogatepass');second=data.decode('utf-8','surrogatepass')
 decoder_identity.append(dict(point=point,bytes=data.hex(),same=first is second,ascii_cross_same=(bytes([point]).decode('ascii')is first)if point<128 else None))
source=a.site/'torch/_weights_only_unpickler.py'
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},trained_model_weights=0,cuda_work=False,scope='scalar VM output/memo/stack owner graphs; not complete inner exception frames/global/tensor/file loading',cases=cases,decoder_identity=decoder_identity,none_hash=hash(None)),separators=(',',':'))+'\n');print('Captured',len(cases),'scalar VM byte streams and',sum(len(c['loads'])for c in cases),'loads')
