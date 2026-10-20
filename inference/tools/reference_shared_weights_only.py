#!/usr/bin/env python3
"""Actual locked restricted-unpickler byte/stack/memo histories, CPU only."""
import argparse, hashlib, io, json, pickle, struct, sys, warnings
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
import torch._weights_only_unpickler as original
assert torch.__version__=='2.13.0+cu130'
class Graph:
 def __init__(self):self.ids={};self.nodes=[]
 def value(self,value):
  if value is None:return dict(type='none')
  if type(value)is bool:return dict(type='bool',value=value)
  if type(value)is int:
   raw=value.to_bytes((value.bit_length()+8)//8,'little',signed=True)
   while len(raw)>1 and((raw[-1]==0 and raw[-2]<128)or(raw[-1]==255 and raw[-2]>=128)):raw=raw[:-1]
   return dict(type='int',bytes=raw.hex())
  if type(value)is float:return dict(type='float',bits=struct.pack('>d',value).hex())
  if type(value)is str:return dict(type='str',points=[ord(c)for c in value])
  if type(value)is bytes:return dict(type='bytes',bytes=value.hex())
  identity=id(value)
  if identity in self.ids:return dict(ref=self.ids[identity])
  index=len(self.nodes);self.ids[identity]=index;self.nodes.append(None)
  if type(value)in[list,tuple]:node=dict(type=type(value).__name__,items=[self.value(v)for v in value])
  elif type(value)is dict:node=dict(type='dict',items=[[self.value(k),self.value(v)]for k,v in value.items()])
  elif type(value)is set:
   assert not value;node=dict(type='set',items=[])
  else:raise AssertionError(type(value))
  self.nodes[index]=node;return dict(ref=index)
values=[None,False,True,0,1,-1,255,256,65535,65536,-2**31,2**31-1,2**53+1,2**63-1,-2**63,2**64-1,2**127,-2**127,2**255,-2**255,0.,-0.,float('nan'),float('inf'),float('-inf'),'','own','å猫😀','\ud800',[],(),{},[1,True,None],(1,'own',False),{'weight':[1,2],'metadata':{'epoch':3}}]
shared=[1,2];values.extend([[shared,shared],{'a':shared,'b':shared},((shared,),shared)])
cycle=[];cycle.append(cycle);values.append(cycle)
raws=[('value/'+str(i),pickle.dumps(value,protocol=2))for i,value in enumerate(values)]
raws += [('raw/'+label,bytes.fromhex(raw))for label,raw in [
 ('empty',''),('stop','2e'),('mark-stop','282e'),('int32-short','4a0102'),('int16-short','4d01'),('int8-short','4b'),('float-short','4700'),
 ('unicode-header-short','580100'),('unicode-body-short','58040000006162'),('string-header-short','55'),('string-body-short','55046162'),
 ('long-header-short','8a'),('long-body-short','8a050100'),('get-short','68'),('get-missing','6804'),('long-get-short','6a01'),
 ('put-empty','7100'),('long-put-empty','7200000000'),('mark-absent','74'),('tuple1-empty','85'),('tuple2-short','4b0186'),('tuple3-short','4b014b0287'),
 ('append-empty','61'),('append-int','4b014b0261'),('appends-missing','65'),('appends-int','4b01284b0265'),
 ('setitem-empty','73'),('setitem-int','4b014b024b0373'),('setitems-odd','7d284b0175'),
 ('reduce-invalid','4b072952'),('newobj-invalid','4b072981'),('build-invalid','5d7d62'),
 ('persistent-int','4b0151'),('persistent-bool','8851'),('persistent-list','5d51'),('persistent-empty-tuple','2951'),('persistent-wrong-tag','5801000000788551'),
 ('empty-set','8f2e'),('proto-truncated','80'),('proto4-frame','800495'),('proto3-stop','80034b012e'),
 ('memo-replace','5d71005d71006800862e'),('memo-long','5d72ffffffff6affffffff862e'),('unicode-lone-surrogate','5803000000eda0802e'),
 ('memo-insertion-order','5d71085d710168086801862e'),('unknown-op','ff'),('global-blocked','636f730a73797374656d0a2e'),('global-unknown','636f776e0a556e6b6e6f776e0a2e'),
 ('global-no-newline','636f776e0a556e6b6e6f776e'),('two-loads','5d71002e68002e'),('memo-after-failure','5d7100ff68002e')]]
cases=[]
for encoding in ['bytes','ASCII','utf-8']:
 for label,raw in raws:
  stream=io.BytesIO(raw);reader=original.Unpickler(stream,encoding=encoding);runs=[]
  for run in range(2 if label in['raw/two-loads','raw/memo-after-failure']else 1):
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:output=reader.load();result=dict(output=output)
    except Exception as error:result=dict(error=type(error).__name__,message=str(error))
   graph=Graph()
   if 'output'in result:result['output']=graph.value(result['output'])
   result.update(position=stream.tell(),proto=reader.proto,stack=graph.value(reader.stack),metastack=[graph.value(v)for v in reader.metastack],memo=[[key,graph.value(value)]for key,value in reader.memo.items()],warnings=[dict(category=type(w.message).__name__,message=str(w.message))for w in caught],nodes=graph.nodes)
   runs.append(result)
  cases.append(dict(label=encoding+'/'+label,encoding=encoding,bytes=raw.hex(),runs=runs))
source=a.site/'torch/_weights_only_unpickler.py'
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_access=False,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),cases=cases),indent=2)+'\n')
print('Captured',len(cases),'original byte/stack/memo histories; no weights or CUDA')
