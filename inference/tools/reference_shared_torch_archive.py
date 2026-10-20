#!/usr/bin/env python3
"""Original PyTorchFileReader on tiny own ZIP files; no trained weights."""
import argparse, gc, hashlib, io, json, sys, zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
from reference_checks import reject_resource_failure
a.work.mkdir(parents=True,exist_ok=True)
archives=[]
buffer=io.BytesIO();torch.save({'base':torch.arange(12,dtype=torch.float32).reshape(3,4),'view':torch.arange(4,dtype=torch.int64)},buffer)
archives.append(('torch-save',buffer.getvalue()))
for compression in [zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED]:
 for version in [b'3\n',b'0\n',b'999\n']:
  buffer=io.BytesIO()
  with zipfile.ZipFile(buffer,'w',compression=compression) as file:
   for name,data in [('root/version',version),('root/data.pkl',b'\x80\x02N.'),('root/empty',b''),('root/data/0',bytes(range(16))),('root/porträtt/猫',b'own UTF-8 record'),('root/byteorder',b'little')]:file.writestr(name,data)
  archives.append((f'zip-{compression}-{version.decode().strip()}',buffer.getvalue()))
for label,records in [('no-prefix',[('version',b'3\n')]),('missing-version',[('root/data/0',b'1234')])]:
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w') as file:
  for name,data in records:file.writestr(name,data)
 archives.append((label,buffer.getvalue()))
archives.extend([('not-zip',b'own short invalid archive'),('truncated',archives[0][1][:100])])
cases=[]
def failure(error):
 reject_resource_failure(error);return dict(error=type(error).__name__,message=str(error))
def outcome(call):
 try:return dict(value=call())
 except Exception as error:return failure(error)
for label,data in archives:
 path=a.work/'own-archive.pt';path.write_bytes(data)
 case=dict(label=label,archive_hex=data.hex())
 try:reader=torch._C.PyTorchFileReader(str(path))
 except Exception as error:case.update(failure(error));cases.append(case);continue
 case['records']=reader.get_all_records();case['queries']=[]
 for key in case['records']+['absent','data/999','empty\0suffix']:
  case['queries'].append(dict(key=key,has=outcome(lambda:reader.has_record(key)),data=outcome(lambda:reader.get_record(key).hex()),offset=outcome(lambda:reader.get_record_offset(key))))
 case['storages']=[]
 for key in ['data/0','empty','absent']:
  for dtype_name,code,width in [('uint8',0,1),('float32',6,4),('float64',7,8),('int64',4,8),('float16',5,2)]:
   length=len(reader.get_record(key)) if reader.has_record(key) else 16
   for count in [0,1,length//width]:
    item=dict(key=key,dtype=dtype_name,code=code,count=count)
    try:result=reader.get_storage_from_record(key,count,getattr(torch,dtype_name))
    except Exception as error:item.update(failure(error))
    else:
     storage=result.untyped_storage();weak=storage._weak_ref()
     second=reader.get_storage_from_record(key,count,getattr(torch,dtype_name))
     item.update(shape=list(result.shape),stride=list(result.stride()),offset=result.storage_offset(),bytes=storage.nbytes(),payload_hex=bytes(storage).hex(),resizable=storage.resizable(),repeated_storage_alias=second.untyped_storage()._cdata==storage._cdata)
     del second
     del result,storage;gc.collect();item['expired_after_output_release']=torch.UntypedStorage._expired(weak);torch.UntypedStorage._free_weak_ref(weak)
    case['storages'].append(item)
 # Storage data is independently owned after closing the reader.
 hold=reader.get_storage_from_record('data/0',len(reader.get_record('data/0')),torch.uint8)
 del reader;gc.collect();case['held_payload_hex']=bytes(hold.untyped_storage()).hex();del hold
 cases.append(case)
sources=[Path('/home/thhel/ChatGPT/imagine/comfyui/moss-20261018/sources/pytorch/torch/csrc/jit/python/init.cpp'),a.site/'torch/include/caffe2/serialize/inline_container.h']
a.output.write_text(json.dumps(dict(schema=1,trained_model_weights=0,cuda_access=False,max_archive_bytes=max(len(d) for _,d in archives),sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original own archive histories,',sum(len(c.get('storages',[])) for c in cases),'storage operations')
