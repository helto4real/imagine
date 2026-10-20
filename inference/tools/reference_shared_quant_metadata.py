#!/usr/bin/env python3
"""Original checkpoint quant metadata; tiny initialized tensors, no model load."""
from pathlib import Path
loader = Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n', 1)[0], str(loader), 'exec'))
import hashlib
import random
import struct
definitions('comfy/utils.py', ['convert_old_quants'])
ns['json'] = json
stdlib = {}
for module in [json, json.decoder, json.encoder, json.scanner]:
 path = Path(module.__file__)
 stdlib[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
cases = []
def record(label, sd, metadata, prefix='', device='cpu'):
 initial, meta_initial = encode(sd), encode(metadata)
 old = list(sd.values()); out = None; out_meta = None
 def run():
  nonlocal out, out_meta
  out, out_meta = ns['convert_old_quants'](sd, prefix, metadata)
  return out
 outcome = result(run)
 def owned(v):return v is None or isinstance(v,(torch.Tensor,dict,list,tuple))
 def aliases(values):return [[v is other if owned(v) and owned(other)else None for other in old]for v in values]
 cases.append(dict(label=label,device=device,prefix=prefix,initial=initial,metadata=meta_initial,
     post=encode(sd),metadata_post=encode(metadata),root_same=out is sd,metadata_same=out_meta is metadata,
     metadata_out=encode(out_meta),aliases=aliases(list(out.values())if out is not None else[]),
     post_aliases=aliases(list(sd.values())),**outcome))
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for prefix in ['', 'p.', 'model.diffusion_model.']:
    for count in [0,1,2,3]:
     for pattern in ['empty','standard','reverse','collision','input_only','bad_item','many_item','zero_item']:
      sd={'outside':data((1,),device,dtype),'unrelated.comfy_quant':torch.tensor([7],dtype=torch.uint8)}
      sd[prefix+'scaled_fp8']=data((count,),device,dtype)
      entries=[(prefix+'a.weight',data((2,2),device,dtype)),(prefix+'a.scale_weight',data((),device,dtype)),(prefix+'a.scale_input',torch.tensor(1.,device=device,dtype=dtype)),(prefix+'b.scale_weight',data((),device,dtype)),(prefix+'b.scale_input',torch.tensor(.5,device=device,dtype=dtype))]
      if pattern=='empty':entries=[]
      elif pattern=='reverse':entries.reverse()
      elif pattern=='collision':entries=[(prefix+'a.weight_scale',data((1,),device,dtype))]+entries+[(prefix+'a.input_scale',data((1,),device,dtype)),(prefix+'a.comfy_quant',torch.tensor([9],dtype=torch.uint8))]
      elif pattern=='input_only':entries=[entries[2],entries[4]]
      elif pattern in ['bad_item','many_item','zero_item']:entries[2]=(entries[2][0],[]if pattern=='bad_item'else data((2,)if pattern=='many_item'else(0,),device,dtype))
      sd.update(entries)
      record('legacy-'+pattern,sd,None if count%2 else {'note':'own synthetic'},prefix,device)
 for marker in [None,[],{},'own']:
  record('bad-marker',{'scaled_fp8':marker,'a.scale_weight':torch.tensor(.5)}, {})
 for metadata in [None,{}, {'note':True}]:
  record('no-marker',{'a.weight':torch.tensor([.5])},metadata)
 # Explicit metadata wins even over an invalid legacy marker; preserves dict order.
 documents=['null','{}','[]','true','1','1.0','"own"','{"layers":null}','{"layers":[]}','{"layers":"x"}',
  '{"layers":{}}','{"layers":{"b":{"format":"int8","rank":2},"a":{"format":"nvfp4"}}}',
  '{"layers":{"a":{"z":1,"a":2,"z":3,"unicode":"å🙂","escapes":"\\u0000\\b\\t\\n\\f\\r\\\\\\\""}}}',
  '{"layers":{"a":{"surrogates":["\\ud800","\\udfff","\\ud800x\\udc00","\\ud83d\\ude42","\\ud800\\ud800"]}}}',
  '{"layers":{"a":{"nonfinite":[NaN,Infinity,-Infinity,1e999,-1e999,-0,-0.0,1e-999]}}}',
  '{"layers":{"a":0},"layers":{"b":2,"b":3,"a":1}}',
  '{"layers":{"a":'+('9'*4300)+'}}','{"layers":{"a":'+('9'*4301)+'}}']
 for item in [None,False,[],{},1]:
  record('json-type',{'scaled_fp8':None},{'_quantization_metadata':item})
 bad=['',' ','\ufeff{}','None','nan','[1,]','{"a":1,}','{"a" 1}','{"a":}','{"a":1 2}','"unterminated','"a\\q"','"\\u1"','"\\uzzzz"','"a\n"','"a\x00"','01','-','1.','1e','[true false]','{} []','\n\n🙂','{"å🙂": }','"\\ud800\\u12"']
 for doc in ['{"layers":{"a":[true,null,1.25]}}','[1,2,3]','{"a":"å🙂"}']:
  bad.extend(doc[:i] for i in range(len(doc)))
 documents.extend(bad)
 rng=random.Random(782390)
 for _ in range(2048):
  bits=rng.getrandbits(64); value=struct.unpack('d',bits.to_bytes(8,'little'))[0]
  documents.append('{"layers":{"float":'+json.dumps(value)+'}}')
 for exponent in [-324,-309,-308,-16,-7,-6,-5,-4,-3,0,15,16,17,20,23,100,308,309]:
  for mantissa in ['0','1','-1','1.0000000000000001','9.999999999999999','2.2250738585072014','4.9406564584124654']:
   documents.append('{"layers":{"float":'+mantissa+'e'+str(exponent)+'}}')
 for doc in documents:
  record('explicit-json',{'keep':torch.tensor([.25]),'scaled_fp8':None,'a.comfy_quant':torch.tensor([3],dtype=torch.uint8)}, {'_quantization_metadata':doc,'note':'own'})
a.output.write_text(json.dumps(dict(schema=1,sources=sources,stdlib_sources=stdlib,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'quantization metadata conversion cases; trained weights=0',flush=True)
