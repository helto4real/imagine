#!/usr/bin/env python3
"""Original adapter/patch shape metadata, tuple conversion and equality."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import struct
def restore(desc):
 kind=desc['type']
 if kind=='scalar':return struct.unpack('>d',bytes.fromhex(desc['float_bits']))[0]if 'float_bits'in desc else desc['value']
 if kind in ['tuple','list']:
  values=[restore(v)for v in desc['items']];return tuple(values)if kind=='tuple'else values
 if kind=='dict':return {key:restore(value)for key,value in desc['items']}
 if kind=='tensor':
  dtype=getattr(torch,desc['dtype'].split('.')[-1]);raw=bytes.fromhex(desc['payload_hex'])
  value=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(desc['shape'])if raw else torch.empty(desc['shape'],dtype=dtype)
  return value.to(desc['device'])
 raise AssertionError(kind)
fixture=json.loads((project/'inference/fixtures/shared-pad-values.json').read_text())
cases=[]
with torch.no_grad():
 for source in fixture['cases']:
  reshape=restore(source['shape'])
  initial=torch.Size(source['weight']['shape']);weight=NS(shape=initial)
  item=namespace['LoRAAdapter'](set(),(None,None,None,None,None,reshape))
  diff=(1.,('diff',(NS(shape=torch.Size([9,8])),{'pad_weight':True})),1.,None,None)
  regular=(1.,item,1.,None,None)
  for mode,patches in [('lora',[regular]),('offset',[(1.,item,1.,(0,0,1),None)]),('diff-first',[diff,regular]),('diff-last',[regular,diff])]:
   result={}
   try:
    output=namespace['calculate_shape'](patches,weight,'unit.weight')
    result=dict(output=encode(output),container='torch.Size'if isinstance(output,torch.Size)else type(output).__name__,initial_alias=output is initial)
    try:result['matches_initial']=bool(output==initial)
    except Exception as error:
     reject_resource_failure(error);result['comparison_error']=dict(error=type(error).__name__,message=str(error))
   except Exception as error:
    reject_resource_failure(error);result=dict(error=type(error).__name__,message=str(error))
   cases.append(dict(label=source['label']+'/'+mode,mode=mode,initial=list(initial),reshape=source['shape'],**result))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=72,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original shape metadata/equality histories; no trained weights')
