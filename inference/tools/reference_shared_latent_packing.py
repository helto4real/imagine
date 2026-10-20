#!/usr/bin/env python3
"""Original bounded multimodal latent packing, views and malformed shapes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from itertools import product
definitions('comfy/utils.py',['pack_latents','unpack_latents'])
cases=[]
def storage(a,b):return a.untyped_storage()._cdata==b.untyped_storage()._cdata
with torch.no_grad():
 for device,dtype,layout in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16,torch.float64],['empty','scalar','one','two','three','transpose','mixed','bad-batch','batch-zero','feature-zero','rank-one']):
  shapes=[]if layout=='empty'else[()]if layout=='scalar'else[(2,)]if layout=='rank-one'else[(0,2)]if layout=='batch-zero'else[(2,0,3)]if layout=='feature-zero'else[(2,3)]if layout=='one'else[(2,2,3),(1,1,3)]if layout=='bad-batch'else[(2,2,3),(2,3,2),(2,1)]if layout=='three'else[(2,2,3),(2,1,4)]
  values=[data(shape,device,torch.float64 if layout=='mixed'and i==1 else dtype,i*3)for i,shape in enumerate(shapes)]
  if layout=='transpose':values=[v.transpose(-1,-2)for v in values]
  owners={}
  def pack():
   packed,metadata=ns['pack_latents'](values)
   restored=ns['unpack_latents'](packed,metadata)
   owners.update(shape_types=[type(v).__name__ for v in metadata],packed_input_storage=[storage(packed,v)for v in values],unpacked_storage=[storage(v,packed)for v in restored],unpacked_objects=[v is packed for v in restored])
   return packed,metadata,restored
  cases.append(dict(name='pack',params=dict(device=device,layout=layout),inputs=encode(values),**result(pack),owners=owners))
 shape_sets=[[],[(2,3)],[(2,3),(2,2)],[(2,0),(2,6)],[(2,-1),(2,2)],[(2,-2),(2,2)],[(),()],[(2,3,2),(2,1,4)],[(0,3),(0,3)],[(2,20),(2,2)]]
 for device,dtype,shape,metadata,layout in product(['cpu','cuda:0'],[torch.float32,torch.bfloat16],[(2,1,10),(2,2,5),(1,1,10),(2,10),(10,),()],shape_sets,['contiguous','transpose']):
  combined=data(shape,device,dtype)
  if layout=='transpose'and combined.ndim>1:combined=combined.transpose(0,1)
  owners={}
  def unpack():
   values=ns['unpack_latents'](combined,metadata)
   owners.update(same_object=[v is combined for v in values],same_storage=[storage(v,combined)for v in values],storage_matrix=[[storage(a,b)for b in values]for a in values])
   return values
  cases.append(dict(name='unpack',params=dict(device=device,metadata=[list(v)for v in metadata]),input=encode(combined),**result(unpack),owners=owners))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'pack/unpack cases with original view/storage behavior; no trained weights',flush=True)
