#!/usr/bin/env python3
"""Original Qwen-VL patching, positions, DeepStack assembly and Qwen21 filter.

Only explicitly shaped synthetic image/hidden tensors; no model construction,
trained weights, production startup or model directories are accessed.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_text_encoder.py')
exec(compile(loader.read_text().split('\ndef model_class(',1)[0],str(loader),'exec'))
for name in ['comfy/text_encoders/qwen_vl.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/text_encoders/qwen_vl.py',['process_qwen2vl_images','qwen2vl_mrope_position_ids'])
comfy.text_encoders.qwen_vl=NS(qwen2vl_mrope_position_ids=namespace['qwen2vl_mrope_position_ids'])
# Only the parent's completed hidden/mask tuple is a fixture. The filtering
# method itself is the unmodified original class body, including super().
class FilterParent:
 def encode_token_weights(self,pairs):return self.fixture
namespace['sd1_clip']=NS(SD1ClipModel=FilterParent)
definitions('comfy/text_encoders/qwen_image21.py',['QwenImage21TEModel'])
def stamp(v):
 if isinstance(v,torch.Tensor):
  raw=v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
  return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),payload_sha256=hashlib.sha256(raw).hexdigest(),requires_grad=v.requires_grad)
 if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[stamp(x)for x in v])
 if isinstance(v,dict):return dict(type='dict',items=[[k,stamp(x)]for k,x in v.items()])
 return dict(type='scalar',value=v)
patches=[];positions=[];filters=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   for shape,opts in [([1,32,64,3],{}),([1,48,80,4],{}),([2,35,97,4],{}),([1,5,7,3],{}),([1,96,160,3],dict(max_pixels=4096)),([1,64,64,3],dict(interpolation='bicubic')),([1,42,71,3],dict(patch_size=14,image_mean=None,image_std=None))]:
    kw=dict(patch_size=16,image_mean=[.5]*3,image_std=[.5]*3);kw.update(opts)
    x=data(shape,device,dtype,3)
    out=namespace['process_qwen2vl_images'](x,**kw)
    patches.append(dict(device=device,dtype=str(dtype),shape=shape,options=opts,output=stamp(out),input_after=stamp(x)))
  for specs in [[],[(3,4,[1,4,4])],[(2,6,[1,4,6]),(12,4,[1,4,4])],[(12,4,[1,4,4]),(2,6,[1,4,6])],[(0,8,[2,4,4])]]:
   info=[]
   for i,(index,size,grid)in enumerate(specs):
    ds=[data([size,8],device,torch.float32,7+i+d)for d in range(3)]
    info.append(dict(type='image',index=index,size=size,extra=dict(grid=torch.tensor([grid],device=device),deepstack=ds)))
   owner=namespace['Qwen3VL'].__new__(namespace['Qwen3VL']);torch.nn.Module.__init__(owner)
   # Both dict-grid and legacy tensor-grid routes are accepted by the original helper.
   raw=namespace['qwen2vl_mrope_position_ids'](info,24,torch.device(device))
   legacy=[dict(e,extra=e['extra']['grid'])for e in info]
   raw_legacy=namespace['qwen2vl_mrope_position_ids'](legacy,24,torch.device(device))
   built=owner.build_image_inputs(torch.empty([1,24,8],device=device),info)
   positions.append(dict(device=device,specs=specs,seq=24,raw=stamp(raw),legacy=stamp(raw_legacy),built=stamp(built)))
  for tokens,spans,seq in [([31,71],[],2),([151644,31,151644,71],[],4),([151644,31,151644,'image',71,'image',151644],[(3,4),(8,6)],15),([151644,'image',151644,71,'image',151644],[(1,3),(6,4)],11),(['image',31,151644,71,151644],[(0,2)],6),(['image',151644,151644],[],3)]:
   for keep in [False,True]:
    for masked in [False,True]:
     owner=namespace['QwenImage21TEModel'].__new__(namespace['QwenImage21TEModel'])
     owner.clip='clip_model';owner.clip_model=NS(image_spans=spans)
     context=data([1,seq,8],device,torch.float32,11)
     mask=torch.ones([1,seq],device=device,dtype=torch.int64)
     if masked:mask[0,-1]=0
     owner.fixture=(context,None,dict(attention_mask=mask))
     pairs=[[(dict(type='image')if t=='image'else t,1.)for t in tokens]]
     out=owner.encode_token_weights(dict(qwen3vl_8b=pairs,keep_vision=keep))
     filters.append(dict(device=device,tokens=tokens,spans=spans,seq=seq,keep_vision=keep,masked=masked,output=stamp(out)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,provenance=__doc__,patches=patches,positions=positions,filters=filters),separators=(',',':'))+'\n')
print(f'Captured {len(patches)} patching, {len(positions)} multimodal-position/DeepStack and {len(filters)} original Qwen21 filtering cases; no model constructors')
