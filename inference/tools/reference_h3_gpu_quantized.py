#!/usr/bin/env python3
"""Original tiny selected INT8 language/DiT forwards with actual load rules."""
from pathlib import Path
loader=Path(__file__).with_name('reference_h3_loading.py')
setup=loader.read_text().split('\nrows=[]\n',1)[0].replace('args.cpu=True;','args.use_pytorch_cross_attention=True;args.cpu=False;')
exec(compile(setup,str(loader),'exec'))
torch.set_num_threads(2)
def tensor(v):return torch.frombuffer(bytearray.fromhex(v['bytes']),dtype=getattr(torch,v['dtype'].removeprefix('torch.'))).clone().reshape(v['shape'])
def stamp(t):
 data=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
 return dict(type='tensor',dtype=str(t.dtype),shape=list(t.shape),device=str(t.device),strides=list(t.stride()),payload_sha256=hashlib.sha256(data).hexdigest(),requires_grad=t.requires_grad)
f=json.loads((a.own_root/'h3-loading-input.json').read_text());cases=[]
with torch.no_grad():
 for entry in f['cases']:
  role=entry['role']
  if role not in ('text','dit'):continue
  state={n:tensor(t)for n,t in entry['input'].items()}
  state,metadata=comfy.utils.convert_old_quants(state,metadata=entry['metadata'].copy())
  model=create(role,comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.bfloat16,full_precision_mm=role=='text'))
  model.load_state_dict(state,strict=True);model.requires_grad_(False);model.cuda()
  forwards=[]
  if role=='text':
   for ids in [[1,3,7,11],[3],[2,3,4,5,6,7]]:
    input=torch.tensor([ids],device='cuda');output=model.model(input,dtype=torch.float32)
    forwards.append(dict(ids=ids,output=stamp(output[0])))
  else:
   for t in [1000.,500.,0.]:
    def data(shape):return (torch.arange(__import__('math').prod(shape),dtype=torch.float32)%29*.03125-.40625).reshape(shape).to(device='cuda',dtype=torch.bfloat16)
    v=data((1,24,2,2,4));au=data((1,32,2,3));context=data((1,3,256))
    out=model([v,au],torch.tensor([t],device='cuda'),context,transformer_options={},minimax_payload=dict(seed=9,audio_scale=4.))
    forwards.append(dict(timestep=t,output=[stamp(x)for x in out]))
  cases.append(dict(role=role,loading=entry,forwards=forwards))
  del model,state
  torch.cuda.empty_cache()
torch.cuda.synchronize()
sources={str(Path(m.__file__).relative_to(a.comfy)):dict(path=m.__file__,sha256=frozen[m.__file__])for m in list(sys.modules.values())if getattr(m,'__file__',None)in frozen}
a.output.write_text(json.dumps(dict(schema=1,cpu_only=False,trained_weights=0,kitchen_triton_enabled=comfy.quant_ops.ck.registry.is_available('triton'),sources=sources,cases=cases),separators=(',',':'))+'\n')
print('Captured 6 original tiny trained-format quantized language/DiT forwards')
