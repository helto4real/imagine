#!/usr/bin/env python3
"""Composed original AV latent/flow boundary; synthetic velocity, no trained DiT."""
import argparse,ast,hashlib,json,math,sys,types
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
ns=dict(torch=torch,math=math);sources={}
def nodes(relative,names):
 path=a.comfy/relative;sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest();tree=ast.parse(path.read_text());return[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
def execute(body):exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'locked-av-composition','exec'),ns)
execute(nodes('comfy/nested_tensor.py',['NestedTensor']));N=ns['NestedTensor'];ns['comfy']=types.SimpleNamespace(nested_tensor=types.SimpleNamespace(NestedTensor=N))
execute(nodes('comfy/latent_formats.py',['LatentFormat','MiniMaxH3Video','MiniMaxH3AV']))
execute(nodes('comfy/utils.py',['pack_latents','unpack_latents']))
execute(nodes('comfy/model_sampling.py',['time_snr_shift','ModelSamplingDiscreteFlow','ModelSamplingAV','reshape_sigma','CONST']))
execute(nodes('comfy/sample.py',['prepare_noise_inner','prepare_noise']))
base,h3=nodes('comfy/model_base.py',['BaseModel','MiniMaxH3'])
# Retain actual method source bodies and their class/super contexts, without
# constructing the unrelated learned-module hierarchy or any model weights.
for cls,names in [(base,['process_latent_in','process_latent_out']),(h3,['audio_scale','_scale_audio_slice','process_latent_in','process_latent_out'])]:
 cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in names]
base.bases=[];h3.bases=[ast.Name(id='BaseModel',ctx=ast.Load())];execute([base,h3])
ns['np']=__import__('numpy')
def tensor(t):return dict(shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'),bytes=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex())
class Sampling(ns['ModelSamplingAV'],ns['CONST']):pass
cases=[]
with torch.no_grad():
 for dtype in [torch.float32,torch.bfloat16,torch.float64]:
  for frames in [1,6,11]:
   for shape_metadata in [False,True]:
    latent_format=ns['MiniMaxH3AV']();base_latent=torch.zeros((1,32,frames,2,2),dtype=dtype)
    original=latent_format.fix_empty_latent(base_latent);values=original.unbind();values[0].copy_(torch.arange(values[0].numel()).reshape(values[0].shape)/127)
    sampling=Sampling(types.SimpleNamespace(sampling_settings=dict(shift=12.0,audio_shift=3.0)))
    model=object.__new__(ns['MiniMaxH3']);model.model_sampling=sampling;model.latent_format=latent_format
    packed,shapes=ns['pack_latents'](values);model.latent_shapes=shapes if shape_metadata else None
    prepared=model.process_latent_in(packed)
    nested_prepared=model.process_latent_in(original)
    noise=ns['prepare_noise'](original,42);packed_noise,_=ns['pack_latents'](noise.unbind())
    sigma=sampling.sigma(torch.tensor([500.0]));x=sampling.noise_scaling(sigma,packed_noise,prepared)
    velocity=x*.125+sigma.reshape(1,1,1)*.01
    denoised=sampling.calculate_denoised(sigma,velocity,x)
    final=sampling.inverse_noise_scaling(torch.tensor([0.0]),denoised)
    output=model.process_latent_out(final);streams=ns['unpack_latents'](output,shapes)
    nested_output=model.process_latent_out(N(ns['unpack_latents'](final,shapes)))
    cases.append(dict(dtype=str(dtype).removeprefix('torch.'),frames=frames,shape_metadata=shape_metadata,original=[tensor(t)for t in values],packed=tensor(packed),prepared=tensor(prepared),nested_prepared=[tensor(t)for t in nested_prepared.unbind()],noise=tensor(packed_noise),sigma=tensor(sigma),x=tensor(x),denoised=tensor(denoised),output=tensor(output),streams=[tensor(t)for t in streams],nested_output=[tensor(t)for t in nested_output.unbind()],audio_scale=model.audio_scale(),sigmas=tensor(sampling.sigmas)))
errors=[]
model.latent_shapes=shapes;model.model_sampling=types.SimpleNamespace(audio_scale=0.0)
for nested,value in [(False,packed),(True,original)]:
 try:model.process_latent_out(value)
 except Exception as e:errors.append(dict(nested=nested,type=type(e).__name__,message=str(e)))
assert len(errors)==2
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_only=True,trained_weight_tensors=0,velocity='explicit synthetic x * .125 + sigma * .01; not a learned model',cases=cases,errors=errors),separators=(',',':'))+'\n');print('Captured',len(cases),'composed original AV latent/flow histories')
