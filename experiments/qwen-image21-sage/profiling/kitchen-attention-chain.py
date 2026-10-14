"""Synthetic replay of Comfy Qwen21's Kitchen prefix-reuse attention path."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
import comfy_kitchen as ck

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
p.add_argument('prefix', type=int)
p.add_argument('iterations', type=int)
a = p.parse_args()
torch.set_grad_enabled(False)
def tensor(name, dtype, shape):
    return torch.from_numpy(np.fromfile(a.input/name, dtype=dtype).reshape(shape)).cuda()
x = tensor('x.f32', np.float32, (1, 5600, 4096)).bfloat16()
w = [tensor(f'w{i}.i8', np.int8, (4096, 4096)) for i in range(4)]
s = tensor('scale.f32', np.float32, (4096,))
norm = tensor('norm.f32', np.float32, (128,)).bfloat16()
pe = tensor('pe.f32', np.float32, (1,5600,1,64,2,2))
pk = tensor(f'pv-{a.prefix}.f32', np.float32, (1,a.prefix,32,128)).bfloat16()
# Generator writes identical logical synthetic K and V prefix values.
pv = pk.clone()
state = {}
def qkv():
    state['q'], state['k'], state['v'] = [ck.int8_linear(x, wi, s, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256).view(1,5600,32,128) for wi in w[:3]]
def rms_rope():
    state['qr'],state['kr'] = ck.rms_rope(state['q'],state['k'],pe,norm,norm,1e-6)
def prefix_concat():
    state['kc'] = torch.cat([pk,state['kr']],dim=1)
    state['vc'] = torch.cat([pv,state['v']],dim=1)
def attention():
    packed = ck.prequantize_int8_attention(state['qr'].transpose(1,2),state['kc'].transpose(1,2),state['vc'].transpose(1,2))
    state['attn'] = ck.int8_attention_from_prequantized(packed).transpose(1,2).reshape(1,5600,4096)
def out_projection():
    state['y'] = ck.int8_linear(state['attn'],w[3],s,out_dtype=torch.bfloat16,convrot=True,convrot_groupsize=256)
parts = dict(qkv=qkv,rms_rope=rms_rope,prefix_concat=prefix_concat,attention=attention,out_projection=out_projection)
def whole():
    for fn in parts.values(): fn()
results=[]
with ck.use_backend('cuda'):
    for _ in range(20): whole()
    torch.cuda.synchronize()
    for label,fn in [*parts.items(),('whole',whole)]:
        for _ in range(5): fn()
        torch.cuda.synchronize()
        for batch in range(5):
            start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin=time.monotonic_ns();start.record()
            for _ in range(a.iterations):fn()
            stop.record();stop.synchronize();end=time.monotonic_ns()
            results.append(dict(part=label,start_ns=begin,end_ns=end,iterations=a.iterations,cuda_ms=start.elapsed_time(stop)/a.iterations))
    state['y'].float().cpu().numpy().tofile(a.output/'y.f32')
    metadata={key:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),contiguous=v.is_contiguous()) for key,v in state.items()}
(a.output/'result.json').write_text(json.dumps(dict(engine='kitchen',prefix=a.prefix,batches=results,tensors=metadata,torch=torch.__version__,cuda=torch.version.cuda,kitchen=importlib.metadata.version('comfy-kitchen')),indent=2)+'\n')
