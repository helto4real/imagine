#!/usr/bin/env python3
"""Compare actual Rust CPU solver traces with extracted local ComfyUI source.
No ComfyUI imports/services, weights, RNG or CUDA calls. Input is h3_res20_cpu JSON.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
root=Path(sys.argv[1]);rust=json.loads(Path(sys.argv[2]).read_text())
ns={'torch':torch,'trange':lambda n,**kw:range(n)}
def extract(path,names):
    text=path.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names),(path,names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return hashlib.sha256(text.encode()).hexdigest()
# append_dims has no dependencies beyond torch.
utils=root/'comfy/k_diffusion/utils.py'
extract(utils,{'append_dims'});ns['utils']=SimpleNamespace(append_dims=ns['append_dims'])
hashes={}
for relative,names in [
 ('comfy/ldm/minimax/model.py',{'time_shift_sigma'}),
 ('comfy/model_sampling.py',{'time_snr_shift','ModelSamplingDiscreteFlow','ModelSamplingAV'}),
 ('comfy/samplers.py',{'simple_scheduler'}),
 ('comfy/k_diffusion/sampling.py',{'get_ancestral_step','to_d','res_multistep','sample_res_multistep'}),
]: hashes[relative]=extract(root/relative,names)
schedule=ns['ModelSamplingAV'](SimpleNamespace(sampling_settings={'shift':12.,'audio_shift':3.}))
assert schedule.audio_scale==rust['audio_scale']==4.
max_error=0.;differences=[]
def check(a,b):
    global max_error
    a=torch.tensor(a,dtype=torch.float64);b=torch.tensor(b,dtype=torch.float64)
    assert a.shape==b.shape,(a.shape,b.shape)
    delta=(a-b).abs().max().item();max_error=max(max_error,delta)
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-6),delta
check(ns['time_shift_sigma'](ns['simple_scheduler'](schedule,20),12.,3.).tolist(),rust['audio_sigmas'])
check((torch.tensor([.1,-.4,.75])*4.).tolist(),rust['scaled_audio'])
class Model:
    inner_model=SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda _:schedule))
    def __call__(self,x,sigma,**kw):
        return torch.tanh(x*0.37+x.flip([-1])*0.13+sigma.reshape(-1,1)*0.19)
def forbidden_noise(*args):raise AssertionError('eta0 requested stochastic noise')
for case in rust['cases']:
    sigmas=ns['simple_scheduler'](schedule,case['steps']);check(sigmas.tolist(),case['sigmas'])
    trace=[]
    def cb(s):trace.append({k:(v.flatten().tolist() if torch.is_tensor(v) else v) for k,v in s.items()})
    out=ns['sample_res_multistep'](Model(),torch.tensor([[-.4,.2,.7,-.1,.4,-.8,.3,.5]]),sigmas,callback=cb,disable=True,noise_sampler=forbidden_noise)
    assert len(trace)==len(case['trace'])==case['steps']
    for py,rs in zip(trace,case['trace']):
        assert py['i']==rs['i']
        for key in ['sigma','x','denoised']:check(py[key],rs[key])
    check(out.flatten().tolist(),case['out'])
    delta=(torch.tensor(case['out'])-torch.tensor(case['euler'])).abs().max().item()
    if case['steps']==20:assert delta>1e-5,'multistep unexpectedly aliases Euler'
    if case['steps']<=2:assert delta<2e-6,'first/last steps should be Euler'
    differences.append({'steps':case['steps'],'max_difference_from_euler':delta})
print(json.dumps({'result':'PASS','device':'cpu','max_abs_error_all_sigmas_steps_outputs':max_error,'cases':differences,'audio_scale':schedule.audio_scale,'reference_source_sha256':hashes,'limits':'Synthetic positive-only predictor; no model, GPU, visual or audio quality acceptance.'},indent=2))
