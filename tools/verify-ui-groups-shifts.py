#!/usr/bin/env python3
"""Compare own CPU schedules with unchanged ComfyUI source functions; no runtime import."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import torch

OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
COMFY=Path('/home/thhel/git/ComfyUI')
def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert a.probe.is_relative_to(OWN)and a.output.is_relative_to(OWN)
    torch.set_num_threads(1)
    ns={'torch':torch,'math':math};sources=[]
    for name,wanted in [('comfy/model_sampling.py',{'time_snr_shift','ModelSamplingDiscreteFlow','ModelSamplingAV'}),('comfy/samplers.py',{'simple_scheduler'}),('comfy/ldm/minimax/model.py',{'time_shift_sigma'})]:
        path=COMFY/name;code=path.read_text();tree=ast.parse(code);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in wanted];assert len(nodes)==len(wanted)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        sources.append({'path':str(path),'sha256':hashlib.sha256(code.encode()).hexdigest()})
    report={'device':'cpu','model_runtime_initialized':False,'sources':sources,'cases':[]}
    for case in json.loads(a.probe.read_text())['cases']:
        model=ns['ModelSamplingAV']();model.set_parameters(shift=case['video_shift'],audio_shift=case['audio_shift']);video=ns['simple_scheduler'](model,20);audio=ns['time_shift_sigma'](video,case['video_shift'],case['audio_shift'])
        metrics={}
        for name,reference in [('video_sigmas',video),('audio_sigmas',audio)]:
            rust=torch.tensor(case[name],dtype=torch.float32);diff=(rust-reference).double();metrics[name]={'byte_identical':torch.equal(rust.view(torch.uint8),reference.view(torch.uint8)),'mae':diff.abs().mean().item(),'rmse':diff.square().mean().sqrt().item(),'max':diff.abs().max().item()};assert metrics[name]['byte_identical']
        assert case['audio_scale']==model.audio_scale
        report['cases'].append({'video_shift':case['video_shift'],'audio_shift':case['audio_shift'],'audio_scale':model.audio_scale,'metrics':metrics})
    report['passed']=True
    with a.output.open('x')as output:json.dump(report,output,indent=2)
    print(json.dumps({'passed':True,'cases':len(report['cases']),'output':str(a.output)}))
if __name__=='__main__':main()
