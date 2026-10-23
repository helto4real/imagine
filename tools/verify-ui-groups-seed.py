#!/usr/bin/env python3
"""Unchanged ComfyUI prepare_noise functions on synthetic AV tensors, CPU only."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch

OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
COMFY=Path('/home/thhel/git/ComfyUI')
def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert a.probe.is_relative_to(OWN) and a.output.is_relative_to(OWN)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    ns={'torch':torch};sources=[]
    for name,wanted in [('comfy/nested_tensor.py',{'NestedTensor'}),('comfy/sample.py',{'prepare_noise_inner','prepare_noise'})]:
        path=COMFY/name;code=path.read_text();tree=ast.parse(code)
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted];assert len(nodes)==len(wanted)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        sources.append({'path':str(path),'sha256':hashlib.sha256(code.encode()).hexdigest()})
    ns['comfy']=SimpleNamespace(nested_tensor=SimpleNamespace(NestedTensor=ns['NestedTensor']))
    report={'device':'cpu','model_runtime_initialized':False,'sources':sources,'cases':[]}
    for case in json.loads(a.probe.read_text())['cases']:
        dtype=getattr(torch,case['dtype']);latent=ns['NestedTensor']([torch.zeros(s['shape'],dtype=dtype) for s in case['streams']])
        reference=ns['prepare_noise'](latent,case['seed']).unbind();metrics=[]
        for stream,ref in zip(case['streams'],reference,strict=True):
            rust=torch.tensor(stream['values'],dtype=dtype).reshape(stream['shape']);diff=(rust.double()-ref.double())
            item={'byte_identical':torch.equal(rust.view(torch.uint8),ref.view(torch.uint8)),'mae':diff.abs().mean().item(),'rmse':diff.square().mean().sqrt().item(),'max':diff.abs().max().item()}
            assert item['byte_identical'];metrics.append(item)
        report['cases'].append({'seed':case['seed'],'dtype':case['dtype'],'streams':metrics})
    report['passed']=True
    with a.output.open('x') as output:json.dump(report,output,indent=2)
    print(json.dumps({'passed':True,'cases':len(report['cases']),'output':str(a.output)}))
if __name__=='__main__':main()
