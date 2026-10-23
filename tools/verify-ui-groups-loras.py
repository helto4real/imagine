#!/usr/bin/env python3
"""Actual H3 aliases vs ComfyUI function, on our own CPU adapter fixture only."""
import ast,json,os,subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0];assert(mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    folder=OWN/'loracpu02';assert not folder.exists()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TMPDIR':str(OWN/'tmp'),'PYTHONDONTWRITEBYTECODE':'1','LD_LIBRARY_PATH':'/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib'}
    result=json.loads(subprocess.check_output([str(ROOT/'comfyui/ui-groups-20261021/target-worker/debug/examples/h3_ui_loras_cpu'),str(folder)],env=env))
    path=Path('/home/thhel/git/ComfyUI/comfy/lora.py');function=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='model_lora_keys_unet')
    utils=Path('/home/thhel/git/ComfyUI/comfy/utils.py');conversion=next(n for n in ast.parse(utils.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='unet_to_diffusers');scope={};exec(compile(ast.Module(body=[conversion],type_ignores=[]),str(utils),'exec'),scope)
    base=SimpleNamespace()
    for node in ast.walk(function):
        if isinstance(node,ast.Attribute)and isinstance(node.value,ast.Attribute)and isinstance(node.value.value,ast.Name)and node.value.value.id=='comfy'and node.value.attr=='model_base':setattr(base,node.attr,type(node.attr,(),{}))
    model=base.MiniMaxH3();model.model_config=SimpleNamespace(unet_config={});model.state_dict=lambda:{'diffusion_model.audio_patch_proj.weight':None,'diffusion_model.audio_patch_proj.bias':None}
    scope={'comfy':SimpleNamespace(model_base=base,utils=SimpleNamespace(unet_to_diffusers=scope['unet_to_diffusers']))}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    expected=scope['model_lora_keys_unet'](model,{})
    assert result['mapping']==expected,(result['mapping'],expected)
    result.update(comfy_aliases_exact=True,scope='AST actual ComfyUI key-map function with a tiny MiniMaxH3 model-state facade; no module import/runtime/config reads; CPU PatchOwner registration, not full GPU patch-numerics parity')
    with(folder/'result.json').open('x')as f:json.dump(result,f,indent=2)
    print(json.dumps({'passed':True,'aliases':len(expected),'adapter_cases':len(result['cases']),'output':str(folder/'result.json')}))
if __name__=='__main__':main()
