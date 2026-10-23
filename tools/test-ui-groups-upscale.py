#!/usr/bin/env python3
"""Own synthetic/full-weight component parity. No product services or runtime log reads."""
import argparse,ast,json,os,select,socket,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
SOURCE=Path('/home/thhel/git/ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py')
PYTHON='/home/thhel/git/ComfyUI/.venv/bin/python'
MODEL=Path('/mnt/media/gemensam/AI/ComfyUI/models/latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors')
def definitions():
    import torch,re,gc
    from enum import Enum
    from typing import TypedDict
    from types import SimpleNamespace
    scope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'re':re,'gc':gc,'Enum':Enum,'TypedDict':TypedDict,'override':lambda f:f,'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda x:(x,)),'HAS_COMFY_MM':False}
    wanted={'normalization','zero_module','AttnBlock3D','ResBlockEmb3D','TemporalConv','LatentResizer3D','_make_norm_tensors','_is_rocm_build','_resolve_device','_backend_label','_extract_upscaler_sd','_detect_arch','UpscaleMode','UpscaleConfig','MinimaxH3LatentUpscaler3D'}
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in wanted:nodes.append(n)
        if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id in {'LATENTS_MEAN','LATENTS_STD','VAE_DOWNSAMPLE'}for x in n.targets):nodes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),scope)
    return scope
def python_worker(args,folder):
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(2);s=definitions()
    weights=s['_extract_upscaler_sd'](load_file(args.model));cfg=s['_detect_arch'](weights)
    model=s['LatentResizer3D'](**cfg);model.load_state_dict(weights,strict=True);del weights
    model.eval();input=load_file(args.input)['input']
    if args.mode=='forward':
        with torch.no_grad():
            w=max(1,round(round(input.shape[-1]*16*args.factor/32)*32/16));h=max(1,round(round(input.shape[-2]*16*args.factor/32)*32/16))
            output=model(input,scale=args.factor,target_size=(input.shape[2],h,w),enable_chunking=True)
    else:
        def load_model(name,device,precision):return model.to(device).to({'fp16':torch.float16,'fp32':torch.float32}[precision])
        s['load_model']=load_model
        output=s['MinimaxH3LatentUpscaler3D'].execute(latent={'samples':input},model_name=Path(args.model).name,mode={'mode':'scale by multiplier','scale':args.factor},align=32,enable_temporal_chunking=True,force_unload=True,device=args.device,precision='fp16')[0]['samples']
        if args.device=='cuda':torch.cuda.synchronize()
    save_file({'output':output.contiguous()},args.output)
def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',action='store_true');p.add_argument('--temporal',type=int);p.add_argument('--python-worker',action='store_true');p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');p.add_argument('--mode',default='forward');p.add_argument('--factor',type=float,default=1.5);p.add_argument('--device',default='cpu');a=p.parse_args();assert a.case.isalnum()
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert(mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    folder=OWN/a.case
    if a.python_worker:
        assert Path(a.input).is_relative_to(folder)and Path(a.output).is_relative_to(folder)
        python_worker(a,folder);return 0
    folder.mkdir()
    for name in ['tmp','cache','evidence']:(folder/name).mkdir()
    env=dict(os.environ,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),CUDA_VISIBLE_DEVICES='0'if a.gpu else'',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib')
    if a.gpu:
        runtime=Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2/full-fp16-20261020-5e4a923b/candidate-runtime.json')
        env.update(json.loads(runtime.read_text())['h3']['environment'])
    import torch
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(2);scope=definitions()
    report={'gpu':a.gpu,'runtime_logs_read':False,'production_changed':False,'python_provider_scope':'AST-selected provider math/execute with its no-Comfy-MM cache fallback; not whole ComfyUI memory policy','passed':False,'cases':[],'own_children_reaped':False,'reservation_closed':not a.gpu}
    if a.gpu:weights=MODEL
    else:
        weights=folder/'own-small-model.safetensors'
        model=scope['LatentResizer3D'](channels=32,in_blocks=2,out_blocks=2,temporal_every=2,temporal_kernel=3,attn=False).eval()
        with torch.no_grad():
            for name,tensor in model.state_dict().items():
                values=torch.arange(tensor.numel(),dtype=torch.float32).reshape(tensor.shape).sin()*.02
                if name.endswith('weight')and tensor.ndim==1:values+=1
                tensor.copy_(values)
        save_file(model.state_dict(),str(weights));del model
    ticket=None;children=[]
    def receive():
        line=b''
        while not line.endswith(b'\n'):
            if not select.select([ticket],[],[],30)[0]:print(json.dumps({'waiting_for_gpu':True}),flush=True);continue
            chunk=ticket.recv(1)
            if not chunk:raise RuntimeError('Own coordinator ticket disconnected')
            line+=chunk
        return json.loads(line)
    try:
        if a.gpu:
            ticket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);ticket.connect(str(ROOT/'.imagine/gpu.sock'))
            ticket.sendall((json.dumps({'job':'imagine-upscale-component-'+uuid.uuid4().hex,'bytes':64*1024*1024})+'\n').encode());assert receive()['queued'];assert receive()['granted'];report['exclusive_ticket_granted']=True
        for temporal in ([a.temporal]if a.temporal is not None else[3]if a.gpu else[3,35]):
            input=folder/f'input-{temporal}.safetensors';values=torch.linspace(-.75,.75,24*temporal*4*6).reshape(1,24,temporal,4,6);save_file({'input':values},str(input))
            rust=folder/f'rust-{temporal}.safetensors';python=folder/f'python-{temporal}.safetensors';mode='video'if a.gpu else'forward';device='cuda'if a.gpu else'cpu'
            commands=[['/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/target-worker/debug/examples/h3_ui_upscale_component',str(weights),str(input),str(rust),mode,str(a.factor),device],[PYTHON,str(Path(__file__).resolve()),a.case,'--python-worker','--model',str(weights),'--input',str(input),'--output',str(python),'--mode',mode,'--factor',str(a.factor),'--device',device]]
            if a.gpu:commands[0].append(env['IMAGINE_KITCHEN_LIBRARY'])
            durations=[]
            for i,command in enumerate(commands):
                started=time.monotonic()
                with(folder/'evidence'/f'{temporal}-{i}-console.log').open('x') as output:
                    process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT);children.append(process)
                    while process.poll()is None:
                        if ticket and select.select([ticket],[],[],.2)[0]:raise RuntimeError('Own GPU ticket lost while component was running')
                        time.sleep(.05)
                    assert process.returncode==0,f'Own component {i} failed with exit {process.returncode}; runtime console was not read'
                durations.append(time.monotonic()-started)
            left,right=[load_file(str(path))['output']for path in [rust,python]];assert left.shape==right.shape and left.dtype==right.dtype
            diff=left.double()-right.double();same=torch.equal(left.view(torch.uint8),right.view(torch.uint8))
            metric={'temporal':temporal,'shape':list(left.shape),'dtype':str(left.dtype),'byte_identical':same,'mae':diff.abs().mean().item(),'rmse':diff.square().mean().sqrt().item(),'max':diff.abs().max().item(),'seconds_rust_python':durations}
            report['cases'].append(metric);print(json.dumps(metric),flush=True);assert same,'Component differs; preserve files and stop before runtime binding'
        report['passed']=True
    except Exception as error:report['error']=repr(error)
    finally:
        for process in children:
            if process.poll()is None:process.terminate()
            process.wait(timeout=30)
        report['own_children_reaped']=all(p.poll()is not None for p in children)
        if ticket:
            try:ticket.sendall(b'{"finished":true}\n')
            finally:ticket.close();report['reservation_closed']=True
        (folder/'evidence/result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report.get(k)for k in ['passed','error','own_children_reaped','reservation_closed']}),flush=True)
    return not report['passed']
if __name__=='__main__':raise SystemExit(main())
