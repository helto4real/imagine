#!/usr/bin/env python3
"""Own large-reference text-only isolation under the normal exclusive coordinator."""
import argparse,json,os,select,socket,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PREVIOUS=Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2')
PYTHON=Path('/home/thhel/git/ComfyUI/.venv/bin/python')
PROMPT='An adult person smiles and waves gently. Audio: quiet ambience.'
def python_probe(case,model,dynamic):
    sys.argv=[sys.argv[0]];sys.path.insert(0,'/home/thhel/git/ComfyUI')
    import comfy.cli_args
    args=comfy.cli_args.args
    args.cpu=False;args.disable_dynamic_vram=not dynamic;args.disable_comfy_compiler=True;args.use_pytorch_cross_attention=True;args.async_offload=0
    args.base_directory=str(case/'python-base');args.output_directory=str(case);args.input_directory=str(case);args.temp_directory=str(case/'tmp');args.models_directory=str(case/'empty-models')
    import torch
    if dynamic:
        import comfy_aimdo.control
        comfy_aimdo.control.init(simple_vram_headroom=None,nvml_pressure=not args.disable_nvml_pressure)
    import comfy.sd,comfy.model_management as mm,comfy.quant_ops
    from safetensors.torch import load_file,save_file
    comfy.quant_ops.ck.registry.disable('triton');torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_fp16_accumulation=False
    if dynamic:
        # Reproduce main.py's normal in-process bootstrap, without importing main,
        # loading custom nodes/config or starting a service.
        import comfy_aimdo.control,comfy.memory_management,comfy.model_patcher
        try:initialized=comfy_aimdo.control.init_devices((d.index,int(args.vram_headroom*1024**3))for d in mm.get_all_torch_devices())
        except TypeError:initialized=comfy_aimdo.control.init_devices(d.index for d in mm.get_all_torch_devices())
        assert initialized,'Normal Comfy AIMDO initialization unavailable; no alternate path'
        comfy_aimdo.control.set_log_critical()
        comfy.model_patcher.CoreModelPatcher=comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled=True
    folder=case/'python';folder.mkdir();rows=[];result={'passed':False,'trace':rows}
    def mark(name,value=None):
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        rows.append({'stage':name,'shape':list(value.shape)if torch.is_tensor(value)else None,'dtype':str(value.dtype)if torch.is_tensor(value)else None,'cuda_memory_free_total_active_reserved':[free,total,torch.cuda.memory_allocated(),torch.cuda.memory_reserved()]})
    try:
        mark('initial');image=load_file(str(case/'input.safetensors'))['image']
        clip=comfy.sd.load_clip([str(model)],clip_type=comfy.sd.CLIPType.MINIMAX,model_options={'dtype':torch.bfloat16},disable_dynamic=not dynamic)
        result['requested_dynamic']=dynamic;result['actual_dynamic']=clip.patcher.is_dynamic()
        original_load=clip.load_model
        def loaded(tokens):
            value=original_load(tokens);mark('model.loaded');return value
        clip.load_model=loaded
        transformer=clip.cond_stage_model.qwen3vl_32b.transformer
        for i,b in enumerate(transformer.visual.blocks):b.register_forward_hook(lambda m,a,o,i=i:mark(f'vision_block_{i}',o))
        for i,b in enumerate(transformer.model.layers):b.register_forward_hook(lambda m,a,o,i=i:mark(f'language_block_{i}',o[0]if isinstance(o,tuple)else o))
        tokens=clip.tokenize(PROMPT,minimax_ref_items=[{'type':'image','data':image}])
        conditioning=clip.encode_from_tokens_scheduled(tokens)
        save_file({'context':conditioning[0][0].contiguous(),'tags':conditioning[0][1]['minimax_token_tags'].contiguous()},str(folder/'conditioning.safetensors'))
        mark('complete');result['passed']=True
    except Exception as error:
        import re
        text=str(error);allocation=re.search(r'Tried to allocate .*? (?:MiB|GiB|KiB)',text)
        result['error']=allocation.group(0)if allocation else text.split('Process ')[0];result['error_type']=type(error).__name__
    finally:
        mm.unload_all_models();mm.soft_empty_cache();(folder/'result.json').write_text(json.dumps(result,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--python-probe',action='store_true');p.add_argument('--dynamic',action='store_true');p.add_argument('--python-only',action='store_true');p.add_argument('--rust-only',action='store_true');p.add_argument('--rust-dynamic',action='store_true');a=p.parse_args();assert a.case.isalnum()
    case=OWN/a.case
    runtime=json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text())['h3'];model=Path(runtime['model_root'])/'text_encoders/minimax_h3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors'
    if a.python_probe:return python_probe(case,model,a.dynamic)
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert(mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case.mkdir()
    for name in ['evidence','tmp','cache','config','share','empty-models','python-base']:(case/name).mkdir()
    env={**os.environ,**runtime['environment'],'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(case/'tmp'),'XDG_CACHE_HOME':str(case/'cache'),'XDG_CONFIG_HOME':str(case/'config'),'XDG_DATA_HOME':str(case/'share'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    for k,name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:path=case/'cache'/name;path.mkdir();env[k]=str(path)
    # Use the exact own portrait and native RGB8 Lanczos + pad preprocessing.
    import torch,numpy as np,ast
    from PIL import Image
    from safetensors.torch import save_file,load_file
    scope={'H3_REFERENCE_QUALITY_PIXELS':{'Standard':1_000_000}}
    src=Path('/home/thhel/git/comfyui-helto-director/shared/h3/reference_quality.py')
    nodes=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='reference_image_geometry'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),scope)
    image=Image.open(PREVIOUS/'own-portrait.png').convert('RGB');geometry=scope['reference_image_geometry'](*image.size,'Standard')
    image=image.resize((geometry['content_width'],geometry['content_height']),Image.Resampling.LANCZOS)
    values=torch.from_numpy(np.array(image).copy()).float().div(255).permute(2,0,1).unsqueeze(0)
    values=torch.nn.functional.pad(values,(0,geometry['padding_right'],0,geometry['padding_bottom']),mode='replicate').permute(0,2,3,1).contiguous();save_file({'image':values},str(case/'input.safetensors'))
    ticket=None;children=[];result={'geometry':geometry,'image_shape':list(values.shape),'full_generation':False,'runtime_console_logs_read':False,'production_changed':False,'workers':[]}
    def receive():
        line=b''
        while not line.endswith(b'\n'):
            if not select.select([ticket],[],[],30)[0]:print(json.dumps({'waiting_for_gpu':True}),flush=True);continue
            part=ticket.recv(1)
            if not part:raise RuntimeError('Own coordinator disconnected')
            line+=part
        return json.loads(line)
    try:
        ticket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);ticket.connect(str(ROOT/'.imagine/gpu.sock'));ticket.sendall((json.dumps({'job':'imagine-text-component-'+uuid.uuid4().hex,'bytes':64*1024*1024})+'\n').encode());assert receive()['queued'];assert receive()['granted'];result['exclusive_ticket_granted']=True
        commands=[['rust',str(ROOT/'comfyui/ui-groups-20261021/target-worker/debug/examples/h3_ui_text_component'),str(model),str(case/'input.safetensors'),str(case/'rust'),env['IMAGINE_KITCHEN_LIBRARY'],env['IMAGINE_AIMDO_LIBRARY']],['python',str(PYTHON),str(Path(__file__).resolve()),a.case,'--python-probe']]
        if a.rust_dynamic:commands[0].append('dynamic')
        if a.python_only:commands=commands[1:]
        if a.rust_only:commands=commands[:1]
        if a.dynamic:commands[-1].append('--dynamic')
        for name,*command in commands:
            started=time.monotonic()
            with(case/'evidence'/(name+'-console.log')).open('x')as out:process=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT);children.append(process)
            while process.poll()is None:
                if select.select([ticket],[],[],.2)[0]:raise RuntimeError('Own ticket lost')
                time.sleep(.1)
            row={'worker':name,'pid':process.pid,'exit':process.returncode,'seconds':time.monotonic()-started};path=case/name/'result.json'
            if path.exists():row['result']=json.loads(path.read_text())
            result['workers'].append(row);(case/'evidence/result-in-progress.json').write_text(json.dumps(result,indent=2));print(json.dumps({'worker':name,'exit':process.returncode,'passed':row.get('result',{}).get('passed'),'last_stage':row.get('result',{}).get('trace',[{}])[-1].get('stage')}),flush=True)
        if all(w.get('result',{}).get('passed')for w in result['workers']):
            reference=OWN/'textoom04/python/conditioning.safetensors'if a.rust_only else case/'python/conditioning.safetensors'
            lefts=load_file(str(case/'rust/conditioning.safetensors'));rights=load_file(str(reference));result['reference']=str(reference)
            for name in ['context','tags']:
                left,right=lefts[name],rights[name];diff=left.double()-right.double();rmse=diff.square().mean().sqrt();rms=right.double().square().mean().sqrt()
                result[name]={'shape':list(left.shape),'mae':diff.abs().mean().item(),'rmse':rmse.item(),'relative_rmse':(rmse/rms).item()if rms else 0,'max':diff.abs().max().item(),'byte_identical':torch.equal(left.view(torch.uint8),right.view(torch.uint8)),'rms_left':left.double().square().mean().sqrt().item(),'rms_right':rms.item(),'correlation':torch.corrcoef(torch.stack([left.double().flatten(),right.double().flatten()]))[0,1].item()}
    except Exception as error:result['harness_error']=repr(error)
    finally:
        for process in children:
            if process.poll()is None:process.terminate();process.wait(timeout=60)
        if ticket:ticket.close()
        result.update(own_children_reaped=all(c.poll()is not None for c in children),reservation_closed=True);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'result':str(case/'evidence/result.json'),'own_children_reaped':result['own_children_reaped'],'reservation_closed':True}),flush=True)
if __name__=='__main__':main()
