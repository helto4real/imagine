"""Run the installed AIO Qwen adapter without a ComfyUI HTTP server.
Only metadata/timings and requested output images are persisted; no activations.
"""
import argparse
import contextlib
import datetime
import functools
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

START = time.perf_counter()
p = argparse.ArgumentParser()
p.add_argument('--runs', type=int, default=3)
p.add_argument('--profile-index', type=int, default=-1)
p.add_argument('--probe', action='store_true')
p.add_argument('--chain-scopes', action='store_true', help='Add profile-only operation-chain scopes and tensor metadata')
a = p.parse_args()
assert a.runs > 0 and -1 <= a.profile_index < a.runs
ROOT = Path(__file__).resolve().parents[1]
COMFY = Path('/path/to/ComfyUI')
AIO = Path('/path/to/comfyui-all-on-one-image-generation-node')
MODELS = Path('/path/to/models')
REF = ROOT/'runs/20261013T074331Z-debug-7nor7cre'
OUT = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-comfy-'), dir=ROOT/'runs'))
print(OUT, flush=True)
for key, subdir in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda-cache')]:
    os.environ[key] = str(OUT/subdir)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0]]  # Never pass this harness's arguments into ComfyUI.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
import torch
import numpy as np
from PIL import Image
import comfy.options
comfy.options.enable_args_parsing(False)
import folder_paths
# Explicit user-selected root; do not read private extra_model_paths.yaml.
for category, paths in {'diffusion_models':['diffusion_models'], 'text_encoders':['text_encoders','clip'], 'vae':['vae'], 'embeddings':['embeddings'], 'loras':['loras']}.items():
    folder_paths.folder_names_and_paths[category] = ([str(MODELS/x) for x in paths], folder_paths.supported_pt_extensions)
folder_paths.set_output_directory(str(OUT))
folder_paths.set_temp_directory(str(OUT/'tmp'))
folder_paths.set_input_directory(str(OUT/'input'))
import comfy.model_management as mm
import comfy.samplers
import comfy.sample
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as kc
from comfy.ldm.qwen_image21 import model as qm
from comfy.ldm.wan.model_animate2 import PoseBranchCache
# Namespace packages avoid importing AIO's route-registration __init__.py.
for name, path in [('aio_bench',AIO),('aio_bench.adapters',AIO/'adapters')]:
    package=types.ModuleType(name); package.__path__=[str(path)]; sys.modules[name]=package
from aio_bench.adapters.qwen_image_2_1 import QwenImage21Adapter
from aio_bench.services import pipeline, performance, qwen_image_2_1 as qwen
from aio_bench.services import qwen_encoding_cache
from aio_bench.services import profiles
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

adapter=QwenImage21Adapter()
settings=adapter.resolve_settings(model_settings=dict(family='qwen_image_2_1',reference_megapixels=1.6,edit_canvas='main controls',memory_policy='auto',cache_device='auto',cache_dtype='default',attention_mode='comfy_kitchen_int8',reference_upscale_method='lanczos',sampling_schedule='comfy',color_match_strength=.8,color_match=False,brightness_strength=0.,contrast_strength=0.,microcontrast_strength=0.,privacy_mode=True,use_zero_negative_conditioning=False),width=1600,height=896,steps=40,cfg=2.5,sampler='er_sde',scheduler='beta')
settings.update(max_side=1600,aspect_ratio='16:9',size_mode='use aspect ratio',multiple_value=32)
settings=adapter.resolve_reference_settings(settings,None)
models=dict(diffusion_model='diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors',text_encoder='clip/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors',vae='vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors')
paths=pipeline.safetensors_backend.resolve_paths(**models)
metadata=dict(comfy_revision=subprocess.check_output(['git','-C',str(COMFY),'rev-parse','HEAD'],text=True).strip(),aio_revision=subprocess.check_output(['git','-C',str(AIO),'rev-parse','HEAD'],text=True).strip(),torch=torch.__version__,torch_cuda=torch.version.cuda,kitchen=importlib.metadata.version('comfy-kitchen'),gpu=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),settings=settings,models={k:dict(path=str(v),size=v.stat().st_size,inode=v.stat().st_ino) for k,v in vars(paths).items()},prompt_hashes={n:hashlib.sha256((REF/n).read_bytes()).hexdigest() for n in ['prompt.txt','negative-prompt.txt']},preview='headless callback; no live preview image',execution='AIO adapter.generate per image; models loaded by AIO each call; process and encoding cache retained',privacy='no prompt text, model activations, conditioning tensors or workflow in traces/output PNG metadata')
(OUT/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT/'harness.py').write_bytes(Path(__file__).read_bytes())
if a.probe:
    print('PROBE OK', flush=True)
    raise SystemExit(0)

active = None
profiling = False
forward_samples = set()
@contextlib.contextmanager
def region(name, sync=False):
    if sync: torch.cuda.synchronize()
    begin=time.perf_counter()
    cm=torch.profiler.record_function('bench::'+name) if profiling else contextlib.nullcontext()
    with cm:
        yield
    if sync: torch.cuda.synchronize()
    if active is not None:
        active['phases'].setdefault(name,[]).append(time.perf_counter()-begin)

def instrument(obj, attr, name, sync=False):
    old=getattr(obj,attr)
    @functools.wraps(old)
    def call(*args,**kwargs):
        with region(name,sync):return old(*args,**kwargs)
    setattr(obj,attr,call)

for name in ['load_diffusion_model','load_text_encoder','load_vae','sample_with_comfy_ksampler','decode_latent']:
    instrument(pipeline,name,name,True)
instrument(qwen,'prepare_models','prepare_models',True)
# The native node runs both prompt encodings; classmethod binding is preserved.
old_encode=TextEncodeQwenImage21.execute
@classmethod
def encode(cls,*args,**kwargs):
    with region('conditioning',True):return old_encode(*args,**kwargs)
TextEncodeQwenImage21.execute=encode

old_forward=qm.QwenImage21Transformer2DModel.forward
@functools.wraps(old_forward)
def forward(self,*args,**kwargs):
    if profiling and a.chain_scopes:
        install_linear_scopes(self)
    if active is not None:
        active['forward_count']+=1
        x=args[0] if args else kwargs.get('x')
        key=(str(x.dtype),tuple(x.shape)) if isinstance(x,torch.Tensor) else ('unknown',())
        if key not in forward_samples:
            forward_samples.add(key);active['forward_inputs'].append({'dtype':key[0],'shape':key[1]})
    with region('diffusion_forward'):
        return old_forward(self,*args,**kwargs)
qm.QwenImage21Transformer2DModel.forward=forward
old_select=qm.QwenImage21Transformer2DModel.select_prefix_cache
@functools.wraps(old_select)
def select(self,*args,**kwargs):
    result=old_select(self,*args,**kwargs)
    cache,cached=result
    if active is not None:
        item=dict(device=str(cache.store_device) if cache is not None else None,dtype=cache.dtype if cache is not None else None,cached=bool(cached))
        key=json.dumps(item,sort_keys=True);active['prefix_selections'][key]=active['prefix_selections'].get(key,0)+1
    return result
qm.QwenImage21Transformer2DModel.select_prefix_cache=select
old_put=PoseBranchCache.put
@functools.wraps(old_put)
def put(self,i,t):
    if active is not None:
        item=dict(dtype=str(t.dtype),device=str(t.device),store_device=str(self.store_device),shape=list(t.shape),cache_dtype=self.dtype)
        if item not in active['cache_tensors']:active['cache_tensors'].append(item)
    return old_put(self,i,t)
PoseBranchCache.put=put

# Labels classify CUDA kernels in the profiler through their CPU launch ancestry.
for name in ['int8_linear','rms_rope','adaln']:
    instrument(ck,name,'kitchen_'+name)
for name in ['_modulated_norm','_gated_residual']:
    instrument(qm,name,name)
for name in ['quantize_int8_rowwise_convrot','quantize_int8_rowwise_convrot64','cutlass_int8_dequant','cublas_gemm_int8','dequantize_int8_linear','sage_sdpa_prequantized','sage_quantize_q','sage_quantize_k','sage_quantize_v']:
    if not hasattr(kc._C,name):continue
    old=getattr(kc._C,name)
    def call(*args,_name=name,_old=old,**kwargs):
        cm=torch.profiler.record_function('bench::cuda_'+_name) if profiling else contextlib.nullcontext()
        with cm:result=_old(*args,**kwargs)
        if active is not None:
            key=_name+(':'+str(result) if isinstance(result,bool) else '')
            active['dispatch'][key]=active['dispatch'].get(key,0)+1
        return result
    setattr(kc._C,name,call)


# Profile-only labels; delegate to the installed methods without changing math.
block_scopes=[]
def tensor_meta(value):
    if isinstance(value,torch.Tensor):
        return dict(dtype=str(value.dtype),shape=list(value.shape),stride=list(value.stride()))
    return None

def record_linear(role,x,y):
    item=dict(role=role,input=tensor_meta(x),output=tensor_meta(y))
    bucket=active.setdefault('linear_io',[])
    if item not in bucket:bucket.append(item)

def wrap_linear(module,role):
    old=module.forward
    @functools.wraps(old)
    def wrapped(x,*args,**kwargs):
        if not profiling:return old(x,*args,**kwargs)
        with region('part_'+role):out=old(x,*args,**kwargs)
        record_linear(role,x,out)
        return out
    module.forward=wrapped

def install_linear_scopes(model):
    if getattr(model,'_bench_linear_scopes',False):return
    for block in model.transformer_blocks:
        for name in ['to_q','to_k','to_v']:
            wrap_linear(getattr(block.attn,name),'qkv_'+name[-1])
        wrap_linear(block.attn.to_out[0],'attention_output')
        assert block.img_mlp.fused
        wrap_linear(block.img_mlp.gate_up,'mlp_gate_up')
    model._bench_linear_scopes=True

if a.chain_scopes:
    old_block=qm.QwenImage21TransformerBlock.forward
    @functools.wraps(old_block)
    def block_forward(self,x,mod,pe,attn_fn,prefix_len,transformer_options={}):
        if not profiling:return old_block(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        state=dict(norm=0,gated=0);block_scopes.append(state)
        try:
            label='target_block' if prefix_len==0 and x.shape[1]==5600 else 'prefix_block'
            with region(label):out=old_block(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
            assert state==dict(norm=2,gated=2),state
            return out
        finally:block_scopes.pop()
    qm.QwenImage21TransformerBlock.forward=block_forward
    for attr,key in [('_modulated_norm','norm'),('_gated_residual','gated')]:
        old=getattr(qm,attr)
        def labeled(*args,_old=old,_key=key,**kwargs):
            if not profiling or not block_scopes:return _old(*args,**kwargs)
            state=block_scopes[-1];index=state[_key];state[_key]+=1
            assert index in (0,1)
            role=('attention_' if index==0 else 'mlp_')+_key
            with region('part_'+role):return _old(*args,**kwargs)
        setattr(qm,attr,labeled)
    for cls,label in [(qm.Attention,'attention_chain'),(qm.SwiGLUFeedForward,'mlp_chain')]:
        old=cls.forward
        def labeled_forward(self,*args,_old=old,_label=label,**kwargs):
            if not profiling:return _old(self,*args,**kwargs)
            with region(_label):return _old(self,*args,**kwargs)
        cls.forward=labeled_forward
    old_input_act=comfy.ops.linear_input_act
    @functools.wraps(old_input_act)
    def input_act(linear,x,*args,**kwargs):
        if not profiling or not block_scopes:return old_input_act(linear,x,*args,**kwargs)
        with region('part_mlp_down'):out=old_input_act(linear,x,*args,**kwargs)
        record_linear('mlp_down',x,out)
        return out
    comfy.ops.linear_input_act=input_act

class Progress:
    def phase(self, message): print("PHASE " + message, flush=True)
    def __call__(self,*args,**kwargs):pass
    def prepare_sampling_callback(self,model,steps,x0_output_dict=None):
        def callback(step,x0,x,total_steps):
            if active is not None:active['steps_executed']+=1
        return callback
progress=Progress()
monitor_log=(OUT/'gpu.csv').open('w')
monitor_err=(OUT/'gpu-monitor.log').open('w')
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv','-l','1'],stdout=monitor_log,stderr=monitor_err)
results=[]
try:
    with torch.inference_mode():
        for index in range(a.runs):
            profiling=index==a.profile_index
            forward_samples.clear()
            active=dict(index=index,profiled=profiling,phases={},forward_count=0,forward_inputs=[],prefix_selections={},cache_tensors=[],dispatch={},steps_executed=0)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            print(f'BEGIN image {index} profiled={profiling}',flush=True)
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) if profiling else contextlib.nullcontext()
            with profiler:
                result=adapter.generate(**models,positive_prompt=(REF/'prompt.txt').read_text(),negative_prompt=(REF/'negative-prompt.txt').read_text(),width=1600,height=896,seed=42,sampler='er_sde',scheduler='beta',settings=settings,batch_count=1,lora_config=None,loaded_model=None,loaded_clip=None,reference_inputs=None,decode_image=True,return_vae=False,pid_capture_step=None,progress=progress)
                torch.cuda.synchronize()
            active['generation_seconds']=time.perf_counter()-begin
            image=Image.fromarray(np.clip(result.image[0].float().cpu().numpy()*255.,0,255).astype(np.uint8))
            image.save(OUT/f'image-{index}.png')
            active['saved_seconds']=time.perf_counter()-begin
            active['process_start_to_save_seconds']=time.perf_counter()-START
            active['rgb_sha256']=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            active['size']=image.size
            active['encoding_cache']=settings.get('encoding_cache')
            active['resolved_settings']=dict(settings)
            active['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            active['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
            if profiling:
                profiler.export_chrome_trace(str(OUT/f'trace-{index}.json'))
                # analyze.py joins launch correlations from this trace. CUDA
                # FunctionEvent.cpu_parent is empty and also includes GPU annotations;
                # it cannot reliably attribute or total the actual kernel activities.
            results.append(active)
            (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:active[k] for k in ['index','profiled','generation_seconds','saved_seconds','rgb_sha256','forward_count','steps_executed','peak_allocated_mib']}),flush=True)
            assert image.size==(1600,896) and active['steps_executed']==40
            assert settings.get('resolved_attention_mode')=='comfy_kitchen_int8'
            del result,image
            active=None
finally:
    monitor.terminate();monitor.wait();monitor_log.close();monitor_err.close()
print('DONE '+str(OUT),flush=True)
