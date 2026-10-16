"""Matched LoRA benchmark through installed Comfy/AIO; timing-only output.

Ordinary stdout/stderr are discarded at the file-descriptor level before imports.
No existing server, history, user configuration, logs or image files are read.
Generated images are encoded in memory and discarded without inspection.
"""
import argparse
import ast
import functools
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy-root', type=Path, required=True)
p.add_argument('--aio-root', type=Path, required=True)
p.add_argument('--model-root', type=Path, required=True)
p.add_argument('--run', action='store_true')
p.add_argument('--trace-library', type=Path, help='Authorized CUDA kernel/copy measurements only')
p.add_argument('--verify-merge', action='store_true', help='Assert the ordinary FP16-weight / FP32-delta merge contract')
p.add_argument('--verify-lora', action='store_true', help='One additional LoRA run with application assertions')
a = p.parse_args()
# Reuse the source-defined fixture, never a previous request or runtime plan.
source = ast.parse((ROOT/'profiling/compare-lora-cfg1.py').read_text())
PROMPT = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in n.targets))
sequence = ['without_first', 'without_warm_1', 'without_warm_2',
            'with_first', 'with_warm_1', 'with_warm_2',
            'without_return_first', 'without_return_warm']
if a.verify_lora:
    sequence = ['with_verification']
if not a.run:
    print(json.dumps(dict(width=1600,height=896,steps=40,cfg=1,seed=42,
                         sampler='er_sde',scheduler='beta',sequence=sequence)))
    raise SystemExit(0)
assert not any(os.environ.get(k) for k in ['LD_PRELOAD','QWEN_ACTIVITY_TRACE'])
stable = 0
for _ in range(120):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(rows) == 1
    free, utilization = map(float, rows[0].split(','))
    stable = stable + 1 if free >= 13000 and utilization < 20 else 0
    if stable >= 4: break
    time.sleep(.5)
else: raise RuntimeError('GPU is not idle; no service was stopped')
OUT = Path(tempfile.mkdtemp(prefix='comfy-lora-cfg1-',dir=ROOT/'runs'))
report_fd = os.dup(1)
def report(item):
    os.write(report_fd, (json.dumps(item)+'\n').encode())
report(dict(results=str(OUT)))
with open(os.devnull,'w') as sink:
    os.dup2(sink.fileno(),1)
    os.dup2(sink.fileno(),2)
for key, subdir in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda-cache')]:
    os.environ[key] = str(OUT/subdir)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0,str(a.comfy_root.resolve()))
sys.argv = [sys.argv[0]]
stage = 'imports'
try:
    trace_library = None
    if a.trace_library:
        import ctypes
        os.environ['QWEN_ACTIVITY_TRACE'] = str(OUT/'activity.tsv')
        os.environ['QWEN_ACTIVITY_FLUSH'] = str(OUT/'flush.request')
        trace_library = ctypes.CDLL(str(a.trace_library.resolve()), mode=ctypes.RTLD_GLOBAL)
    import torch
    import numpy as np
    from PIL import Image
    import comfy.options
    comfy.options.enable_args_parsing(False)
    import folder_paths
    for category, paths in {'diffusion_models':['diffusion_models'], 'text_encoders':['text_encoders','clip'],
                            'vae':['vae'], 'embeddings':['embeddings'], 'loras':['loras']}.items():
        folder_paths.folder_names_and_paths[category] = ([str(a.model_root/x) for x in paths],folder_paths.supported_pt_extensions)
    folder_paths.set_output_directory(str(OUT/'unused-output'))
    folder_paths.set_temp_directory(str(OUT/'tmp'))
    folder_paths.set_input_directory(str(OUT/'unused-input'))
    for name, path in [('aio_bench',a.aio_root),('aio_bench.adapters',a.aio_root/'adapters')]:
        package=types.ModuleType(name); package.__path__=[str(path.resolve())]; sys.modules[name]=package
    from aio_bench.adapters.qwen_image_2_1 import QwenImage21Adapter
    from aio_bench.services import pipeline, qwen_image_2_1 as qwen
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21
    adapter=QwenImage21Adapter()
    settings=adapter.resolve_settings(model_settings=dict(family='qwen_image_2_1',reference_megapixels=1.6,
        edit_canvas='main controls',memory_policy='auto',cache_device='auto',cache_dtype='default',
        attention_mode='comfy_kitchen_int8',reference_upscale_method='lanczos',sampling_schedule='comfy',
        color_match_strength=.8,color_match=False,brightness_strength=0.,contrast_strength=0.,
        microcontrast_strength=0.,privacy_mode=True,use_zero_negative_conditioning=False),
        width=1600,height=896,steps=40,cfg=1.,sampler='er_sde',scheduler='beta')
    settings.update(max_side=1600,aspect_ratio='16:9',size_mode='use aspect ratio',multiple_value=32)
    settings=adapter.resolve_reference_settings(settings,None)
    models=dict(diffusion_model='diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors',
        text_encoder='clip/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors',
        vae='vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors')
    lora=dict(loras=[dict(name='qwen-image-21/lenovo_qwen21.safetensors',strength_model=1.,strength_clip=1.)])
    # Avoid a directory-wide LoRA inventory; only the explicitly requested file is needed.
    from aio_bench.services import lora_config
    lora_config._available_loras=lambda: [lora['loras'][0]['name']]
    metadata=dict(comfy_revision=subprocess.check_output(['git','-C',str(a.comfy_root),'rev-parse','HEAD'],text=True).strip(),
        aio_revision=subprocess.check_output(['git','-C',str(a.aio_root),'rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,kitchen=importlib.metadata.version('comfy-kitchen'),
        cpu_threads=torch.get_num_threads(),width=1600,height=896,steps=40,cfg=1,seed=42,
        sampler='er_sde',scheduler='beta',attention='comfy_kitchen_int8',profiled=bool(a.trace_library),
        sequence=sequence,measurement='adapter.generate through PNG encoded in memory, discarded')
    (OUT/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Fail closed if the selected adapter was silently ignored. These checks
    # expose no weights, keys, prompts, images or runtime metadata.
    if a.verify_lora:
        import comfy.sd
        original_load = comfy.sd.load_lora_for_models
        def checked_load(model, clip, weights, strength_model, strength_clip, **kwargs):
            assert strength_model == strength_clip == 1.0
            assert not model.patches
            patched, patched_clip = original_load(model,clip,weights,strength_model,strength_clip,**kwargs)
            assert patched.patches
            assert patched.patches_uuid != model.patches_uuid
            return patched, patched_clip
        comfy.sd.load_lora_for_models = checked_load
        original_prepare = qwen.prepare_models
        def checked_prepare(*args,**kwargs):
            prepared = original_prepare(*args,**kwargs)
            assert prepared[0].patches
            return prepared
        qwen.prepare_models = checked_prepare
    merge_calls = [0, 0]
    if a.verify_merge:
        import comfy.model_patcher
        from comfy.weight_adapter.lora import LoRAAdapter
        from comfy_kitchen.tensor import QuantizedTensor
        assert comfy.model_patcher.CoreModelPatcher is comfy.model_patcher.ModelPatcher
        assert not torch.backends.cuda.matmul.allow_tf32
        original_calculate = LoRAAdapter.calculate_weight
        @functools.wraps(original_calculate)
        def checked_calculate(self, weight, key, strength, strength_model, offset, function,
                              intermediate_dtype=torch.float32, original_weight=None):
            assert weight.dtype == torch.float16
            assert intermediate_dtype == torch.float32
            merge_calls[0] += 1
            return original_calculate(self, weight, key, strength, strength_model, offset, function,
                                      intermediate_dtype, original_weight)
        LoRAAdapter.calculate_weight = checked_calculate
        original_requantize = QuantizedTensor.requantize_from_float
        @functools.wraps(original_requantize)
        def checked_requantize(self, tensor, **kwargs):
            assert tensor.dtype == torch.float16
            assert self._params.convrot and self._params.convrot_groupsize == 256
            assert kwargs['scale'] == 'recalculate' and kwargs['stochastic_rounding'] > 0
            merge_calls[1] += 1
            return original_requantize(self, tensor, **kwargs)
        QuantizedTensor.requantize_from_float = checked_requantize
    active=None
    def instrument(obj,attr,name):
        old=getattr(obj,attr)
        @functools.wraps(old)
        def call(*args,**kwargs):
            torch.cuda.synchronize(); start=time.perf_counter()
            result=old(*args,**kwargs)
            torch.cuda.synchronize()
            active['phases'].setdefault(name,[]).append(time.perf_counter()-start)
            return result
        setattr(obj,attr,call)
    instrument(qwen,'prepare_models','prepare_models')
    instrument(pipeline,'sample_with_comfy_ksampler','sampling')
    instrument(pipeline,'decode_latent','vae')
    old_encode=TextEncodeQwenImage21.execute
    @classmethod
    def encode(cls,*args,**kwargs):
        torch.cuda.synchronize(); start=time.perf_counter()
        result=old_encode(*args,**kwargs)
        torch.cuda.synchronize()
        active['phases'].setdefault('conditioning',[]).append(time.perf_counter()-start)
        return result
    TextEncodeQwenImage21.execute=encode
    class Progress:
        def phase(self,*args,**kwargs):pass
        def __call__(self,*args,**kwargs):pass
        def prepare_sampling_callback(self,model,steps,x0_output_dict=None):
            def callback(step,x0,x,total_steps):
                active['steps_executed']+=1
            return callback
    results=[]
    with torch.inference_mode():
        for name in sequence:
            stage=name
            active=dict(name=name,phases={},steps_executed=0,start_ns=time.monotonic_ns())
            torch.cuda.synchronize(); start=time.perf_counter()
            result=adapter.generate(**models,positive_prompt=PROMPT,negative_prompt='',width=1600,height=896,
                seed=42,sampler='er_sde',scheduler='beta',settings=settings,batch_count=1,
                lora_config=lora if name.startswith('with_') else None,loaded_model=None,loaded_clip=None,
                reference_inputs=None,decode_image=True,return_vae=False,pid_capture_step=None,progress=Progress())
            torch.cuda.synchronize()
            active['generation_seconds']=time.perf_counter()-start
            with io.BytesIO() as sink:
                image=Image.fromarray(np.clip(result.image[0].float().cpu().numpy()*255.,0,255).astype(np.uint8))
                image.save(sink,format='PNG')
                del image
            active['total_seconds']=time.perf_counter()-start
            active['end_ns']=time.monotonic_ns()
            assert active['steps_executed']==40
            assert settings.get('resolved_attention_mode')=='comfy_kitchen_int8'
            results.append(active)
            (OUT/'timings.json').write_text(json.dumps(results,indent=2)+'\n')
            report(active)
            del result
    if a.verify_merge:
        assert all(merge_calls)
    if trace_library:
        checkpoint=OUT/'flush.request'
        checkpoint.touch()
        deadline=time.monotonic()+15
        while checkpoint.exists() and time.monotonic()<deadline:
            time.sleep(.1)
        assert not checkpoint.exists()
    report(dict(complete=True))
except BaseException as exc:
    # Only type and source locations; exception messages may contain runtime data.
    import traceback
    report(dict(failed_stage=stage,error_type=type(exc).__name__,
                source_locations=[dict(file=f.filename,line=f.lineno,function=f.name)
                                  for f in traceback.extract_tb(exc.__traceback__)]))
    raise SystemExit(1)
