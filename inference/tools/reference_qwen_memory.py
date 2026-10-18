#!/usr/bin/env python3
"""Original static ModelPatcher placement decisions; meta/tiny own weights only."""
from pathlib import Path
import resource
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
# Per-process hard ceilings are installed before Torch import; never change the parent.
resource.setrlimit(resource.RLIMIT_DATA, (2*1024**3, 2*1024**3))
resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(20)
print('memory oracle: importing frozen CPU-only constructor environment', flush=True)
loader = Path(__file__).with_name('generate_qwen_schema.py')
exec(compile(loader.read_text().split("\nwith torch.device('meta'):", 1)[0], str(loader), 'exec'))
print("memory oracle: imports finished", flush=True)
faulthandler.dump_traceback_later(15, repeat=True)
import comfy.model_patcher as mp
import comfy.model_management as mm
import gc
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
print("memory oracle: configuring single CPU thread", flush=True)
torch.set_num_threads(1)
print("memory oracle: installing meta guard", flush=True)
class MetaOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [v for v in tree_flatten((args, kwargs))[0] if isinstance(v, torch.Tensor)]
        device = kwargs.get("device")
        if any(not t.is_meta for t in tensors) or (device is not None and torch.device(device).type != "meta"):
            raise RuntimeError(f"Non-meta operation forbidden in placement oracle: {func}")
        if not tensors and any("Tensor" in str(r.type) for r in func._schema.returns) and device is None:
            raise RuntimeError(f"Unspecified allocation device forbidden: {func}")
        result = func(*args, **kwargs)
        assert all(v.is_meta for v in tree_flatten(result)[0] if isinstance(v, torch.Tensor))
        return result
# Default meta context also covers ModelPatcher lazy zero initialization. The
# dispatch guard rejects an explicit real device BEFORE its allocator is called.
meta_context = torch.device("meta")
meta_context.__enter__()
meta_guard = MetaOnly()
meta_guard.__enter__()
try:
    torch.empty((2**34,), device="cpu")
except RuntimeError as error:
    assert "Non-meta operation forbidden" in str(error)
else:
    raise RuntimeError("Meta allocation guard failed")
print("memory oracle: real allocation guard and hard process limits enabled", flush=True)

def make():
    with torch.device('meta'):
        core = QwenImage21Transformer2DModel(device='meta', dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        model = torch.nn.Module()
        model.diffusion_model = core
        model.manual_cast_dtype = torch.bfloat16
    patcher = mp.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    return model, patcher

model, patcher = make()
print('memory oracle: original full meta constructor ready', flush=True)
modules = []
for cost, size, name, module, params in patcher._load_list():
    modules.append(dict(name=name.removeprefix('diffusion_model.'), module_bytes=size,
        offload_bytes=cost, cast_weights=hasattr(module, 'comfy_cast_weights'),
        parameters=[f'{name}.{p}'.removeprefix('diffusion_model.') for p in params]))
cases = []
# Exercise the original selection code, replacing only real allocation/transfer
# effects. Meta tensors never become trained data and no GPU is touched.
for budget in [0.1, 16384., 419430400., 2**30, 8*2**30, 11*2**30, 13*2**30,
               14230249472., 14230249473., 1e32]:
    print(f'memory oracle: placement budget {budget}', flush=True)
    gc.collect()
    model, patcher = make()
    pinned = []
    patcher.patch_weight_to_device = lambda *args, **kw: None
    patcher.pin_weight_to_device = lambda key: pinned.append(key.removeprefix('diffusion_model.'))
    for module in model.modules():
        module.to = lambda *args, _m=module, **kw: _m
    full = budget > patcher.model_size()
    patcher.load(torch.device('cpu'), lowvram_model_memory=budget, full_load=full)
    assert all(p.is_meta for p in model.parameters())
    resident = sorted(n.removeprefix('diffusion_model.') for n,m in model.named_modules()
                      if getattr(m, 'comfy_patched_weights', False))
    cases.append(dict(budget=budget, full_load=full, resident=resident,
        loaded_bytes=model.model_loaded_weight_memory, buffer_bytes=model.model_offload_buffer_memory,
        offloaded_parameters=pinned))
    # Original partial unload, ascending order, strict stopping condition.
    for amount in [0., 1., 419430400., 2*2**30]:
        # Rebuild the exact initial state without sharing previous unloads.
        gc.collect()
        target, p2 = make()
        p2.patch_weight_to_device = lambda *args, **kw: None
        p2.pin_weight_to_device = lambda key: None
        for module in target.modules():
            module.to = lambda *args, _m=module, **kw: _m
        p2.load(torch.device('cpu'), lowvram_model_memory=budget, full_load=full)
        assert all(p.is_meta for p in target.parameters())
        freed = p2.partially_unload(torch.device('cpu'), amount)
        cases[-1].setdefault('unloads', []).append(dict(request=amount, freed=freed,
            loaded_bytes=target.model_loaded_weight_memory, buffer_bytes=target.model_offload_buffer_memory,
            resident=sorted(n.removeprefix('diffusion_model.') for n,m in target.named_modules()
                            if getattr(m, 'comfy_patched_weights', False))))
    for delta in [-2*2**30, -1., 0., 0.1, 33554432., 2**30]:
        gc.collect()
        target, p2 = make()
        p2.patch_weight_to_device = lambda *args, **kw: None
        p2.pin_weight_to_device = lambda key: None
        for module in target.modules():
            module.to = lambda *args, _m=module, **kw: _m
        p2.load(torch.device("cpu"), lowvram_model_memory=budget, full_load=full)
        p2.partially_load(torch.device("cpu"), delta)
        assert all(p.is_meta for p in target.parameters())
        cases[-1].setdefault("reloads", []).append(dict(extra=delta,
            loaded_bytes=target.model_loaded_weight_memory, buffer_bytes=target.model_offload_buffer_memory,
            resident=sorted(n.removeprefix("diffusion_model.") for n,m in target.named_modules()
                            if getattr(m, "comfy_patched_weights", False))))
# Budget formula is exercised by original load_models_gpu, without a real model.
class Fake:
    load_device = torch.device('cuda:0')
class Loaded:
    device = torch.device('cuda:0')
    model = Fake()
    def model_memory_required(self, device): return 0
    def model_memory(self): return 1
    def model_loaded_memory(self): return self.loaded
    def model_load(self, budget, **kwargs): self.selected = budget
    def model_offloaded_memory(self): return 0
    def is_dead(self): return False
Fake.is_dynamic = lambda self: False
Fake.is_clone = lambda self, other: False
Fake.model_patches_models = lambda self: []
Fake.loaded_ram_size = lambda self: 0
Fake.model_options = {}
Fake.loaded_size = lambda self: 0
Fake.model = None
budgets = []
originals = {n:getattr(mm,n) for n in ('LoadedModel','get_free_memory','free_memory','cleanup_models_gc')}
mm.cleanup_models_gc = lambda: None
mm.free_memory = lambda *args, **kwargs: []
mm.vram_state = mm.VRAMState.NORMAL_VRAM
mm.MIN_WEIGHT_MEMORY_RATIO = 0.0
mm.EXTRA_RESERVED_VRAM = 400*1024**2
for free in [0., 2**30, 8*2**30, 13*2**30]:
    for loaded in [0., 2**30]:
        for minimum in [None, 0., 2*2**30]:
            item = Loaded(); item.loaded = loaded
            mm.LoadedModel = lambda m: item
            mm.get_free_memory = lambda device: free
            mm.current_loaded_models = []
            mm.load_models_gpu([item.model], minimum_memory_required=minimum)
            budgets.append(dict(free=free, loaded=loaded, minimum=minimum, budget=item.selected))
# Same-model registration is removed BEFORE free_memory, without weight unload.
from types import SimpleNamespace
lifecycle=[]
detaches=[]
item=Loaded(); item.loaded=1024
item.model_finalizer=SimpleNamespace(detach=lambda: None)
Fake.is_clone=lambda self,other:self is other
Fake.detach=lambda self,unpatch_all=True:detaches.append(unpatch_all)
mm.LoadedModel=lambda m:item
mm.get_free_memory=lambda device:8*2**30
mm.free_memory=lambda *args,**kwargs:lifecycle.append(len(mm.current_loaded_models)) or []
mm.current_loaded_models=[]
mm.load_models_gpu([item.model])
mm.load_models_gpu([item.model])
assert lifecycle==[0,0] and detaches==[False]
mm.current_loaded_models = []
for name,value in originals.items(): setattr(mm,name,value)
sources = {}
for name in ('comfy/model_patcher.py','comfy/model_management.py','comfy/model_base.py','comfy/supported_models.py','comfy/system_memory.py','comfy/ops.py'):
    path = a.comfy / name
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = next(x['sha256'] for x in manifest if x['root']=='comfy' and x['path']==name)
    assert sha == expected
    sources[name] = sha
del model, patcher, target, p2, module
gc.collect()
faulthandler.cancel_dump_traceback_later()
meta_guard.__exit__(None,None,None)
meta_context.__exit__(None,None,None)
a.output.write_text(json.dumps(dict(schema=1, safety=dict(all_parameters_remained_meta=True, real_allocation_guard_tested=True, hard_data_limit_bytes=2*1024**3, hard_address_space_limit_bytes=8*1024**3, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), provenance='Original frozen full BF16 meta constructor and original static load/partial-unload/budget algorithms; transfers stubbed, no trained weights or GPU use',
    sources=sources, clone_lifecycle=dict(registered_models_seen_by_free_memory=lifecycle,detach_unpatch_all=detaches), modules=modules, cases=cases, budgets=budgets), indent=2)+'\n')
print(f'Captured {len(modules)} modules, {len(cases)} placements, {sum(len(c["unloads"]) for c in cases)} partial unloads, {len(budgets)} budgets, {sum(len(c["reloads"]) for c in cases)} reloads')
