#!/usr/bin/env python3
"""Regenerate source-derived Rust data in an owned temporary directory and compare."""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--comfy',type=Path,help='Also check locked Stage 3 formula generation');p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];tools=project/'inference/tools';source=project/'inference/src';q=a.site/'comfy_kitchen/backends/triton/quantization.py';jobs=[]
def job(script,target,*args):jobs.append((script,target,list(map(str,args))))
job('generate_triton_int8_program.py','triton_frontend_int8_program.rs','--source',q)
job('generate_triton_int8_program.py','triton_frontend_w4a8_program.rs','--family','w4a8','--source',a.site/'comfy_kitchen/backends/triton/w4a8_int8.py')
job('generate_triton_mxfp8_program.py','triton_frontend_mxfp8_program.rs','--quantization',q,'--standard',a.site/'triton/language/standard.py')
for family,target in [('quantize','nvfp4'),('dequantize','dequant_nvfp4')]:job('generate_triton_nvfp4_program.py',f'triton_frontend_{target}_program.rs','--source',q,'--family',family)
for family in ['rope','rms_rope','na']:job(f'generate_triton_{family}_program.py',f'triton_frontend_{family}_program.rs','--source',a.site/f'comfy_kitchen/backends/triton/{family}.py')
job('generate_triton_nvfp4_sources.py','triton_nvfp4_sources.rs','--site',a.site)
job('generate_python_decimal_digits.py','python_decimal_digits.rs')
job('generate_triton_scalar_sources.py','triton_scalar_sources.rs','--site',a.site)
job('generate_triton_environment_names.py','triton_environment_names.rs','--source',project/'inference/native/triton')
job('generate_native_argument_signatures.py','native_argument_signatures.rs','--site',a.site)
if a.comfy is not None:
 job('generate_weights_only_globals.py','checkpoint/weights_only_globals_data.rs','--site',a.site)
 job('generate_torch_error_types.py','python_error_types.rs','--source',a.root/'sources/pytorch/torch/csrc/Exceptions.cpp')
 job('generate_ipndm_v_formulas.py','sampling/ipndm_v_formulas.rs','--comfy',a.comfy)
 job('generate_model_detection.py','checkpoint/detector_program.rs','--comfy',a.comfy)
 job('generate_model_detection.py','checkpoint/diffusers_program.rs','--function','unet_config_from_diffusers_unet','--comfy',a.comfy)
 job('generate_model_registry.py','checkpoint/registry_data.rs','--comfy',a.comfy)
 job('generate_unet_names.py','checkpoint/unet_names_program.rs','--comfy',a.comfy)
report=dict(status='in progress',source_kind='frozen source and native module metadata only; no model or runtime data',files={})
with tempfile.TemporaryDirectory(prefix='imagine-generated-restore-',dir=a.output.parent)as own:
 for script,target,args in jobs:
  restored=Path(own)/target;restored.parent.mkdir(parents=True,exist_ok=True);subprocess.run([sys.executable,'-S','-B',str(tools/script),*args,'--output',str(restored)],check=True)
  assert restored.read_bytes()==(source/target).read_bytes(),f'Source restoration differs: {target}'
  report['files'][target]=hashlib.sha256(restored.read_bytes()).hexdigest()
 signature=Path(own)/'signatures.json'
 subprocess.run([sys.executable,'-S','-B',str(tools/'reference_backend_signatures.py'),'--site',str(a.site),'--output',str(signature),'--check-rust',str(source/'backend_signatures.rs')],check=True)
 report['files']['backend_signatures.rs']=hashlib.sha256((source/'backend_signatures.rs').read_bytes()).hexdigest()
 constraints=Path(own)/'constraints.json'
 subprocess.run([sys.executable,'-S','-B',str(tools/'reference_backend_contract.py'),'--site',str(a.site),'--output',str(constraints),'--check-rust',str(source/'backend_contract/tables.rs')],check=True)
 report['files']['backend_contract/tables.rs']=hashlib.sha256((source/'backend_contract/tables.rs').read_bytes()).hexdigest()
 if a.comfy is not None:
  table=Path(own)/'schedule-tables.json'
  subprocess.run([sys.executable,'-S','-B',str(tools/'generate_shared_schedule_tables.py'),'--comfy',str(a.comfy),'--output',str(table)],check=True)
  retained=project/'inference/resources/shared-schedule-tables.json'
  assert table.read_bytes()==retained.read_bytes(),'Source scheduler tables differ'
  report['files']['resources/shared-schedule-tables.json']=hashlib.sha256(retained.read_bytes()).hexdigest()
  # Deterministic own-metadata oracles. Random-address/hash-state corpora are
  # replayed with their captured inputs instead of silently replacing salts.
  for script,name,extra in [
   ('reference_shared_deepcopy.py','shared-deepcopy.json',[]),
   ('reference_shared_deepcopy.py','shared-general-deepcopy.json',['--general-keys']),
   ('reference_shared_patcher_constructor.py','shared-patcher-constructor.json',[]),
   ('reference_shared_whole_clone.py','shared-whole-clone.json',[]),
   ('reference_shared_weak_finalizers.py','shared-weak-finalizers.json',[]),
   ('reference_shared_patch_calls.py','shared-patch-calls.json',[]),
   ('reference_shared_patch_calls.py','shared-cast-options.json',['--cast-options']),
   ('reference_shared_load_selection.py','shared-load-selection.json',[]),
   ('reference_shared_load_selection.py','shared-fallible-load-selection.json',['--fallible']),
   ('reference_shared_lifecycle.py','shared-lifecycle.json',[]),
   ('reference_shared_deepclone.py','shared-deepclone.json',[]),
   ('reference_shared_guider_outer.py','shared-guider-outer.json',[]),
   ('reference_shared_nested_copy.py','shared-nested-copy.json',[]),
   ('reference_shared_guider_inner.py','shared-guider-inner.json',[]),
   ('reference_shared_guider_state.py','shared-guider-state.json',[]),
   ('reference_shared_guider_prediction.py','shared-guider-prediction.json',[]),
   ('reference_shared_guider_sample.py','shared-guider-sample.json',[]),
   ('reference_shared_registry_fields.py','shared-registry-fields.json',[]),
   ('reference_shared_model_access.py','shared-model-access.json',[]),
   ('reference_shared_memory_queries.py','shared-memory-queries.json',[]),
   ('reference_shared_loaded_errors.py','shared-loaded-errors.json',[]),
   ('reference_shared_loaded_query.py','shared-loaded-query.json',[]),
   ('reference_shared_physical_cleanup.py','shared-physical-cleanup.json',[]),
   ('reference_shared_free_queries.py','shared-free-queries.json',[]),
   ('reference_shared_free_owners.py','shared-free-owners.json',[]),
   ('reference_shared_loaded_unload.py','shared-loaded-unload.json',[]),
   ('reference_shared_loaded_load.py','shared-loaded-load.json',[]),
   ('reference_shared_load_completion.py','shared-load-completion.json',[]),
   ('reference_shared_memory_helpers.py','shared-memory-helpers.json',[]),
   ('reference_shared_memory_gc.py','shared-memory-gc.json',[]),
   ('reference_shared_dtype_size.py','shared-dtype-size.json',[]),
   ('reference_shared_dtype_size.py','shared-dtype-size-kinds.json',['--native-kinds']),
   ('reference_shared_dtype_size.py','shared-dtype-size-registered.json',['--registered-types']),
   ('reference_shared_async_discard.py','shared-async-discard.json',[]),
   ('reference_shared_oom_dispatch.py','shared-oom-dispatch.json',[]),
   ('reference_shared_model_pinning.py','shared-model-pinning.json',[]),
   ('reference_shared_model_pinning.py','shared-model-pinning-native-sets.json',['--native-sets']),
   ('reference_shared_pinned_key_set.py','shared-pinned-key-set.json',[]),
   ('reference_shared_pin_registry.py','shared-pin-registry.json',[]),
   ('reference_shared_sampler_factory.py','shared-sampler-factory.json',[]),
   ('reference_shared_max_denoise.py','shared-max-denoise.json',[]),
   ('reference_shared_sampler_state.py','shared-sampler-state.json',[]),
   ('reference_shared_inpaint_state.py','shared-inpaint-state.json',[]),
   ('reference_shared_dpm_call.py','shared-dpm-call.json',[]),
   ('reference_shared_public_sample.py','shared-public-sample.json',[]),
   ('reference_shared_public_sample_errors.py','shared-public-sample-errors.json',[]),
  ]:
   restored=Path(own)/name
   subprocess.run([sys.executable,'-S','-B',str(tools/script),'--comfy',str(a.comfy),*extra,'--output',str(restored)],check=True)
   retained=project/'inference/fixtures'/name
   assert restored.read_bytes()==retained.read_bytes(),f'Own metadata restoration differs: {name}'
   report['files']['fixtures/'+name]=hashlib.sha256(retained.read_bytes()).hexdigest()
  restored=Path(own)/'shared-exception-context.json'
  subprocess.run([sys.executable,'-S','-B',str(tools/'reference_shared_exception_context.py'),'--output',str(restored)],check=True)
  retained=project/'inference/fixtures/shared-exception-context.json'
  assert restored.read_bytes()==retained.read_bytes(),'Own exception context restoration differs'
  report['files']['fixtures/shared-exception-context.json']=hashlib.sha256(retained.read_bytes()).hexdigest()
  restored=Path(own)/'shared-exception-roots.json'
  subprocess.run([sys.executable,'-S','-B',str(tools/'reference_shared_exception_roots.py'),'--output',str(restored)],check=True)
  retained=project/'inference/fixtures/shared-exception-roots.json'
  assert restored.read_bytes()==retained.read_bytes(),'Own exception frame restoration differs'
  report['files']['fixtures/shared-exception-roots.json']=hashlib.sha256(retained.read_bytes()).hexdigest()
  restored=Path(own)/'shared-real-finalizers.json'
  subprocess.run([sys.executable,'-S','-B',str(tools/'reference_shared_real_finalizers.py'),'--output',str(restored)],check=True)
  retained=project/'inference/fixtures/shared-real-finalizers.json'
  assert restored.read_bytes()==retained.read_bytes(),'Own real finalizer restoration differs'
  report['files']['fixtures/shared-real-finalizers.json']=hashlib.sha256(retained.read_bytes()).hexdigest()
  restored=Path(own)/'shared-unicode.json'
  subprocess.run([sys.executable,'-S','-B',str(tools/'reference_shared_unicode.py'),'--output',str(restored)],check=True)
  retained=project/'inference/fixtures/shared-unicode.json'
  assert restored.read_bytes()==retained.read_bytes(),'Own Unicode restoration differs'
  report['files']['fixtures/shared-unicode.json']=hashlib.sha256(retained.read_bytes()).hexdigest()
report['status']='passed';a.output.write_text(json.dumps(report,indent=2)+'\n');print('Restored and matched',len(report['files']),'source-derived Rust files',flush=True)
