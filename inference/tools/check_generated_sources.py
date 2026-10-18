#!/usr/bin/env python3
"""Regenerate source-derived Rust data in an owned temporary directory and compare."""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];tools=project/'inference/tools';source=project/'inference/src';q=a.site/'comfy_kitchen/backends/triton/quantization.py';jobs=[]
def job(script,target,*args):jobs.append((script,target,list(map(str,args))))
job('generate_triton_int8_program.py','triton_frontend_int8_program.rs','--source',q)
job('generate_triton_int8_program.py','triton_frontend_w4a8_program.rs','--family','w4a8','--source',a.site/'comfy_kitchen/backends/triton/w4a8_int8.py')
job('generate_triton_mxfp8_program.py','triton_frontend_mxfp8_program.rs','--quantization',q,'--standard',a.site/'triton/language/standard.py')
for family,target in [('quantize','nvfp4'),('dequantize','dequant_nvfp4')]:job('generate_triton_nvfp4_program.py',f'triton_frontend_{target}_program.rs','--source',q,'--family',family)
for family in ['rope','rms_rope','na']:job(f'generate_triton_{family}_program.py',f'triton_frontend_{family}_program.rs','--source',a.site/f'comfy_kitchen/backends/triton/{family}.py')
job('generate_triton_nvfp4_sources.py','triton_nvfp4_sources.rs','--site',a.site)
job('generate_python_decimal_digits.py','python_decimal_digits.rs')
job('generate_triton_scalar_sources.py','triton_scalar_sources.rs','--site',a.site)
job('generate_triton_environment_names.py','triton_environment_names.rs','--source',a.root/'sources/triton-candidate')
job('generate_native_argument_signatures.py','native_argument_signatures.rs','--site',a.site)
report=dict(status='in progress',source_kind='frozen source and native module metadata only; no model or runtime data',files={})
with tempfile.TemporaryDirectory(prefix='imagine-generated-restore-',dir=a.output.parent)as own:
 for script,target,args in jobs:
  restored=Path(own)/target;subprocess.run([sys.executable,'-S','-B',str(tools/script),*args,'--output',str(restored)],check=True)
  assert restored.read_bytes()==(source/target).read_bytes(),f'Source restoration differs: {target}'
  report['files'][target]=hashlib.sha256(restored.read_bytes()).hexdigest()
 signature=Path(own)/'signatures.json'
 subprocess.run([sys.executable,'-S','-B',str(tools/'reference_backend_signatures.py'),'--site',str(a.site),'--output',str(signature),'--check-rust',str(source/'backend_signatures.rs')],check=True)
 report['files']['backend_signatures.rs']=hashlib.sha256((source/'backend_signatures.rs').read_bytes()).hexdigest()
report['status']='passed';a.output.write_text(json.dumps(report,indent=2)+'\n');print('Restored and matched',len(report['files']),'source-derived Rust files',flush=True)
