#!/usr/bin/env python3
"""Original registered Python signatures and binding-only synthetic failures."""
from reference_paths import reference_dir
import argparse,hashlib,inspect,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rust-output',type=Path);p.add_argument('--check-rust',type=Path);a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
for name,expected in hashes.items():
 if name.startswith('comfy_kitchen/')and name.endswith('.py'):assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==expected
sys.path.insert(0,str(a.site));from comfy_kitchen.backends import eager,cuda,triton
signatures=[];cases=[];unresolved=[]
for backend,module in [('eager',eager),('cuda',cuda),('triton',triton)]:
 for operation in sorted(module._build_constraints()):
  fn=getattr(module,operation)
  try:sig=inspect.signature(fn)
  except Exception as error:unresolved.append(dict(backend=backend,operation=operation,error=type(error).__name__,message=str(error)));continue
  params=list(sig.parameters.values());names=[p.name for p in params];sig_record=dict(backend=backend,operation=operation,function_name=getattr(fn,'__qualname__',getattr(fn,'__name__',operation)),parameters=[dict(name=p.name,kind=p.kind.name,required=p.default is inspect.Parameter.empty)for p in params]);signatures.append(sig_record)
  def record(args,kw,label):
   try:fn(*args,**kw)
   except Exception as error:cases.append(dict(backend=backend,operation=operation,args_count=len(args),kwargs=list(kw),label=label,error=type(error).__name__,message=str(error)))
   else:raise AssertionError('A binding-only failure unexpectedly entered a valid call')
  required=[p for p in params if p.default is inspect.Parameter.empty and p.kind not in [inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD]]
  if required:record([],{},'missing_all')
  if not any(p.kind==inspect.Parameter.VAR_KEYWORD for p in params):
   record([] ,{'__own_unrecognized_keyword__':None},'unknown')
  positional=[p for p in params if p.kind in [inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD]]
  if positional and positional[0].kind==inspect.Parameter.POSITIONAL_OR_KEYWORD:record([None],{positional[0].name:None},'duplicate')
  if not any(p.kind==inspect.Parameter.VAR_POSITIONAL for p in params):record([None]*(len(positional)+1),{},'too_many')
  for omitted in required:
   if any(p.kind==inspect.Parameter.POSITIONAL_ONLY for p in params):continue
   kw={p.name:None for p in required if p.name!=omitted.name};record([],kw,'missing_'+omitted.name)
assert len(signatures)==135 and not unresolved
if a.rust_output or a.check_rust:
 out=['//! Generated from the locked registered Python function signatures.','//! Validation only; each body still owns its original defaults and operations.',"pub(crate) fn signature(backend:&str,operation:&str)->Option<(&'static [&'static str],usize)>{","Some(match (backend,operation){"]
 for s in signatures:
  assert s['function_name']==s['operation']
  assert all(p['kind']=='POSITIONAL_OR_KEYWORD' for p in s['parameters'])
  required=sum(p['required'] for p in s['parameters'])
  assert all(p['required']==(i<required) for i,p in enumerate(s['parameters']))
  names=[p['name'] for p in s['parameters']]
  out.append(f'({json.dumps(s["backend"])},{json.dumps(s["operation"])})=>(&{json.dumps(names)},{required}),')
 out+=['_=>return None,','})','}']
 generated=subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input='\n'.join(out)+'\n',text=True,capture_output=True,check=True).stdout
 if a.check_rust:assert a.check_rust.read_text()==generated,'Frozen registered Rust signatures differ from the original'
 if a.rust_output:a.rust_output.write_text(generated)
a.output.write_text(json.dumps(dict(scope='Locked registered function argument binding only; synthetic None sentinels; no tensor payloads',signatures=signatures,cases=cases,unresolved=unresolved),indent=2)+'\n');print('Original:',len(signatures),'signatures,',len(cases),'binding failures;',len(unresolved),'unresolved',flush=True)
