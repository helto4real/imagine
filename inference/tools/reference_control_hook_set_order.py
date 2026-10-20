#!/usr/bin/env python3
"""Diagnose original ControlBase identity-set order; no inference or tensors."""
import __future__,argparse,ast,enum,hashlib,itertools,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--child',action='store_true');p.add_argument('--runs',type=int,default=8);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={}
ns=dict(enum=enum,Enum=enum.Enum,itertools=itertools,torch=NS(float32=object()))
spec=[('comfy/hooks.py',['EnumHookType','EnumHookScope','_HookRef','default_should_register','Hook','HookGroup','HookKeyframe','HookKeyframeGroup']),('comfy/controlnet.py',['StrengthType','ControlBase']),('comfy/sampler_helpers.py',['get_hooks_from_cond'])]
for relative,names in spec:
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest
 body=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef))and node.name in names]
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
ns['comfy']=NS(hooks=NS(HookGroup=ns['HookGroup']))
if a.child:
 cases=[]
 for count in [2,3,4,8,16]:
  for padding in [0,1,7,31,64]:
   # Own inert ControlBase instances vary only prior allocator occupancy.
   # No source hash/equality method, collection or hook algorithm is replaced.
   junk=[ns['ControlBase']()for _ in range(padding)]
   controls=[ns['ControlBase']()for _ in range(count)];references=[]
   for control in controls:
    hook=ns['Hook']();references.append(hook.hook_ref);group=ns['HookGroup']();group.add(hook);control.extra_hooks=group
   conditions=[dict(control=c)for c in controls]+[dict(control=controls[-1]),dict(control=controls[0])]
   full=ns['get_hooks_from_cond'](conditions,ns['HookGroup']())
   order=[references.index(hook.hook_ref)for hook in full.hooks];set_order=[controls.index(c)for c in set(controls)]
   assert sorted(order)==list(range(count))and order==set_order
   cases.append(dict(count=count,padding=padding,order=order,identity_hashes=[hash(c)for c in controls]))
 assert 'torch'not in sys.modules
 print(json.dumps(dict(cases=cases),separators=(',',':')))
else:
 assert a.output is not None and 2<=a.runs<=16
 runs=[];environment=os.environ.copy();environment['PYTHONHASHSEED']='0'
 for _ in range(a.runs):
  child=subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--comfy',str(a.comfy),'--child'],env=environment,text=True,capture_output=True,check=True);runs.append(json.loads(child.stdout))
 variation=[]
 for count in [2,3,4,8,16]:
  for padding in [0,1,7,31,64]:
   orders=sorted({tuple(c['order'])for r in runs for c in r['cases']if c['count']==count and c['padding']==padding})
   variation.append(dict(count=count,padding=padding,unique_orders=len(orders),orders=orders))
 result=dict(schema=1,sources=sources,python_version=list(sys.version_info[:3]),pythonhashseed='0',trained_model_weights=0,torch_python_imported=False,fixture_tensors=0,unmodified_called_bodies=['ControlBase.__init__','Hook.__init__','Hook.clone','HookGroup.add','HookGroup.clone','HookGroup.clone_and_combine','HookGroup.combine_all_hooks','get_hooks_from_cond'],unused_namespace_placeholders=['torch.float32 function default; no torch function is called'],runs=runs,variation=variation,changed_inputs=False,rust_policy_changed=False)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print('Original identity-set diagnostic:',len(runs),'processes;',sum(v['unique_orders']>1 for v in variation),'of',len(variation),'identical fixtures varied across processes; no tensors, weights or Rust-policy changes',flush=True)
