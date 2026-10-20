#!/usr/bin/env python3
"""Replay observed hook orders through original patch merging on one CPU float."""
import __future__,argparse,ast,enum,hashlib,itertools,json,logging,math,sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--orders',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.site));import torch
project=Path(__file__).resolve().parents[2];reference=reference_dir(project)
environment=json.loads((reference/'environment.json').read_text())['torch_source_version'];assert torch.__version__==environment['__version__']and torch.version.git_version==environment['git_version']
manifest=json.loads((reference/'source-manifest.json').read_text());sources={};ns=dict(torch=torch,enum=enum,Enum=enum.Enum,itertools=itertools,math=math,logging=logging)
def load(relative,names,methods=()):
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest;tree=ast.parse(path.read_text())
 body=[v for v in tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef))and v.name in names]
 for cls,name in methods:body.append(next(v for c in tree.body if isinstance(c,ast.ClassDef)and c.name==cls for v in c.body if isinstance(v,ast.FunctionDef)and v.name==name))
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
load('comfy/hooks.py',['EnumHookType','EnumHookScope','_HookRef','default_should_register','Hook','HookGroup','HookKeyframe','HookKeyframeGroup'])
load('comfy/weight_adapter/base.py',['WeightAdapterBase']);ns['weight_adapter']=NS(WeightAdapterBase=ns['WeightAdapterBase'])
load('comfy/model_management.py',['CPUState','is_intel_xpu','is_device_type','is_device_mps','device_supports_non_blocking','cast_to','cast_to_device'])
ns.update(cpu_state=ns['CPUState'].GPU,xpu_available=False,directml_enabled=False,args=NS(force_non_blocking=False,deterministic=False))
ns['comfy']=NS(model_management=NS(cast_to_device=ns['cast_to_device']))
load('comfy/model_patcher.py',[],[('ModelPatcher','get_combined_hook_patches')]);load('comfy/lora.py',['calculate_weight'])
diagnostic=json.loads(a.orders.read_text());orders=next(v['orders']for v in diagnostic['variation']if v['count']==2 and v['padding']==0);assert orders==[[0,1],[1,0]]
results=[]
with torch.no_grad():
 hooks=[ns['Hook'](),ns['Hook']()];diffs=[torch.tensor([16777216.],dtype=torch.float32),torch.tensor([-16777216.],dtype=torch.float32)]
 owner=NS(hook_patches={h.hook_ref:{'w':[(1.,(diff,),1.,None,None)]}for h,diff in zip(hooks,diffs)})
 for order in orders:
  group=ns['HookGroup']()
  for i in order:group.add(hooks[i])
  combined=ns['get_combined_hook_patches'](owner,group)
  output=ns['calculate_weight'](combined['w'],torch.tensor([1.],dtype=torch.float32),'synthetic.w')
  results.append(dict(order=order,values=output.tolist(),payload_hex=bytes(output.contiguous().view(torch.uint8).tolist()).hex()))
assert [v['values']for v in results]==[[0.],[1.]]
a.output.write_text(json.dumps(dict(schema=1,sources=sources,torch_version=torch.__version__,torch_git=torch.version.git_version,orders_sha256=hashlib.sha256(a.orders.read_bytes()).hexdigest(),trained_model_weights=0,fixture_tensor_shape=[1],input_diff_float_values=2,base_float_values_per_calculation=1,device='cpu',dtype='torch.float32',base_value=1.,diff_values=[16777216.,-16777216.],results=results,rust_policy_changed=False),indent=2)+'\n')
print('Original one-float CPU probe:',[(v['order'],v['values'])for v in results],'; zero model weights and no Rust-policy change',flush=True)
