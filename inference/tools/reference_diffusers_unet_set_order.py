#!/usr/bin/env python3
"""Source-only Diffusers UNet set-order and alias-selection diagnostic."""
import __future__,argparse,ast,hashlib,json,logging,os,subprocess,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--child',action='store_true');a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={}
def tree(relative):
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest;return path,ast.parse(path.read_text())
path,utils=tree('comfy/utils.py');names=['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_RESNET','UNET_MAP_BASIC'];body=[v for v in utils.body if isinstance(v,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in v.targets)or isinstance(v,ast.FunctionDef)and v.name=='unet_to_diffusers'];assert len(body)==5
ns={};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
sd_path,sd=tree('comfy/sd.py');loader=next(v for v in sd.body if isinstance(v,ast.FunctionDef)and v.name=='load_diffusion_model_state_dict')
# Retain the original conversion loop, not a rewritten equivalent. The full
# model loader is deliberately not called: no models/weights/dependencies exist.
loops=[v for v in ast.walk(loader)if isinstance(v,ast.For)and isinstance(v.target,ast.Name)and v.target.id=='k'and isinstance(v.iter,ast.Name)and v.iter.id=='diffusers_keys'];assert len(loops)==1
loop=loops[0];loop_code=compile(ast.Module(body=[loop],type_ignores=[]),str(sd_path),'exec',flags=__future__.annotations.compiler_flag)
if a.child:
 config=dict(num_res_blocks=[1,1],channel_mult=[1,2],transformer_depth=[1,0],transformer_depth_output=[0,1,0,1],transformer_depth_middle=1)
 mapping=ns['unet_to_diffusers'](config)
 # Both documented external aliases map to the same internal weight key.
 # Own scalar sentinels demonstrate selection; there are no tensor allocations.
 initial={'class_embedding.linear_1.weight':101,'add_embedding.linear_1.weight':202}
 state=initial.copy();converted={};warnings=[]
 class Capture:
  @staticmethod
  def warning(message):warnings.append(message)
 scope=dict(diffusers_keys=mapping,sd=state,new_sd=converted,logging=Capture)
 exec(loop_code,scope)
 sets={name:list(ns[name])for name in ['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_BASIC']}
 assert converted['label_emb.0.0.weight']in [101,202]and not state
 assert 'torch'not in sys.modules
 print(json.dumps(dict(seed=os.environ['PYTHONHASHSEED'],sets=sets,map_items=list(mapping.items()),input=initial,remaining=state,selected=converted,warnings=warnings,config_after=config),separators=(',',':')))
else:
 assert a.output is not None
 runs=[]
 for seed in ['0','0','1','1']+[str(i)for i in range(2,16)]+['42','12345']:
  env=os.environ.copy();env['PYTHONHASHSEED']=seed
  child=subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--comfy',str(a.comfy),'--child'],env=env,text=True,capture_output=True,check=True);runs.append(json.loads(child.stdout))
 assert runs[0]==runs[1]and runs[2]==runs[3], 'Matched explicit seeds must remain reproducible'
 selected=sorted({r['selected']['label_emb.0.0.weight']for r in runs});assert selected==[101,202]
 result=dict(schema=1,sources=sources,python_version=list(sys.version_info[:3]),trained_model_weights=0,torch_python_imported=False,fixture_tensors=0,loaded_full_model=False,unmodified_source=['comfy/utils.py constant assignments','comfy/utils.py::unet_to_diffusers','comfy/sd.py::load_diffusion_model_state_dict original diffusers_keys conversion loop only'],unchanged_inputs=True,rust_policy_changed=False,seed_varied=True,selection_values=selected,runs=runs)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print('Original set-order diagnostic:',len(runs),'fresh processes; matched seeds identical; alias selections',selected,'; zero tensors/weights/Torch',flush=True)
