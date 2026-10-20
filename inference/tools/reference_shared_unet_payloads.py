#!/usr/bin/env python3
"""Exact original UNet alias selection on tiny own CPU tensor payloads."""
import argparse,hashlib,itertools,json,marshal,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--child',action='store_true');opts=p.parse_args()
loader=Path(__file__).with_name('reference_shared_unet_names.py');scope={'__file__':str(loader)};saved=sys.argv;sys.argv=[str(loader),'--comfy',str(opts.comfy),'--child']
exec(compile(loader.read_text().split('\nif a.child:',1)[0],str(loader),'exec'),scope);sys.argv=saved
if opts.child:
 sys.path.insert(0,str(opts.site));import torch
 from reference_paths import reference_dir
 environment=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'environment.json').read_text())['torch_source_version'];assert torch.__version__==environment['__version__']and torch.version.git_version==environment['git_version']
 original=scope['ns'];cached={};exec(marshal.loads(marshal.dumps(scope['code'])),cached);cases=[]
 def encode(v):return dict(type='tensor',device=str(v.device),dtype=str(v.dtype),shape=list(v.shape),strides=list(v.stride()),requires_grad=v.requires_grad,payload_hex=bytes(v.detach().contiguous().reshape(-1).view(torch.uint8).tolist()).hex())
 with torch.no_grad():
  for origin,dtype,pattern in itertools.product(['source','marshalled'],[torch.float32,torch.float16,torch.bfloat16,torch.float64],['same-tensor','equal-values','different-values','two-views']):
   base=torch.tensor([101.125,202.25,303.5,404.75],dtype=dtype);left=base[::2]
   right=left if pattern=='same-tensor'else left.clone()if pattern=='equal-values'else torch.tensor([-11.5,12.75],dtype=dtype)if pattern=='different-values'else base[1::2]
   inputs=[left,right,base];config=dict(num_res_blocks=[1],channel_mult=[1],transformer_depth=[1],transformer_depth_output=[1,0],transformer_depth_middle=1)
   mapping=(original if origin=='source'else cached)['unet_to_diffusers'](config);state={'class_embedding.linear_1.weight':left,'add_embedding.linear_1.weight':right};out={};warnings=[]
   class Logging:
    @staticmethod
    def warning(message):warnings.append(message)
   scope['convert_original_loop'](mapping,state,out,Logging)
   result=list(out.items());assert len(result)==1;key,value=result[0]
   cases.append(dict(origin=origin,pattern=pattern,base=encode(base),inputs=[encode(v)for v in inputs],key=key,output=encode(value),output_aliases=[value is v for v in inputs],storage_aliases=[value.untyped_storage()._cdata==v.untyped_storage()._cdata for v in inputs],storage_offset=value.storage_offset(),remaining=list(state),warnings=warnings))
 print(json.dumps(dict(seed=os.environ['PYTHONHASHSEED'],cases=cases,torch_version=torch.__version__,torch_git=torch.version.git_version),separators=(',',':')))
else:
 assert opts.output is not None;runs=[]
 for seed in ['0','6']:
  env=os.environ.copy();env['PYTHONHASHSEED']=seed
  child=subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--site',str(opts.site),'--comfy',str(opts.comfy),'--child'],env=env,text=True,capture_output=True,check=True);runs.append(json.loads(child.stdout))
 result=dict(schema=1,sources=scope['sources'],trained_model_weights=0,device='cpu',maximum_base_elements=4,generated_images=0,loaded_full_model=False,runs=runs)
 opts.output.write_text(json.dumps(result,separators=(',',':'))+'\n');print('Captured',sum(len(r['cases'])for r in runs),'original CPU tensor-selection cases; at most four elements per own base, no trained weights or CUDA',flush=True)
