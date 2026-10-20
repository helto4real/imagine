#!/usr/bin/env python3
"""Selected header metadata through original convert_old_quants, CPU only.

Trained weight entries are opaque object sentinels. Only the small quantization
metadata entries are tensors; no model-sized tensor or trained payload is read.
"""
import argparse,ast,gc,hashlib,json,os,sys,struct,tempfile
from pathlib import Path
from inspect_native_av_headers import FILES
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.site))
import torch
source=a.comfy/'comfy/utils.py';tree=ast.parse(source.read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='convert_old_quants');namespace=dict(torch=torch,json=json)
exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(source),'exec'),namespace)
cases=[]
for role,relative in FILES.items():
 path=a.models/relative
 with path.open('rb')as stream:
  n=struct.unpack('<Q',stream.read(8))[0];assert n<=16*1024*1024;header_bytes=stream.read(n);header=json.loads(header_bytes);meta=header.pop('__metadata__',{})
  quant={}
  for name,entry in header.items():
   if name.endswith('.comfy_quant'):
    start,end=entry['data_offsets'];assert end-start<=16384;stream.seek(8+n+start);quant[name]=stream.read(end-start)
 names=list(header);quant_text=meta.get('_quantization_metadata');meta_names=list(meta)
 for custom_kind in ['none','false','opaque']:
  state={name:torch.tensor(list(quant[name]),dtype=torch.uint8)if name in quant else object()for name in names}
  metadata={name:quant_text if name=='_quantization_metadata'else object()for name in meta_names}
  old=dict(state);old_meta=dict(metadata)
  # Source load_clip/load_diffusion_model choose conversion by exact None,
  # not by the truth value of custom_operations.
  custom=None if custom_kind=='none'else(False if custom_kind=='false'else object())
  output,returned_meta=namespace['convert_old_quants'](state,'',metadata)if custom is None else(state,metadata)
  output_quant={name:bytes(value.tolist()).hex()for name,value in output.items()if name.endswith('.comfy_quant')}
  assert all(output[name]is old[name]for name in names if name not in quant)
  assert all(returned_meta[name]is old_meta[name]for name in meta_names)
  cases.append(dict(role=role,header_sha256=hashlib.sha256(header_bytes).hexdigest(),custom_operations=custom_kind,names=names,metadata_keys=meta_names,quantization_metadata=quant_text,initial_quant={name:data.hex()for name,data in quant.items()},output_keys=list(output),output_quant=output_quant,changed_quant=[name for name in output_quant if output[name]is not old.get(name)],root_same=output is state,metadata_same=returned_meta is metadata))
annotation_source=a.comfy/'comfy/storage.py';tree=ast.parse(annotation_source.read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='annotate_state_dict');namespace['os']=os
exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(annotation_source),'exec'),namespace)
def annotation_case(shape,path_a,path_b):
 base=torch.zeros(shape);view=base.view(-1);same=base.to(dtype=base.dtype);clone=base.clone();cast=base.to(torch.int64)
 namespace['annotate_state_dict']({'base':base,'view':view},path_a)
 def paths():return {name:getattr(value.untyped_storage(),'_comfy_source_path',None)for name,value in [('base',base),('view',view),('same',same),('clone',clone),('cast',cast)]}
 before=paths();before_identity=base.untyped_storage()._comfy_source_path is view.untyped_storage()._comfy_source_path
 namespace['annotate_state_dict']({'view':view},path_b);after=paths()
 weak=base.untyped_storage()._weak_ref();del base,same;gc.collect();alive_with_view=not torch.UntypedStorage._expired(weak);del view;gc.collect();expired=torch.UntypedStorage._expired(weak);torch.UntypedStorage._free_weak_ref(weak)
 return dict(shape=shape,path_a=os.path.realpath(path_a),path_b=os.path.realpath(path_b),before=before,after=after,same_path_object=before_identity,alive_with_view=alive_with_view,expired_after_release=expired)
with tempfile.TemporaryDirectory(prefix='imagine-own-av-source-')as own:
 annotation_cases=[annotation_case(shape,own+'/first.safetensors',own+'/second.safetensors')for shape in [[],[0],[2,2]]]
a.output.write_text(json.dumps(dict(schema=1,sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [source,annotation_source]},cuda_work=False,trained_weight_tensors_allocated=0,trained_weight_payload_bytes_read=0,opaque_weight_sentinels=True,cases=cases,storage_annotations=annotation_cases),separators=(',',':'))+'\n');print('Captured',len(cases),'selected AV metadata histories and',len(annotation_cases),'source storage histories; no trained weight tensors or payload')
