#!/usr/bin/env python3
"""Original Diffusers VAE conversion on synthetic tensors and selected header metadata."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
name='comfy/diffusers_convert.py';payload=(a.comfy/name).read_bytes()
expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
assert hashlib.sha256(payload).hexdigest()==expected;sources[name]=dict(path=str(a.comfy/name),sha256=expected)
tree=ast.parse(payload);end=next(n.end_lineno for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='convert_vae_state_dict')
namespace['logging']=NS(debug=lambda *args:None)
nodes=[n for n in tree.body if n.lineno<=end and not isinstance(n,(ast.Import,ast.ImportFrom))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.comfy/name),'exec'),namespace)
cases=[]
for mode in ['diffusers2d','diffusers3d','native','seedvr2']:
 state={'decoder.up_blocks.0.resnets.0.norm1.weight':data((4,),'cpu',torch.float32,3)}
 for i,(sd,hf)in enumerate(namespace['vae_conversion_map']):state[hf+'weight']=data((3,4),'cpu',torch.float32,i)
 for i,(sd,hf)in enumerate(namespace['vae_conversion_map_attn']):state['encoder.mid_block.attentions.0.'+hf+'weight']=data((3,4),'cpu',torch.float32,i)
 state['bn.num_batches_tracked']=torch.tensor(4)
 if mode=='diffusers3d':state['encoder.down_blocks.0.downsamplers.0.conv.weight']=data((2,2,1,1,1),'cpu',torch.float32,7)
 if mode=='native':state={'encoder.norm_out.weight':data((4,),'cpu',torch.float32,3)}
 if mode=='seedvr2':state['decoder.up_blocks.2.upsamplers.0.upscale_conv.weight']=data((3,4),'cpu',torch.float32,7)
 original={k:encode(v)for k,v in state.items()}
 convert='decoder.up_blocks.0.resnets.0.norm1.weight'in state and'decoder.up_blocks.2.upsamplers.0.upscale_conv.weight'not in state
 out=namespace['convert_vae_state_dict'](state)if convert else state
 cases.append(dict(mode=mode,input=original,input_order=list(state),output={k:encode(v)for k,v in out.items()}))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,
    provenance='Original conversion function and maps, small initialized CPU operands; no trained model weights'),separators=(',',':'))+'\n')
print('Captured',len(cases),'VAE conversion groups',flush=True)
