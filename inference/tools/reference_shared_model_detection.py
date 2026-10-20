#!/usr/bin/env python3
"""Whole locked detector on bounded shape-only synthetic checkpoints.

Only tiny explicit norm vectors contain data. Every model-sized placeholder is
an original Torch meta tensor, never allocated CPU/GPU model storage.
"""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
ns['json']=json
definitions('comfy/model_detection.py',['count_blocks','any_suffix_in','calculate_transformer_depth','detect_unet_config'])
source=a.comfy/'comfy/model_detection.py';tree=ast.parse(source.read_text());detector=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='detect_unet_config')
cases=[];construction=[]
def literal(node,env):
 try:return eval(compile(ast.Expression(node),'<synthetic detector constructor>','eval'),{'__builtins__':{},'list':list,'tuple':tuple},env)
 except (NameError,KeyError,TypeError,AttributeError,IndexError,ValueError):return None
def tensor(shape=None):return torch.empty((128,64,2,2,2)if shape is None else shape,device='meta',dtype=torch.float32)
def condition_keys(node,env):
 if isinstance(node,ast.Compare):
  out=[]
  for left,op in zip([node.left,*node.comparators[:-1]],node.ops):
   if isinstance(op,ast.In):
    value=literal(left,env)
    if isinstance(value,str)and value!='config':out.append(value)
  # Shape comparisons also name weights which must exist for this predicate.
  for child in ast.walk(node):
   if isinstance(child,ast.Subscript)and isinstance(child.value,ast.Name)and child.value.id=='state_dict':
    key=literal(child.slice,env)
    if isinstance(key,str):out.append(key)
  return out
 if isinstance(node,(ast.GeneratorExp,ast.ListComp)):
  gen=node.generators[0];items=literal(gen.iter,env)
  if isinstance(items,(list,tuple))and isinstance(gen.target,ast.Name):return [key for item in items for key in condition_keys(node.elt,dict(env,**{gen.target.id:item}))]
 if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='any_suffix_in':
  args=[literal(n,env)for n in node.args]
  if len(args)>=4 and isinstance(args[1],str)and isinstance(args[2],str)and args[3]:return [args[1]+args[2]+args[3][0]]
 return [key for child in ast.iter_child_nodes(node)for key in condition_keys(child,env)]
def apply_shape_predicates(node,env,sd):
 for compare in ast.walk(node):
  if not isinstance(compare,ast.Compare)or len(compare.ops)!=1 or not isinstance(compare.ops[0],ast.Eq):continue
  for expr,value in [(compare.left,compare.comparators[0]),(compare.comparators[0],compare.left)]:
   constant=literal(value,env)
   if not isinstance(constant,int):continue
   if isinstance(expr,ast.Subscript)and isinstance(expr.value,ast.Attribute)and expr.value.attr=='shape':
    base=expr.value.value
    if isinstance(base,ast.Subscript)and isinstance(base.value,ast.Name)and base.value.id=='state_dict':
     key=literal(base.slice,env);index=literal(expr.slice,env)
     if isinstance(key,str)and isinstance(index,int)and key in sd:
      shape=list(sd[key].shape);shape[index]=constant;sd[key]=tensor(shape)
def description(sd):
 out=[]
 for key,value in sd.items():
  if value.device.type=='meta':out.append([key,dict(shape=list(value.shape),kind=str(value.dtype),meta=True)])
  else:out.append([key,dict(value=encode(value),meta=False)])
 return out
def record(label,sd,prefix='',metadata=None,target=None,autofill=True):
 assert len(sd)<256
 # Only missing required keys are supplied. The original detector executes
 # unchanged; all auto-filled headers and its actual return line are retained.
 for attempt in range(80):
  try:ns['detect_unet_config'](sd,prefix,metadata);break
  except KeyError as e:
   if not autofill:break
   key=e.args[0]
   assert isinstance(key,str)and key not in sd,(label,key)
   sd[key]=tensor()
  except Exception as e:reject_resource_failure(e);break
 else:raise AssertionError(('Synthetic detector construction exceeded 80 missing keys',label))
 events=[]
 def trace(frame,event,arg):
  if frame.f_code is ns['detect_unet_config'].__code__ and event=='return':events.append(frame.f_lineno)
  return trace
 initial=description(sd);bytes_real=sum(t.numel()*t.element_size()for t in sd.values()if t.device.type!='meta');assert bytes_real<=8192
 old=sys.gettrace();sys.settrace(trace)
 try:captured=result(lambda:ns['detect_unet_config'](sd,prefix,metadata))
 finally:sys.settrace(old)
 cases.append(dict(label=label,target_if_line=target,prefix=prefix,metadata=encode(metadata),weights=initial,real_data_bytes=bytes_real,return_lines=events,**captured))
for prefix in ['', 'model.diffusion_model.']:
 env=dict(key_prefix=prefix,metadata=None,state_dict={},state_dict_keys=[])
 for node in detector.body:
  if isinstance(node,ast.Assign):
   value=literal(node.value,env)
   if value is not None:
    for target in node.targets:
     if isinstance(target,ast.Name):env[target.id]=value
  if not isinstance(node,ast.If):continue
  keys=list(dict.fromkeys(condition_keys(node.test,env)))
  # A negative-only guard is covered by the independent unknown/base cases.
  if not keys:continue
  sd={key:tensor()for key in keys};apply_shape_predicates(node.test,env,sd)
  construction.append(dict(line=node.lineno,prefix=prefix,keys=keys))
  record(f'branch-{node.lineno}',sd,prefix,target=node.lineno)
  record(f'branch-{node.lineno}/metadata',sd.copy(),prefix,dict(config='{"transformer":{"pdd_steps":7,"_source_test_z":[1,true,null],"_source_test_a":2}}'),target=node.lineno)
 record('unknown',{},prefix)
 record('legacy-unet',{prefix+'input_blocks.0.0.weight':tensor([128,4,3,3])},prefix)
 # Explicit fused/split Qwen and MOSS header alternatives.
 for fused in [False,True]:
  qwen={prefix+key:tensor([128,64])for key in ['txt_in.text_norm.weight','modulation.1.weight','transformer_blocks.0.attn.norm_q.weight','img_in.weight','proj_out.weight']}
  qwen[prefix+'transformer_blocks.0.img_mlp.'+('gate_up'if fused else'proj')+'.weight']=tensor([1024 if fused else 512,128])
  record('qwen21/'+str(fused),qwen,prefix)
  moss={prefix+'shared_block_modulation.weight':tensor([768,128]),prefix+'layers.0.proj.ffn_up_gate.weight':tensor([1024,128])}
  if fused:moss[prefix+'layers.0.proj.qkv.weight']=tensor([384,128])
  record('moss/'+str(fused),moss,prefix)
 # PiD integer-key variant lookup and its source ValueError.
 for channels in [16,32,64]:
  record('pid-v1.5/'+str(channels),{prefix+'lq_proj.latent_proj.0.weight':tensor([128,channels]),prefix+'lq_proj.pit_head.weight':tensor()},prefix)
 for scale in [0.,.4199,.42,.4201]:
  for pad in [False,True]:
   sd={prefix+'cap_embedder.1.weight':tensor([3840,64]),prefix+'noise_refiner.0.attention.k_norm.weight':tensor(),prefix+'layers.1.ffn_norm1.weight':torch.tensor([-scale,scale],dtype=torch.float32)}
   for i in range(3):sd[prefix+f'layers.{i}.own']=tensor()
   if pad:sd[prefix+'cap_pad_token']=tensor()
   record('learned-std/'+str(scale)+'/'+str(pad),sd,prefix)
 for width in [384,768,1024,1536]:
  for camera in [False,True]:
   for auxiliary in [False,True]:
    sd={prefix+'backbone.embeddings.patch_embeddings.projection.weight':tensor([width,3,14,14])}
    for i in range(4):sd[prefix+f'backbone.encoder.layer.{i}.own']=tensor()
    if camera:
     for key in ['backbone.embeddings.camera_token','backbone.encoder.layer.2.attention.q_norm.weight','cam_enc.token_norm.weight','cam_enc.pose_branch.fc2.weight','cam_dec.fc_t.weight']:sd[prefix+key]=tensor()
    if auxiliary:
     for key in ['head.scratch.refinenet1_aux.out_conv.weight','head.scratch.sky_output_conv2.0.weight']:sd[prefix+key]=tensor()
    record(f'da3/{width}/{camera}/{auxiliary}',sd,prefix)
 # Empty meta weights admit large logical dimensions without allocating data.
 # These values exercise Python integer true division before pixel-size sqrt,
 # and the original int(complex) failure for inconsistent decoder dimensions.
 for n in [1,2,3,2**26,2**27,2**28]:
  for delta in [-1,0,1]:
   x_width=3*n*n+3*n+delta
   sd={prefix+'cap_embedder.1.weight':tensor([3840,64]),prefix+'noise_refiner.0.attention.k_norm.weight':tensor(),prefix+'dec_net.cond_embed.weight':tensor([128,0]),prefix+'x_embedder.weight':tensor([0,x_width]),prefix+'dec_net.final_layer.linear.weight':tensor([3,0]),prefix+'dec_net.input_embedder.embedder.0.weight':tensor([0,19])}
   record(f'zimage-pixel-integer-division/{n}/{delta}',sd,prefix)
 for dec_in in [0,1,2,3,4]:
  sd={prefix+'cap_embedder.1.weight':tensor([3840,64]),prefix+'noise_refiner.0.attention.k_norm.weight':tensor(),prefix+'dec_net.cond_embed.weight':tensor([128,0]),prefix+'x_embedder.weight':tensor([0,12]),prefix+'dec_net.final_layer.linear.weight':tensor([3,0]),prefix+'dec_net.input_embedder.embedder.0.weight':tensor([0,dec_in])}
  record(f'zimage-pixel-decoder-max-freq/{dec_in}',sd,prefix)
 for propagation in [False,True]:
  sd={'detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight':tensor(),'detector.transformer.decoder.query_embed.weight':tensor()}
  if propagation:sd['detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight']=tensor()
  record('sam3/'+str(propagation),sd,prefix)
return_lines=sorted({line for case in cases for line in case['return_lines']if 'error'not in case})
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,construction=construction,observed_success_return_lines=return_lines,reference_return_statements=[node.lineno for node in ast.walk(detector)if isinstance(node,ast.Return)],trained_model_weights=0),separators=(',',':'))+'\n')
print('Captured',len(cases),'whole detector headers;',len(return_lines),'distinct successful return lines; trained model data allocated=0')
