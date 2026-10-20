#!/usr/bin/env python3
"""Original UNet maps, hash mechanics and conversion mutations; no Torch."""
import __future__,argparse,ast,copy,ctypes,dis,hashlib,itertools,json,marshal,os,subprocess,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--child',action='store_true');a=p.parse_args();assert sys.version_info[:3]==(3,12,13)and sys.hash_info.algorithm=='siphash13'and sys.hash_info.cutoff==0
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={}
def tree(relative):
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest;return path,ast.parse(path.read_text())
path,utils=tree('comfy/utils.py');names=['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_RESNET','UNET_MAP_BASIC'];assignments={}
body=[]
for node in utils.body:
 if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in node.targets):
  body.append(node);assignments[node.targets[0].id]=node.value
 elif isinstance(node,ast.FunctionDef)and node.name=='unet_to_diffusers':body.append(node)
code=compile(ast.Module(body=body,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag);ns={};exec(code,ns)
sd_path,sd=tree('comfy/sd.py');loader=next(v for v in sd.body if isinstance(v,ast.FunctionDef)and v.name=='load_diffusion_model_state_dict');loops=[v for v in ast.walk(loader)if isinstance(v,ast.For)and isinstance(v.target,ast.Name)and v.target.id=='k'and isinstance(v.iter,ast.Name)and v.iter.id=='diffusers_keys'];assert len(loops)==1
wrapper=ast.FunctionDef(name='convert_original_loop',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in ['diffusers_keys','sd','new_sd','logging']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=loops,decorator_list=[])
loop_code=compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(sd_path),'exec',flags=__future__.annotations.compiler_flag);loop_namespace={};exec(loop_code,loop_namespace);convert_original_loop=loop_namespace['convert_original_loop']
def enc(value):
 if isinstance(value,dict):return dict(type='dict',items=[[k,enc(v)]for k,v in value.items()])
 if isinstance(value,(tuple,list)):return dict(type=type(value).__name__,items=[enc(v)for v in value])
 return value
if a.child:
 secret=bytes((ctypes.c_ubyte*24).in_dll(ctypes.pythonapi,'_Py_HashSecret'))
 strings=['','ascii','é','å','Ł','漢字','🙂','mixedé𐍈','\0']+['x'*i for i in range(1,81)]+[x for n in names if isinstance(assignments[n],ast.Set)for v in assignments[n].elts for x in(ast.literal_eval(v)if isinstance(v,ast.Tuple)else(ast.literal_eval(v),))]
 strings=list(dict.fromkeys(strings));string_hashes=[[v,hash(v)]for v in strings];byte_hashes=[[bytes(range(i)).hex(),hash(bytes(range(i)))]for i in range(81)]
 tuples=[tuple(strings[:i])for i in range(6)]+[ast.literal_eval(v)for v in assignments['UNET_MAP_BASIC'].elts]
 tuple_hashes=[dict(value=v,hashes=[hash(x)for x in v],hash=hash(v))for v in tuples]
 sets={}
 for n in ['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_BASIC']:
  elements=[ast.literal_eval(v)for v in assignments[n].elts];folded=frozenset(elements);merged=frozenset(tuple(folded));constant=next(v for v in code.co_consts if isinstance(v,frozenset)and v==merged)
  assert list(constant)==list(merged),'source compiler folding hypothesis failed'
  assert list(ns[n])==list(set(constant))
  sets[n]=dict(elements=elements,folded=list(folded),constant=list(constant),runtime=list(ns[n]),capacity=8 if ns[n].__sizeof__()==set().__sizeof__()else(ns[n].__sizeof__()-set().__sizeof__())//16)
 cached_code=marshal.loads(marshal.dumps(code));cached_ns={};exec(cached_code,cached_ns)
 cached_sets={n:list(cached_ns[n])for n in ['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_BASIC']}
 configs=[]
 for res,pattern,middle,tuples_config in itertools.product([[],[1],[1,2],[2,1,1],[0,1]],['zero','one','mixed'],[-1,0,1,2],[False,True]):
  nt=sum(res);no=sum(v+1 for v in res);depth=[0 if pattern=='zero'else 1 if pattern=='one'else i%3 for i in range(nt)];out=[0 if pattern=='zero'else 1 if pattern=='one'else i%3 for i in range(no)]
  config=dict(num_res_blocks=tuple(res)if tuples_config else res,channel_mult=tuple(range(len(res)))if tuples_config else list(range(len(res))),transformer_depth=depth,transformer_depth_output=out,transformer_depth_middle=middle)
  configs.append(('ordinary',config))
 valid=dict(num_res_blocks=[1],channel_mult=[1],transformer_depth=[1],transformer_depth_output=[1,0],transformer_depth_middle=1)
 configs.extend([('empty',{}),('absent-first',{'unrelated':'own'})])
 for key in valid:
  for value in ['missing',None,[],(),0,False]:
   c=copy.deepcopy(valid)
   if value=='missing':del c[key]
   else:c[key]=value
   configs.append((key+'-'+str(value),c))
 cases=[]
 for label,initial in configs:
  config=copy.deepcopy(initial)
  try:output=ns['unet_to_diffusers'](config);outcome=dict(output=enc(output))
  except Exception as e:
   assert not isinstance(e,(MemoryError,RecursionError,NameError)),str(e)
   outcome=dict(error=type(e).__name__,message=str(e))
  cases.append(dict(label=label,initial=enc(initial),post=enc(config),**outcome))
 conversions=[];mapping=ns['unet_to_diffusers'](valid)
 for mode in ['empty','all','one-alias','both-aliases','shared-values','warning-error','warning-replace-value','warning-delete-key','warning-add-key','warning-clear-map','warning-clear-state','warning-replace-root','map-is-state','out-is-map','out-is-state']:
  name_map=mapping.copy();state={k:dict(own=i)for i,k in enumerate(mapping)if mode=='all' or mode=='shared-values' or mode=='out-is-state'}
  if mode=='one-alias':state={'class_embedding.linear_1.weight':dict(own=101)}
  if mode in ['both-aliases','out-is-map']:state={'class_embedding.linear_1.weight':dict(own=101),'add_embedding.linear_1.weight':dict(own=202)}
  if mode=='shared-values':
   own=dict(own=303)
   for k in state:state[k]=own
  output={};warnings=[];calls=[0];identities={}
  if mode=='map-is-state':state=name_map
  if mode=='out-is-map':output=name_map
  if mode=='out-is-state':output=state
  for label,obj in [('map',name_map),('state',state),('out',output)]:
   for k,v in obj.items():
    if isinstance(v,dict):identities.setdefault(id(v),label+':'+k)
  def snapshot():
   return dict(map=enc(name_map),state=enc(state),out=enc(output),root_aliases=[name_map is state,name_map is output,state is output],links={name:[[k,identities.get(id(v))]for k,v in obj.items()if isinstance(v,dict)]for name,obj in [('map',name_map),('state',state),('out',output)]})
  before=snapshot()
  class Logging:
   @staticmethod
   def warning(message):
    warnings.append(message);calls[0]+=1
    if calls[0]!=1:return
    if mode=='warning-error':raise ValueError('own warning failure')
    if mode=='warning-replace-value':name_map[next(iter(name_map))]='own.replaced'
    if mode=='warning-delete-key':name_map.pop(next(reversed(name_map)))
    if mode=='warning-add-key':name_map['own.added']='own.target'
    if mode=='warning-clear-map':name_map.clear()
    if mode=='warning-clear-state':state.clear()
    if mode=='warning-replace-root':scope['diffusers_keys']={}
  scope=dict(diffusers_keys=name_map,sd=state,new_sd=output,logging=Logging)
  try:convert_original_loop(**scope);outcome=dict(ok=True)
  except Exception as e:
   assert not isinstance(e,(MemoryError,RecursionError,NameError)),str(e)
   outcome=dict(error=type(e).__name__,message=str(e))
  conversions.append(dict(mode=mode,before=before,after=snapshot(),warnings=warnings,external_map_same=scope['diffusers_keys']is name_map,**outcome))
 cached_config=copy.deepcopy(valid);cached_map=cached_ns['unet_to_diffusers'](cached_config);cached_sd={'class_embedding.linear_1.weight':101,'add_embedding.linear_1.weight':202};cached_out={};cached_warnings=[]
 class CachedLogging:
  @staticmethod
  def warning(message):cached_warnings.append(message)
 convert_original_loop(cached_map,cached_sd,cached_out,CachedLogging)
 cached=dict(sets=cached_sets,config=enc(valid),map=enc(cached_map),remaining=enc(cached_sd),selected=enc(cached_out),warnings=cached_warnings)
 pointer_hash=ctypes.pythonapi._Py_HashPointer;pointer_hash.argtypes=[ctypes.c_void_p];pointer_hash.restype=ctypes.c_ssize_t
 pointers=[[v,pointer_hash(ctypes.c_void_p(v))]for v in [0,1,15,16,255,256,2**63-1,2**63,2**64-1,2**64-2]]
 assert 'torch'not in sys.modules
 print(json.dumps(dict(seed=os.environ['PYTHONHASHSEED'],secret_hex=secret.hex(),strings=string_hashes,bytes=byte_hashes,tuples=tuple_hashes,pointers=pointers,sets=sets,cases=cases,conversions=conversions,cached=cached),separators=(',',':')))
else:
 assert a.output is not None
 runs=[]
 for seed in ['0','1','6','42','4294967295','random']:
  env=os.environ.copy();env['PYTHONHASHSEED']=seed
  child=subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--comfy',str(a.comfy),'--child'],env=env,text=True,capture_output=True,check=True);runs.append(json.loads(child.stdout))
 result=dict(schema=1,sources=sources,python_version=list(sys.version_info[:3]),hash_algorithm='siphash13',hash_cutoff=0,trained_model_weights=0,torch_python_imported=False,fixture_tensors=0,loaded_full_model=False,cached_constants='own in-memory marshal roundtrip; no ordinary Python cache read',conversion_loop_binding='original loop in own function wrapper; source-local parameter binding, not module globals',hash_state_kind='own isolated interpreter state only; no private/service runtime data',runs=runs)
 a.output.write_text(json.dumps(result,separators=(',',':'))+'\n');print('Captured',len(runs),'source hash states,',sum(len(r['cases'])for r in runs),'name-map cases and',sum(len(r['conversions'])for r in runs),'conversion histories; no Torch/tensors/weights',flush=True)
