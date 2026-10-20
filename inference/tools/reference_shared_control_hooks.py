#!/usr/bin/env python3
"""Original ControlNet hook collection with captured identity hashes; no Torch."""
import __future__,argparse,ast,enum,hashlib,itertools,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)and sys.maxsize==2**63-1
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());sources={};ns=dict(enum=enum,Enum=enum.Enum,itertools=itertools,torch=NS(float32=object()))
for relative,names in [('comfy/hooks.py',['EnumHookType','EnumHookScope','_HookRef','default_should_register','Hook','HookGroup','HookKeyframe','HookKeyframeGroup']),('comfy/controlnet.py',['StrengthType','ControlBase']),('comfy/sampler_helpers.py',['get_hooks_from_cond'])]:
 path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']==relative),'Original source changed';sources[relative]=digest
 body=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef))and node.name in names]
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
ns['comfy']=NS(hooks=NS(HookGroup=ns['HookGroup']))
C=ns['ControlBase'];H=ns['Hook'];G=ns['HookGroup'];collect=ns['get_hooks_from_cond']

def run(count,mode='ordinary',layout='list',pattern='identity'):
 hooks=[H()for _ in range(count+2 if pattern=='identity'else 2)];groups=[];events=[];reads={};controls=[];recording=[False];seen=[]
 full=G();full.add(hooks[-1])
 for i in range(count if pattern=='identity'else 0):
  g=G()
  if mode!='empty-groups':g.add(hooks[i]);g.add(hooks[i].clone())
  if mode=='overlap':g.add(hooks[-2])
  groups.append(g)
 def hook_state():
  if pattern!='identity':return []
  return [dict(ref=next(i for i,h in enumerate(hooks)if h.hook_ref is item.hook_ref),same=[item is h for h in hooks],keyframes=[item.hook_keyframe is h.hook_keyframe for h in hooks])for item in full.hooks]
 def attribute(control,name,value):
  if not recording[0]:return value
  i=object.__getattribute__(control,'own_index');key=(i,name);reads[key]=reads.get(key,0)+1;n=reads[key]
  events.append([i,name,n])
  if i==0 and mode==('fail-extra'if name=='extra_hooks'else'fail-previous')and n==1:raise ValueError('own '+name+' failure')
  if i==0 and mode==('extra-second-none'if name=='extra_hooks'else'previous-second-none')and n==2:return None
  if i==0 and mode=='replace-list'and name=='extra_hooks'and n==1:conditions.clear()
  return value
 class Observed(C):
  def __getattribute__(self,name):
   value=super().__getattribute__(name)
   return attribute(self,name,value)if name in ['extra_hooks','previous_controlnet']else value
 # The source ControlBase path keeps its inherited object.__hash__/__eq__.
 # Separate table-stress objects supply controlled hashes but identity equality;
 # this validates probing only, not a changed ControlBase hashing policy.
 class Hashed:
  __slots__=('own_index','own_hash','extra_hooks','previous_controlnet')
  def __hash__(self):return self.own_hash
 cls=Hashed if pattern!='identity'else Observed
 for i in range(count):
  c=cls();c.own_index=i
  if pattern!='identity':c.own_hash=(i*37 if pattern=='spread'else i<<17 if pattern=='collision'else -2 if i%2 else -(i+1)*257)
  c.extra_hooks=None if mode=='no-hooks'else groups[i]
  c.previous_controlnet=None
  controls.append(c)
 if mode in ['chain','shared-chain','previous-second-none']:
  for i in range(count-1):controls[i].previous_controlnet=controls[i+1]
 if mode=='shared-chain'and count>2:controls[0].previous_controlnet=controls[-1]
 if mode=='full-alias'and count:controls[0].extra_hooks=full
 if mode=='shared-group'and count:
  for c in controls:c.extra_hooks=groups[0]
 base=list(range(count))
 if mode=='reverse':base.reverse()
 if mode=='duplicates'and count:base=base+[count-1,0]+base
 conditions=[dict(control=controls[i])for i in base]
 if pattern=='identity':
  conditions.insert(0,dict(hooks=groups[0]if count else G()))
  conditions.append(dict(hooks=full))
 if mode=='bad-hooks':conditions.insert(1,dict(hooks=None))
 if mode=='bad-control':conditions.append(dict(control=[]))
 if layout=='tuple':conditions=tuple(conditions)
 hashes=[hash(c)for c in controls];addresses=[id(c)for c in controls]
 expected_set=set(controls[i]for i in base);order=[c.own_index for c in expected_set];capacity=8+(expected_set.__sizeof__()-set().__sizeof__())//16
 # __sizeof__ includes smalltable even when the active table is separately allocated.
 if capacity!=8:capacity-=8
 before=hook_state();recording[0]=True
 try:result=collect(conditions,full);outcome=dict(return_same=result is full)
 except Exception as e:
  assert not isinstance(e,(MemoryError,RecursionError,NameError)),str(e)
  outcome=dict(error=type(e).__name__,message=str(e))
 recording[0]=False
 result=dict(params=dict(count=count,mode=mode,layout=layout,pattern=pattern),hashes=hashes,addresses=addresses,order=order,capacity=capacity,before=before,events=events,after=hook_state(),**outcome)
 if pattern!='identity':
  # Keep large table-stress fixtures bounded: they contain no per-hook tensors,
  # hook identity matrices or attribute event streams.
  result['addresses']=[];result['after']=[];result['before']=[]
 return result
cases=[]
for count,mode,layout in itertools.product([0,1,2,3,5,19,77],['ordinary','reverse','duplicates','no-hooks','empty-groups','chain','shared-chain','full-alias','shared-group','overlap'],['list','tuple']):cases.append(run(count,mode,layout))
for count,mode in itertools.product([1,2,4],['bad-hooks','bad-control','fail-extra','fail-previous','extra-second-none','previous-second-none','replace-list']):cases.append(run(count,mode))
for count in [4,5,18,19,76,77,306,307,1228,1229,4914,4915,19660,19661,78642,78643,78644]:
 for pattern in ['spread','collision','negative']if count<1000 else['spread']:
  # Do not create O(N^2) hook matrices for probing stress cases.
  cases.append(run(count,'no-hooks','list',pattern))
assert 'torch'not in sys.modules
result=dict(schema=1,sources=sources,python_version=list(sys.version_info[:3]),trained_model_weights=0,torch_python_imported=False,fixture_tensors=0,identity_criterion='D-05-CONTROL-HOOK-SET-ORDER; exact captured hashes and order',set_source='https://github.com/python/cpython/blob/v3.12.13/Objects/setobject.c',hash_source='https://github.com/python/cpython/blob/v3.12.13/Python/pyhash.c',cases=cases)
a.output.write_text(json.dumps(result,separators=(',',':'))+'\n');print('Captured',len(cases),'original control-hook cases; no tensors, Torch or trained weights',flush=True)
