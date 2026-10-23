#!/usr/bin/env python3
"""Own CPU-only processes: original default-global identity-set iteration."""
import argparse,hashlib,io,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--child',type=int);a=p.parse_args()
if a.child is not None:
 padding=[dict(index=n)for n in range(a.child)]
 sys.path.insert(0,str(a.site))
 import torch
 import torch._weights_only_unpickler as original
 assert os.environ['PYTHONHASHSEED']=='0';assert sys.version_info[:3]==(3,12,13)
 # All values belong to this newly imported, isolated reference framework.
 # These are type/function descriptors, not weights or existing runtime data.
 new=torch.storage._new_dtypes();tensors=torch._tensor_classes;storages=torch._storage_classes;rebuild=original._tensor_rebuild_functions()
 def path(value):return value.__module__+'.'+value.__name__
 order=dict(new_dtypes=list(map(str,new)),tensor_classes=[path(v)for v in tensors],storage_classes=[path(v)for v in storages],rebuild_functions=[path(v)for v in rebuild],user_safe_globals=[path(v[0])+' as '+v[1]if isinstance(v,tuple)else path(v)for v in original._marked_safe_globals_set])
 allowed=original._get_allowed_globals();user=original._get_user_allowed_globals();labels={}
 for name,value in allowed.items():labels.setdefault(id(value),name)
 for name,value in user.items():labels.setdefault(id(value),'user:'+name)
 events=[]
 class Probe:
  __hash__=object.__hash__
  def __eq__(self,other):events.append(labels[id(other)]);return False
  def __format__(self,spec):return '<own probe>'
 probe=Probe();stream=io.BytesIO(b'h\x00h\x01R.');vm=original.Unpickler(stream);vm.memo={0:probe,1:()}
 try:vm.load();raise AssertionError('probe must remain unauthorized')
 except original.UnpicklingError as e:failure=dict(type=type(e).__name__,message=str(e))
 def pointer_hash(value):
  address=id(value);h=((address>>4)|(address<<60))&((1<<64)-1);h=h-(1<<64)if h>>63 else h
  return -2 if h==-1 else h
 hash_contract={key:all(hash(v)==pointer_hash(v)for v in values)for key,values in [('new_dtypes',new),('tensor_classes',tensors),('storage_classes',storages),('rebuild_functions',rebuild)]}
 assert all(hash_contract.values())
 def own_a():return 1
 function_padding=[lambda index=index:index for index in range(a.child)]
 def own_b():return 2
 with original._safe_globals([(own_a,'own.same'),(own_b,'own.same')]):
  selected=original._get_user_allowed_globals()['own.same']
  output=original.load(io.BytesIO(b'cown\nsame\n)R.'))
  alias=dict(selected='a'if selected is own_a else'b',output=output)
  assert output==(1 if selected is own_a else 2)
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(padding=a.child,hashseed=0,sets=order,default_keys=list(allowed),user_keys=list(user),comparison_order=events,error=failure,generic_pointer_hash=hash_contract,user_alias=alias,trained_model_weights=0,cuda_work=False),separators=(',',':')))
 raise SystemExit(0)
if a.output is None:p.error('--output required in parent')
cases=[]
for padding in [0,37,512]:
 for repeat in range(3):
  env=dict(os.environ,PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='')
  proc=subprocess.run([sys.executable,'-S','-B',__file__,'--site',str(a.site),'--child',str(padding)],env=env,check=True,text=True,capture_output=True)
  case=json.loads(proc.stdout);case['repeat']=repeat;cases.append(case)
summary=[]
for padding in [0,37,512]:
 group=[c for c in cases if c['padding']==padding]
 summary.append(dict(padding=padding,distinct={key:len({tuple(c['sets'][key])for c in group})for key in group[0]['sets']},default_orders=len({tuple(c['default_keys'])for c in group}),comparison_orders=len({tuple(c['comparison_order'])for c in group}),user_alias_outputs=sorted({c['user_alias']['output']for c in group})))
sources=[a.site/'torch/_weights_only_unpickler.py',a.site/'torch/storage.py']
a.output.write_text(json.dumps(dict(schema=1,scope='source nondeterminism diagnostic; not a parity acceptance or changed criterion',trained_model_weights=0,cuda_work=False,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources},summary=summary,cases=cases),indent=2)+'\n')
print(json.dumps(summary))
