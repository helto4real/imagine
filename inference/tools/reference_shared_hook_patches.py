#!/usr/bin/env python3
"""Original hook-patch sharing, strength combination and strict cache budget."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_hooks.py')
exec(compile(loader.read_text().split('\ncases=[]\n',1)[0],str(loader),'exec'))
from reference_layout_completion import encoder
encode=encoder(torch,lambda t:t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
from reference_checks import reject_resource_failure
def outcome(fn):
 try:return dict(output=encode(fn()))
 except Exception as e:reject_resource_failure(e);return dict(error=type(e).__name__,message=str(e))
definitions('comfy/model_patcher.py',['create_hook_patches_clone','MemoryCounter'])
source=a.comfy/'comfy/model_patcher.py'
body=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ModelPatcher')
fn=next(n for n in body.body if isinstance(n,ast.FunctionDef)and n.name=='get_combined_hook_patches')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
def owned(v):return isinstance(v,(dict,list,tuple,torch.Tensor))
def flatten(v,path='$'):
 out=[]
 if owned(v):out.append((path,v))
 if isinstance(v,dict):
  for i,(key,item)in enumerate(v.items()):out.extend(flatten(item,f'{path}/{i}'))
 elif isinstance(v,(list,tuple)):
  for i,item in enumerate(v):out.extend(flatten(item,f'{path}/{i}'))
 return out
def identities(v):
 seen=[];out=[]
 for path,value in flatten(v):
  match=next((name for name,old in seen if value is old),None)
  out.append([path,match]);seen.append((path,value))
 return out
def alias_matrix(v,old):return [[value is other for _,other in flatten(old)]for _,value in flatten(v)]
clones=[];combined=[];budgets=[]
for form in ['ordinary','repeat','list_patch','tuple_list','empty','empty_patch','none_list','none_patch','dict_list','string_patch']:
 for copy_tuples in [False,True]:
  child={'own':[1,2]};p=(.5,child,1.,None,None);q=[1.,child,.75,None,None]
  values=[p,q]if form in ['ordinary','list_patch']else[p,p]if form=='repeat'else(p,q)if form=='tuple_list'else[]if form=='empty'else[()]if form=='empty_patch'else None if form=='none_list'else[None]if form=='none_patch'else{}if form=='dict_list'else['ab']
  if form=='list_patch':values=[q,q]
  initial={'A':{'weight':values},'B':{'other':[p]}}
  original=encode(initial);links=identities(initial);result=None
  def run():
   global result
   result=ns['create_hook_patches_clone'](initial,copy_tuples);return result
  captured=outcome(run)
  clones.append(dict(form=form,copy_tuples=copy_tuples,initial=original,input_links=links,post=encode(initial),aliases=[]if result is None else alias_matrix(result,initial),output_links=[]if result is None else identities(result),**captured))
for device in ['cpu','cuda:0']:
 for pattern in ['ordinary','repeat','empty','missing','list_patch','empty_patch','none_patch','bad_strength','tensor_strength']:
  for strength in [1.,1.+1e-10,1.+2e-9,0.,-.5,float('nan'),float('inf')]:
   for clip in [False,True]:
    child={'own':[1,2]};p=(.5,child,1.,None,None);q=(1.,child,.75,None,None)
    if pattern=='list_patch':p=list(p)
    elif pattern=='empty_patch':p=()
    elif pattern=='none_patch':p=None
    elif pattern=='bad_strength':p=('own',child,1.,None,None)
    elif pattern=='tensor_strength':p=(torch.tensor(.5,device=device),child,1.,None,None)
    initial={'A':{'w':[p,p]if pattern=='repeat'else[p],'x':[q]},'B':{'w':[q],'z':[]}}
    if pattern=='empty':initial={}
    group=ns['HookGroup']();hooks=[]
    for index in ['A','B','C']:
     h=ns['WeightHook']();h.hook_ref=index;h._strength_model=.7;h._strength_clip=-.25
     h.hook_keyframe.add(ns['HookKeyframe'](strength,0.,0))
     hooks.append(h);group.add(h)
    if pattern=='missing':group=ns['HookGroup']();group.add(hooks[2])
    original=encode(initial);links=identities(initial);result=None
    def run():
     global result
     result=ns['get_combined_hook_patches'](NS(hook_patches=initial),group);return result
    captured=outcome(run)
    combined.append(dict(pattern=pattern,device=device,strength=encode(strength),clip=clip,initial=original,input_links=links,post=encode(initial),aliases=[]if result is None else alias_matrix(result,initial),output_links=[]if result is None else identities(result),**captured))
for initial in [0,1,24,25,64,2**63+25]:
 for minimum in [0,1,24,64]:
  counter=ns['MemoryCounter'](initial,minimum);events=[]
  for count in [0,1,6,1,0,16]:
   w=torch.arange(count,dtype=torch.float32);before=counter.value;usable=counter.is_useable(count*4);used=counter.use(w)
   events.append(dict(bytes=count*4,before=before,usable=usable,used=used,after=counter.value))
  counter.decrement(-7);budgets.append(dict(initial=initial,minimum=minimum,events=events,after=counter.value))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,clones=clones,combined=combined,budgets=budgets),separators=(',',':'))+'\n')
print('Captured',len(clones),'clones,',len(combined),'combinations and',len(budgets),'strict budget histories; trained weights=0')
