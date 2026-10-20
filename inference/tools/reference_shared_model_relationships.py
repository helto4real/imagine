#!/usr/bin/env python3
"""Original model relationships on own metadata owners; no model allocation."""
import __future__,argparse,ast,hashlib,inspect,itertools,json,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/model_patcher.py';digest=hashlib.sha256(path.read_bytes()).hexdigest();manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/model_patcher.py')
names=['set_attachments','remove_attachments','get_attachment','set_additional_models','remove_additional_models','get_additional_models_with_key','get_additional_models','get_nested_additional_models'];cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ModelPatcher');cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(cls.body)==8
ns={};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns);M=ns['ModelPatcher'];outer=M.get_nested_additional_models.__code__;inner=next(c for c in outer.co_consts if inspect.iscode(c)and c.co_name=='_evaluate_sub_additional_models')
graphs=[[[],[],[],[],[]],[[1],[2],[3],[],[]],[[1,2],[3],[3],[],[]],[[1],[2],[0],[],[]],[[1,1,2],[3,3],[3,0],[],[]],[[0,0,1],[],[],[],[]],[[],[2],[3],[],[]],[[0,1,2],[0,2,3],[0,1,3],[0],[]]]
def run(graph,layout,hash_kind,mode):
 events=[];active=[False];calls=[0];models=[]
 def enc(v):
  if isinstance(v,Node):return 'm'+str(v.index)
  if isinstance(v,(list,tuple)):return dict(type=type(v).__name__,items=[enc(x)for x in v])
  if isinstance(v,dict):return dict(type='dict',items=[[k,enc(x)]for k,x in v.items()])
  return v
 def ancestors():
  f=inspect.currentframe().f_back;frames=[];top=None
  while f:
   if f.f_code is inner:frames.append(f)
   if f.f_code is outer:top=f;break
   f=f.f_back
  return top,list(reversed(frames))
 def snapshot(top,frames):
  all_models=top.f_locals.get('all_models')if top else None;cache=top.f_locals.get('models_set')if top else None
  return dict(all=enc(all_models),cache=None if cache is None else[enc(v)for v in cache],frames=[dict(previous=enc(f.f_locals['prev_models']),next=enc(f.f_locals['next_models']),model=enc(f.f_locals.get('model')),candidates=enc(f.f_locals.get('candidates')),candidate=enc(f.f_locals.get('c')))for f in frames])
 class Node(M):
  def __init__(self,index):self.index=index;self.attachments={};self.additional_models={}
  def __hash__(self):
   if active[0]:events.append(dict(tag='hash',model=enc(self)))
   return 42 if hash_kind=='collision'else object.__hash__(self)
  def get_additional_models(self):
   if not active[0]:return M.get_additional_models(self)
   calls[0]+=1;top,frames=ancestors();events.append(dict(tag='models',model=enc(self),locals=snapshot(top,frames)))
   if mode=='error-root'and calls[0]==1 or mode=='error-child'and calls[0]==2:raise ValueError('own model failure')
   if calls[0]==2:
    if mode=='append-previous':frames[-1].f_locals['prev_models'].append(models[4])
    if mode=='clear-previous':frames[-1].f_locals['prev_models'].clear()
    if mode=='replace-root':models[0].additional_models={}
   out=M.get_additional_models(self)
   if mode=='append-returned'and calls[0]==2:out.append(models[4])
   return out
 models=[Node(i)for i in range(len(graph))]
 for model,targets in zip(models,graph):
  values=[models[i]for i in targets];model.additional_models={str(i):tuple(g)if layout=='tuple'else g for i,g in enumerate([values[:1],values[1:]])}
 if mode=='duplicate-groups':models[0].additional_models['again']=models[0].additional_models['0']
 hashes=[42 if hash_kind=='collision'else object.__hash__(m)for m in models];before=[enc(m.additional_models)for m in models];active[0]=True;finished={}
 def profile(frame,event,arg):
  if event=='return'and frame.f_code is outer and arg is not None:finished.update(all=enc(frame.f_locals.get('all_models')),cache=[enc(v)for v in frame.f_locals['models_set']],same_all=arg is frame.f_locals.get('all_models'))
 old=sys.getprofile();sys.setprofile(profile)
 try:out=models[0].get_nested_additional_models();outcome=dict(output=enc(out))
 except Exception as e:
  assert not isinstance(e,(MemoryError,RecursionError,NameError)),str(e)
  outcome=dict(error=type(e).__name__,message=str(e))
 finally:sys.setprofile(old);active[0]=False
 return dict(params=dict(graph=graph,layout=layout,hash=hash_kind,mode=mode),hashes=hashes,before=before,events=events,finished=finished,after=[enc(m.additional_models)for m in models],**outcome)
cases=[run(graph,layout,hash_kind,mode)for graph,layout,hash_kind,mode in itertools.product(graphs,['list','tuple'],['identity','collision'],['observe','error-root','error-child','append-previous','clear-previous','append-returned','replace-root','duplicate-groups'])]
for hash_kind in ['identity','collision']:cases.append(run([[i+1]if i<127 else[]for i in range(128)],'list',hash_kind,'observe'))
histories=[]
for layout,field in itertools.product(['list','tuple','dict','text','none'],['attachments','additional_models']):
 m=M();m.attachments={};m.additional_models={};held={};events=[]
 def enc(v):
  if isinstance(v,(tuple,list)):return dict(type=type(v).__name__,items=[enc(x)for x in v])
  if isinstance(v,dict):return dict(type='dict',items=[[k,enc(x)]for k,x in v.items()])
  return v
 def snap():
  roots=[('attachments',m.attachments),('additional_models',m.additional_models)]+list(held.items());values=roots+[(name+':'+k,v)for name,root in roots[:2]for k,v in root.items()];seen=[];links=[]
  for name,value in values:
   if not isinstance(value,(dict,list,tuple)):continue
   links.append([name,next((n for n,v in seen if v is value),None)]);seen.append((name,value))
  return dict(attachments=enc(m.attachments),additional_models=enc(m.additional_models),held=[[k,enc(v)]for k,v in held.items()],links=links)
 value=['a','b']if layout=='list'else('a','b')if layout=='tuple'else{'a':1,'b':2}if layout=='dict'else'ab'if layout=='text'else None
 actions=[dict(op='get',key='missing',hold='first'),dict(op='get',key='missing',hold='second'),dict(op='set',key='own'),dict(op='get',key='own',hold='value'),dict(op='flatten'),dict(op='hold-root',hold='old-root'),dict(op='replace-root'),dict(op='get',key='own',hold='absent'),dict(op='set',key='new'),dict(op='remove',key='missing'),dict(op='remove',key='new'),dict(op='remove',key='own')]
 before=snap()
 for action in actions:
  try:
   op=action['op'];key=action.get('key')
   if op=='get':held[action['hold']]=m.get_attachment(key)if field=='attachments'else m.get_additional_models_with_key(key)
   if op=='set':(m.set_attachments if field=='attachments'else m.set_additional_models)(key,value)
   if op=='remove':(m.remove_attachments if field=='attachments'else m.remove_additional_models)(key)
   if op=='flatten':held['flatten']=m.get_additional_models()
   if op=='hold-root':held[action['hold']]=getattr(m,field)
   if op=='replace-root':setattr(m,field,{})
   result=dict(ok=True)
  except Exception as e:result=dict(error=type(e).__name__,message=str(e))
  events.append(dict(action=action,state=snap(),**result))
 histories.append(dict(layout=layout,field=field,value=enc(value),before=before,events=events))
assert 'torch'not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,python_version=list(sys.version_info[:3]),trained_model_weights=0,torch_python_imported=False,fixture_tensors=0,constructor='own metadata scaffold; original ModelPatcher constructor/destructor not invoked',cases=cases,histories=histories),separators=(',',':'))+'\n');print('Captured',len(cases),'original dependency graphs and',len(histories),'relationship histories; no Torch, model constructor or weights',flush=True)
