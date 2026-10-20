#!/usr/bin/env python3
"""Original nested DPM call binding over own tokens; no Torch or weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='ksampler')
def run(name,mode,shape,terminal,style,fail=None):
 events=[];output=None;error=None
 def emit(*args):
  i=len(events);events.append(list(args))
  if i==fail:raise ValueError('own DPM call failure')
 class Sigma:
  def __init__(self,tag):self.tag=tag
  def __eq__(self,other):
   emit('equals-zero',self.tag,other)
   if shape=='ambiguous':raise RuntimeError('own ambiguous truth')
   return terminal and self.tag=='last'
 class Sigmas:
  def __len__(self):
   emit('length')
   if shape=='scalar':raise TypeError('own scalar length')
   return dict(empty=0,one=1,normal=4,ambiguous=4)[shape]
  def __getitem__(self,i):emit('index',i);return Sigma({-1:'last',-2:'penultimate',0:'first'}[i])
 def encode(value):
  if isinstance(value,Sigma):return value.tag
  if isinstance(value,Sigmas):return 'sigmas'
  return value
 class Module:
  def __getattr__(self,key):
   emit('module',key)
   def solve(*args,**kwargs):emit('solve',[encode(v)for v in args],[[k,encode(v)]for k,v in kwargs.items()]);return 'result'
   return solve
 space=dict(k_diffusion_sampling=Module(),KSAMPLER=lambda function,*_:function)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),space)
 function=space['ksampler'](name);args=['model','noise',Sigmas(),'extra','callback','disable'];kwargs={}
 if style=='keywords':kwargs=dict(zip(['model','noise','sigmas','extra_args','callback','disable'],args));args=[]
 if style=='options':kwargs=dict(eta=.5,s_noise=1.25,return_info=True,noise_sampler=None,own=7)
 if style=='duplicate':kwargs=dict(model='another')
 if style=='unexpected':kwargs=dict(own=7)
 if style.startswith('arity:'):args=args[:int(style.split(':')[1])]+['overflow']*max(0,int(style.split(':')[1])-6)
 try:output=encode(function(*args,**kwargs))
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 return dict(name=name,mode=mode,shape=shape,terminal=terminal,style=style,failure=fail,events=events,output=output,error=error)
cases=[]
for args in itertools.product(['dpm_fast','dpm_adaptive'],['plain'],['empty','one','normal','scalar','ambiguous'],[False,True],['positional','keywords','options','duplicate','unexpected']):cases.append(run(*args))
for name in ['dpm_fast','dpm_adaptive']:
 for count in range(9):cases.append(run(name,'plain','normal',True,'arity:'+str(count)))
 for style in ['positional','options']:
  args=(name,'plain','normal',True,style);case=run(*args);cases.extend(run(*args,fail=i)for i in range(len(case['events'])))
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original DPM call binding histories; no Torch or tensors')
