#!/usr/bin/env python3
"""Original module sample entry point over own tokens and observable results."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
source=a.comfy/'comfy/samplers.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/samplers.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='sample')
def run(mode,fail=None):
 events=[];output=None;error=None;replaced=[False]
 def emit(*args):
  i=len(events);events.append(list(args))
  if i==fail:raise ValueError('own public sample failure')
 class Ignored:
  def __init__(self,name):self.name=name
  def __del__(self):events.append(['drop-result',self.name])
 class Guider:
  def __getattr__(self,name):
   emit('attribute',name)
   def call(*args):
    emit('call',name,list(args),replaced[0])
    if name=='set_conds'and mode=='replace':replaced[0]=True
    return 'output'if name=='sample'else Ignored(name)
   return call
 def constructor(model):emit('construct',model);return Guider()
 space=dict(CFGGuider=constructor);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),space)
 try:output=space['sample']('model','noise','positive','negative','cfg','unused-device','sampler','sigmas',model_options='unused-options',latent_image='latent',denoise_mask='mask',callback='callback',disable_pbar='disable',seed='seed')
 except Exception as ex:error=[type(ex).__name__,str(ex)]
 return dict(mode=mode,failure=fail,events=events,output=output,error=error)
cases=[]
for mode in ['plain','replace']:
 case=run(mode);cases.append(case)
 for index,event in enumerate(case['events']):
  if event[0]!='drop-result':cases.append(run(mode,index))
assert 'torch'not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original public sample histories; no Torch or tensors')
