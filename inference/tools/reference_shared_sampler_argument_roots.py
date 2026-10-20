#!/usr/bin/env python3
"""Original default registered calls with temporary inputs and weak storage probes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import warnings
source=a.site/'torch/utils/_contextlib.py';sources['torch/utils/_contextlib.py']=hashlib.sha256(source.read_bytes()).hexdigest()
assert sources['torch/utils/_contextlib.py']==next(e['sha256']for e in manifest if e['root']=='site'and e['path']=='torch/utils/_contextlib.py')
unipc_path=a.comfy/'comfy/extra_samplers/uni_pc.py';unipc_tree=ast.parse(unipc_path.read_text())
unipc_fn=next(n for n in unipc_tree.body if isinstance(n,ast.FunctionDef)and n.name=='sample_unipc')
scale_line=next(n.lineno for n in unipc_fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='model_type'for t in n.targets))
function=ns['sample_unipc'];unipc_code=function.__code__
inventory=json.loads((project/'design/stage-3-source-inventory.json').read_text())
names=[name for name in inventory['samplers']if name!='ddim']+['_cfgpp_history']
def capture(name,device,style):
 events=[];probes={};failure={};messages=[]
 holder=[data((2,1,2,3),device,torch.float32,3)];probes['initial']=holder[0].untyped_storage()._weak_ref()
 def observe(op):events.append(dict(op=op,initial=not torch.UntypedStorage._expired(probes['initial']),scaled=None if 'scaled'not in probes else not torch.UntypedStorage._expired(probes['scaled'])))
 class Recording(Model):
  def __call__(self,*args,**kwargs):observe('model');return super().__call__(*args,**kwargs)
 sampling=ns['EPS']();sampling.sigma_data=1.;sampling.noise_scale=.875
 sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
 model=Recording(sampling,[],False);sigmas=torch.tensor([.9,.6,.3,.1,0.],device=device)
 def callback(values):observe('callback')
 def trace(frame,event,value):
  if frame.f_code is unipc_code and event=='line'and frame.f_lineno==scale_line:
   probes['scaled']=frame.f_locals['noise'].untyped_storage()._weak_ref();observe('scaled')
  return trace
 fn=ns['_sample_cfgpp_history']if name=='_cfgpp_history'else ns['sample_'+name]
 extra=dict(seed=901);torch.manual_seed(17);previous=sys.gettrace();sys.settrace(trace)
 try:
  with warnings.catch_warnings(record=True)as recorded:
   warnings.simplefilter('always')
   if name in ['dpm_fast','dpm_adaptive']:
    if style=='positional'and name=='dpm_fast':output=fn(model,holder.pop(),sigmas[-2],sigmas[0],len(sigmas)-1,extra_args=extra,callback=callback,disable=True)
    elif style=='positional':output=fn(model,holder.pop(),sigmas[-2],sigmas[0],extra_args=extra,callback=callback,disable=True)
    elif name=='dpm_fast':output=fn(model=model,x=holder.pop(),sigma_min=sigmas[-2],sigma_max=sigmas[0],n=len(sigmas)-1,extra_args=extra,callback=callback,disable=True)
    else:output=fn(model=model,x=holder.pop(),sigma_min=sigmas[-2],sigma_max=sigmas[0],extra_args=extra,callback=callback,disable=True)
   elif name in ['uni_pc','uni_pc_bh2']and style=='keywords':output=fn(model=model,noise=holder.pop(),sigmas=sigmas,extra_args=extra,callback=callback,disable=True)
   elif style=='keywords':output=fn(model=model,x=holder.pop(),sigmas=sigmas,extra_args=extra,callback=callback,disable=True)
   else:output=fn(model,holder.pop(),sigmas,extra_args=extra,callback=callback,disable=True)
   messages=[str(item.message)for item in recorded]
 except Exception as error:
  if isinstance(error,OracleBudgetError):raise
  reject_resource_failure(error);failure=dict(error=type(error).__name__,message=str(error))
 finally:sys.settrace(previous)
 observe('returned')
 result=dict(name=name,device=device,style=style,events=events,warnings=messages,post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(output))))
 if not failure:output=None
 observe('output-dropped')
 result['state']=stamp(torch.get_rng_state()if device=='cpu'else torch.cuda.get_rng_state(device))
 for probe in probes.values():torch.UntypedStorage._free_weak_ref(probe)
 return result
cases=[]
with torch.no_grad():
 for name in names:
  for device in ['cpu','cuda:0']:
   for style in ['positional','keywords']:cases.append(capture(name,device,style))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_latent_elements=12,scope='44 registered raw default kernels plus undecorated CFG helper; DDIM whole-KSAMPLER evidence remains separate',cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original temporary-input argument-root/default-call histories; no weights')
