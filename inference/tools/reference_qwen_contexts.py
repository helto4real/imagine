#!/usr/bin/env python3
"""Original inference/no-grad context nesting and sampler decorator behavior."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_er_sde.py')
exec(compile(loader.read_text().split('\ncases=[]',1)[0],str(loader),'exec'))
base=torch.ones([1,1,1,2],requires_grad=True)
contexts=[]
def snapshot(label):
 output=base*2
 contexts.append(dict(label=label,requires_grad=output.requires_grad,inference=output.is_inference()))
snapshot('normal')
with torch.inference_mode():
 snapshot('on')
 with torch.inference_mode(False):snapshot('disabled')
 snapshot('on-restored')
 try:
  with torch.inference_mode(False):
   snapshot('disabled-before-error');raise ValueError('own bounded scope exit')
 except ValueError:pass
 snapshot('on-after-error')
snapshot('normal-restored')
with torch.no_grad():
 snapshot('no-grad')
 with torch.inference_mode():
  snapshot('nested-on')
  with torch.inference_mode(False):snapshot('nested-off')
  snapshot('nested-restored')
 snapshot('no-grad-restored')
snapshot('normal-final')
samplers=[]
for method in ['sample_euler','sample_er_sde']:
 for fail in [False,True]:
  calls=[]
  def model(x,sigma,**kwargs):return x*.25
  model.inner_model=NS(model_patcher=NS(get_model_object=lambda name:Sampling(NS(sampling_settings=dict(shift=.69)))))
  def callback(values):
   calls.append(dict(requires_grad=values['denoised'].requires_grad,inference=values['denoised'].is_inference()))
   if fail:raise ValueError('injected observer exit')
  kwargs=dict(extra_args=dict(seed=17),callback=callback,disable=True)
  if method=='sample_er_sde':kwargs['s_noise']=0.
  record=dict(method=method,fail=fail,calls=calls)
  try:
   output=namespace[method](model,base,torch.tensor([.8,.4,0.]),**kwargs)
   record['output']=dict(requires_grad=output.requires_grad,inference=output.is_inference())
  except Exception as error:record['error']=dict(class_name=type(error).__name__,message=str(error))
  record['restored_grad']=torch.is_grad_enabled();samplers.append(record)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,contexts=contexts,samplers=samplers,safety=dict(largest_tensor_elements=200,real_model_constructor=False)),separators=(',',':'))+'\n')
print(f'Captured {len(contexts)} context states and {len(samplers)} sampler context/failure histories')
