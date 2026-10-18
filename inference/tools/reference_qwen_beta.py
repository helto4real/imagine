#!/usr/bin/env python3
"""Exact default/custom beta schedules and their original special-function inputs."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_sampling.py')
exec(compile(loader.read_text().split('\nschedules=[]',1)[0],str(loader),'exec'))
import numpy
import scipy
import scipy.stats
import warnings
for relative in ['scipy/version.py','scipy/special/_ufuncs_cxx.cpython-312-x86_64-linux-gnu.so','scipy/special/_ufuncs.cpython-312-x86_64-linux-gnu.so','scipy/stats/_continuous_distns.py','numpy/_core/function_base.py']:
 path=a.site/relative;expected=next(e['sha256']for e in manifest if e['root']=='site'and e['path']==relative)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
 sources[relative]=dict(path=str(path),sha256=expected)
namespace.update(numpy=numpy,scipy=scipy)
definitions('comfy/samplers.py',['beta_scheduler'])
schedules=[];quantiles=[]
with torch.no_grad():
 for shift in [.69,1.15]:
  sampling=namespace['ModelSamplingFlux'](NS(sampling_settings=dict(shift=shift)))
  for steps in [0,1,2,4,8,20,40,100,1000,10001,-1]:
   for alpha,beta in [(.6,.6),(.5,2.),(2.,.5),(1.,1.),(.01,10.),(10.,.01),(0.,.6),(-1.,.6),(float('nan'),.6)]:
    record=dict(shift=shift,steps=steps,alpha='nan'if numpy.isnan(alpha)else alpha,beta=beta)
    with warnings.catch_warnings(record=True)as caught:
     warnings.simplefilter('always')
     try:record['output']=stamp(namespace['beta_scheduler'](sampling,steps,alpha,beta))
     except Exception as error:record['error']=dict(class_name=type(error).__name__,message=str(error))
     record['warnings']=[dict(class_name=type(w.message).__name__,message=str(w.message))for w in caught]
    schedules.append(record)
 for a_value,b_value in [(.6,.6),(.5,2.),(2.,.5),(1.,1.),(.01,10.),(10.,.01),(1e-12,1e12),(1e12,1e-12)]:
  probabilities=1-numpy.linspace(0,1,1000,endpoint=False)
  with warnings.catch_warnings(record=True)as caught:
   warnings.simplefilter('always')
   output=scipy.special._ufuncs._beta_ppf(probabilities,a_value,b_value)
   quantiles.append(dict(alpha=a_value,beta=b_value,probabilities=probabilities.tolist(),output_sha256=hashlib.sha256(output.astype('<f8').tobytes()).hexdigest(),warnings=[str(w.message)for w in caught]))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,scipy_version=scipy.__version__,numpy_version=numpy.__version__,safety=dict(real_model_constructor=False,largest_schedule_elements=10002),schedules=schedules,quantiles=quantiles),separators=(',',':'))+'\n')
print(f'Captured {len(schedules)} beta schedule/error cases and {len(quantiles)*1000} original Boost quantiles')
