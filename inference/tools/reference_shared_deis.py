#!/usr/bin/env python3
"""Bounded original DEIS intermediate/gradient fixtures; no model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
cases=[]
bits=(numpy.arange(65536,dtype=numpy.uint32)*numpy.uint32(2654435761)+numpy.uint32(12345))
values=bits.view(numpy.float32)
with numpy.errstate(all='ignore'):logs=numpy.log(values)
log_cases=[dict(count=len(values),output_sha256=hashlib.sha256(logs.tobytes()).hexdigest())]
wide=(numpy.arange(65536,dtype=numpy.uint64)*numpy.uint64(6364136223846793005)+numpy.uint64(1442695040888963407))
with numpy.errstate(all='ignore'):log64=numpy.log(wide.view(numpy.float64));exp64=numpy.exp(wide.view(numpy.float64))
log64_cases=[dict(count=len(wide),log_sha256=hashlib.sha256(log64.tobytes()).hexdigest(),exp_sha256=hashlib.sha256(exp64.tobytes()).hexdigest())]

with torch.no_grad():
 for inference in [False,True]:
  for dtype in [torch.float32,torch.float64]:
   with torch.inference_mode(inference):
    sigmas=torch.tensor([.9,.6,.3,.1,0.05],dtype=dtype)
    time,beta0,beta1=ns['edm2t'](sigmas)
    taus=torch.linspace(time[1],time[2],10000)
    integrand=ns['cal_intergrand'](beta0,beta1,taus)
    coefficients=ns['get_deis_coeff_list'](sigmas,3)
    cases.append(dict(inference=inference,kind=str(dtype),time=stamp(time),time_values=time.tolist(),beta0=stamp(beta0),beta0_value=beta0.item(),beta1=stamp(beta1),beta1_value=beta1.item(),taus=stamp(taus),integrand=stamp(integrand),coefficients=[[dict(output=stamp(c),value=c.item())for c in row]for row in coefficients]))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,log_cases=log_cases,log64_cases=log64_cases,numpy_dispatch=numpy.lib.introspect.opt_func_info(func_name="^(log|exp)$",signature="float64")),separators=(',',':'))+'\n')
print('Captured bounded DEIS time, beta, 10000-point integrand and coefficient boundaries')
