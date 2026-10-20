#!/usr/bin/env python3
"""AYS/GITS and exact NumPy interpolation, CPU-only and no models."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
ns.update(np=numpy,io=NS(NodeOutput=lambda *values:values))
cases=[];interpolations=[]
with torch.no_grad():
 for label,path,cls in [('ays','comfy_extras/nodes_align_your_steps.py','AlignYourStepsScheduler'),('gits','comfy_extras/nodes_gits.py','GITSScheduler')]:
  definitions(path,['loglinear_interp'])
  tree=ast.parse((a.comfy/path).read_text())
  table=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='NOISE_LEVELS'for t in n.targets))
  ns['NOISE_LEVELS']=table
  clsnode=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
  fn=next(n for n in clsnode.body if isinstance(n,ast.FunctionDef)and n.name=='execute');fn.decorator_list=[]
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.comfy/path),'exec'),ns)
  keys=list(table)if label=='ays'else[.8,1.,1.2,1.25,1.5,.805,2.]
  for key in keys:
   for steps in [-1,0,1,2,3,7,10,20,21,40,100]:
    for denoise in [0.,.125,.5,1.]:
     c=dict(name=label,key=key,steps=steps,denoise=denoise)
     try:c['output']=stamp(ns['execute'](None,key,steps,denoise)[0])
     except Exception as e:
      reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
     cases.append(c)
  for values in [[],[1.],[10.,3.,.5,.01],[4.,1.,0.],[-1.,1.,.1]]:
   for length in [-1,0,1,2,3,7,40]:
    c=dict(name=label,values=values,length=length)
    try:
     with numpy.errstate(all='ignore'):out=ns['loglinear_interp'](numpy.array(values),length)
     c['output']=stamp(torch.from_numpy(out));c['bits']=[int(x)for x in out.view(numpy.uint64)]
    except Exception as e:
     reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
    interpolations.append(c)
# Preserve signed/payload NaNs and infinities as raw initialized input bits.
  for raw in [0x7ff8000000000000,0xfff8000000000000,0x7ff0000000001234,0xfff0000000001234,0x7ff0000000000000,0xfff0000000000000,0x8000000000000000,0]:
   for values_bits in [[raw],[0x4010000000000000,raw,0x3ff0000000000000]]:
    for length in [0,1,2,3,7]:
     c=dict(name=label,values_bits=values_bits,length=length)
     values=numpy.array(values_bits,dtype=numpy.uint64).view(numpy.float64)
     with numpy.errstate(all='ignore'):out=ns['loglinear_interp'](values,length)
     c['output']=stamp(torch.from_numpy(out));c['bits']=[int(x)for x in out.view(numpy.uint64)];interpolations.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,interpolations=interpolations),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} AYS/GITS and {len(interpolations)} exact Float64 interpolation cases; GPU/model usage: 0')
