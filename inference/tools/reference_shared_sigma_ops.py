#!/usr/bin/env python3
"""Built-in sigma transformations, views and exact parser behavior."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
import re
ns.update(re=re,io=NS(NodeOutput=lambda *args:args))
path='comfy_extras/nodes_custom_sampler.py'
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
source=a.comfy/path
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
sources[path]=expected
tree=ast.parse(source.read_text())
names=['SDTurboScheduler','SplitSigmas','SplitSigmasDenoise','FlipSigmas','SetFirstSigma','ExtendIntermediateSigmas','ManualSigmas']
functions={}
for cls in tree.body:
    if isinstance(cls,ast.ClassDef)and cls.name in names:
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
        fn.decorator_list=[]
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        functions[cls.name]=ns['execute']
cases=[]
with torch.no_grad():
    for dtype in [torch.float32,torch.float64,torch.bfloat16]:
        torch.set_default_dtype(dtype)
        model=NS(get_model_object=lambda name:ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))))
        for device in ['cpu','cuda:0']:
            for vals in [[],[0.],[1.,.6,.2,0.],[1.,.75,.5,.25,.01]]:
                for name,args in [
                    *[('SplitSigmas',[n])for n in [-3,-1,0,2,40]],
                    *[('SplitSigmasDenoise',[n])for n in [0.,.125,.5,.75,1.]],
                    ('FlipSigmas',[]),('SetFirstSigma',[.875]),
                    *[('ExtendIntermediateSigmas',[n,-1.,.2,spacing])for n in [0,1,3]for spacing in ['linear','cosine','sine']],
                ]:
                    tensor=torch.tensor(vals,device=device)
                    result=dict(name=name,kind=str(dtype),device=device,values=vals,args=args)
                    try:
                        out=functions[name](None,tensor,*args)
                        result['output']=[stamp(t)for t in out]
                        result['aliases']=[t.untyped_storage()._cdata==tensor.untyped_storage()._cdata for t in out]
                    except Exception as e:
                        reject_resource_failure(e);result.update(error=type(e).__name__,message=str(e))
                    result['input_after']=stamp(tensor);cases.append(result)
        for steps in [0,1,3,10,20]:
            for denoise in [0.,.25,.75,1.]:
                out=functions['SDTurboScheduler'](None,model,steps,denoise)
                cases.append(dict(name='SDTurboScheduler',kind=str(dtype),steps=steps,denoise=denoise,output=[stamp(t)for t in out]))
        for text in ['1, 0.5','1e-3 .25 +4.75','no numbers here','1..2','12...','.5.6','١.٢ -３.４','1. 2.','-0.0, +0.0','１２３𝟠.𝟡']:
            result=dict(name='ManualSigmas',kind=str(dtype),text=text)
            try:result['output']=[stamp(t)for t in functions['ManualSigmas'](None,text)]
            except Exception as e:
                reject_resource_failure(e);result.update(error=type(e).__name__,message=str(e))
            cases.append(result)
torch.set_default_dtype(torch.float32)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} sigma edit/parser cases; model weights allocated: 0')
