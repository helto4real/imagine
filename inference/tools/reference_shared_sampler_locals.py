#!/usr/bin/env python3
"""Own source sampler frames: retained tensor locals at model/callback sites."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
def snapshot(frame):
    tensors={}
    def own(name,value):
        if isinstance(value,torch.Tensor):tensors[name]=value
        elif isinstance(value,numpy.ndarray)and isinstance(value.base,torch.Tensor):tensors[name]=value.base
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value):own(f'{name}[{i}]',v)
    for name,value in frame.f_locals.items():own(name,value)
    groups={};output={}
    for name,tensor in sorted(tensors.items()):
        storage=tensor.untyped_storage()
        ident=storage._cdata
        if ident not in groups:groups[ident]=len(groups)
        output[name]=dict(value=stamp(tensor),storage_group=groups[ident],storage_bytes=storage.nbytes())
    return output
cases=[]
churn_names=['euler','heun','dpm_2','heunpp2']
old_names=churn_names+['dpmpp_2m','dpmpp_2m_cfg_pp','gradient_estimation','gradient_estimation_cfg_pp']
new_names=['euler_ancestral','dpm_2_ancestral','euler_ancestral_cfg_pp','euler_cfg_pp','lcm','ddpm','ipndm','ipndm_v','lms']
with torch.no_grad():
    for name in old_names+new_names:
        for device in ['cpu','cuda:0']:
            for dtype in [torch.float32,torch.bfloat16]:
                for churn in [0.,.3]if name in churn_names else[0.]:
                    for end in [0.]if name in churn_names else[0.,.05]:
                        for flow in [False,True]if name in new_names and name not in ['ddpm','ipndm','ipndm_v','lms']else[False]:
                            for eta in [0.,1.]if 'ancestral'in name else[0.]:
                                for advanced in [False,True]if name=='lcm'else([2,3,4]if name in ['ipndm','ipndm_v']else[0,1,2,4]if name=='lms'else[False]):
                                    events=[];call_source=[None]
                                    class Recording(Model):
                                        def __call__(self,*args,**kwargs):
                                            parent=sys._getframe(1);call_source[0]=parent.f_code
                                            events.append(dict(event='model',locals=snapshot(parent)))
                                            return super().__call__(*args,**kwargs)
                                    sampling=ns['CONST']()if flow else ns['EPS']();sampling.noise_scale=.875
                                    model=Recording(sampling,[],False)
                                    x=data((2,1,2,3),device,dtype,3)
                                    sigmas=torch.tensor([.9,.6,.3,.1,end],device=device)
                                    def callback(values):
                                        events.append(dict(event='callback',locals=snapshot(sys._getframe(1))))
                                    kwargs={}
                                    if name in churn_names:kwargs=dict(s_churn=churn,s_tmin=.2,s_tmax=.95,s_noise=.875)
                                    elif 'ancestral'in name:kwargs=dict(eta=eta,s_noise=.875)
                                    elif name=='lcm'and advanced:kwargs=dict(s_noise=.875,s_noise_end=.25,noise_clip_std=1.5)
                                    elif name in ['ipndm','ipndm_v']:kwargs=dict(max_order=advanced)
                                    elif name=='lms':kwargs=dict(order=advanced)
                                    torch.manual_seed(17)
                                    def trace(frame,event,value):
                                        if event=='return'and frame.f_code is call_source[0]:events.append(dict(event='return',locals=snapshot(frame)))
                                        return trace
                                    previous=sys.gettrace();sys.settrace(trace)
                                    try:y=ns['sample_'+name](model,x,sigmas,callback=callback,disable=True,**kwargs)
                                    finally:sys.settrace(previous)
                                    cases.append(dict(name=name,device=device,kind=str(dtype),churn=churn,end=end,flow=flow,eta=eta,advanced=advanced,events=events,output=stamp(y)))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} sampler tensor-local lifetimes; model weights allocated: 0')
