#!/usr/bin/env python3
"""Original PerpNeg predict_noise branch/callback histories on own tensors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_conditioning.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
ns['comfy'].samplers=NS(CFGGuider=object)
definitions('comfy_extras/nodes_perpneg.py',['perp_neg','Guider_PerpNeg'])
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for dtype in [torch.float32,torch.bfloat16]:
   for cfg in [1.,1.+5e-10,4.]:
    for negative_scale in [-0.,.5]:
     for disabled in [False,True,None,0]:
      for mode in ['plain','pre','post','both','pre_append','pre_replace','post_append','post_replace','short','long','zeros']:
       events=[];x=data((1,2,3,4),device,dtype,3);time=torch.tensor([.75],device=device)
       model=object();opts=dict(disable_cfg1_optimization=disabled)
       def observe(event,index,args):
        events.append(dict(event=event,index=index,conds=args.get('conds'),cond=args.get('cond'),uncond=args.get('uncond'),empty=args.get('empty_cond'),scale=args['cond_scale'],model_same=args['model']is model,options_same=args['model_options']is opts,input=encode(args['input']),sigma=encode(args['sigma']),output=encode(args['conds_out']if event=='pre'else args['denoised']),positive=encode(args.get('cond_denoised')),negative=encode(args.get('uncond_denoised')),empty_denoised=encode(args.get('empty_cond_denoised'))))
       def pre1(args):observe('pre',1,args);return [v+args['input']*.03125 for v in args['conds_out']]
       def pre0(args):
        observe('pre',0,args)
        if mode=='pre_append':opts['sampler_pre_cfg_function'].append(pre1)
        if mode=='pre_replace':opts['sampler_pre_cfg_function']=[pre1]
        values=[v*.5 for v in args['conds_out']]
        return values[:2]if mode=='short'else values+[values[0]]if mode=='long'else values
       def post1(args):observe('post',1,args);return args['denoised']*.875+args['empty_cond_denoised']*.03125
       def post0(args):
        observe('post',0,args)
        if mode=='post_append':opts['sampler_post_cfg_function'].append(post1)
        if mode=='post_replace':opts['sampler_post_cfg_function']=[post1]
        return args['denoised']+args['empty_cond_denoised']*.125
       if mode in ['pre','both','pre_append','pre_replace','short','long']:opts['sampler_pre_cfg_function']=[pre0]
       if mode in ['post','both','post_append','post_replace']:opts['sampler_post_cfg_function']=[post0]
       def custom(args):raise AssertionError('PerpNeg must not call the ordinary custom CFG')
       opts['sampler_cfg_function']=custom
       def batch(m,conds,x,t,options):
        assert m is model and options is opts
        events.append(dict(event='batch',conds=conds))
        return [torch.zeros_like(x)if c is None or mode=='zeros'else x*.125+dict(p=.5,n=-.5,e=.25)[c]+t.reshape(1,1,1,1)*.0625 for c in conds]
       ns['comfy'].samplers.calc_cond_batch=batch
       guider=ns['Guider_PerpNeg']();guider.cfg=cfg;guider.neg_scale=negative_scale;guider.conds=dict(positive='p',negative='n',empty_negative_prompt='e');guider.inner_model=model
       outcome=result(lambda:guider.predict_noise(x,time,opts,123))
       pre_ids=[0 if f is pre0 else 1 for f in opts.get('sampler_pre_cfg_function',[])]
       post_ids=[0 if f is post0 else 1 for f in opts.get('sampler_post_cfg_function',[])]
       cases.append(dict(device=device,kind=str(dtype),cfg=cfg,negative_scale=negative_scale,disabled=encode(disabled),mode=mode,input=encode(x),time=encode(time),events=events,pre_ids=pre_ids,post_ids=post_ids,**outcome))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original PerpNeg guider histories; trained weights=0')
