#!/usr/bin/env python3
"""Original context sampler wrapper plus real FreeNoise on bounded own tensors."""
from pathlib import Path
import os
loader=Path(__file__).with_name('reference_shared_context_evaluation.py')
exec(compile(loader.read_text().split('\ncases=[]\nwith torch.no_grad():',1)[0],str(loader),'exec'))
definitions('comfy/context_windows.py',['_sampler_sample_wrapper','apply_freenoise'])

def run_wrapper(device,dtype,multi,guides,enabled=True,selection='both',mode='observe',call='positional',seed_name='seven'):
 shapes=[torch.Size((1,1,12,2,2))]+([torch.Size((1,1,6,1,1))]if multi else[])
 vals=[data(v,device,dtype,i*5)for i,v in enumerate(shapes)];noise=ns['pack_latents'](vals)[0]if multi else vals[0]
 initial=encode(noise);positive_inner={};negative_inner={}
 if multi:
  positive_inner['latent_shapes']=ns['CONDConstant'](shapes);negative_inner['latent_shapes']=ns['CONDConstant'](shapes)
 if guides:positive_inner['guide_attention_entries']=ns['CONDConstant']([dict(latent_shape=(2,1,1)),dict(latent_shape=(1,2,2))])
 positive=[dict(model_conds=positive_inner)];negative=[dict(model_conds=negative_inner)]
 first={'positive':positive,'negative':negative}
 if selection in ('negative','negative-none'):first.pop('positive')
 if selection in ('empty','different'):first={}
 if selection=='positive-none':first['positive']=None
 if selection=='positive-empty':first['positive']=[]
 if selection=='negative-none':first['negative']=None
 second={'negative':positive}if selection=='different'else first
 events=[];reads=[0];infos=[];forward=[0];token=['return']
 class Guider:
  @property
  def conds(self):
   reads[0]+=1;events.append(dict(tag='conditions',read=reads[0]))
   if mode=='conditions-error-first'and reads[0]==1 or mode=='conditions-error-second'and reads[0]==2:raise ValueError('own guider condition read failure')
   if mode=='conditions-bad-first'and reads[0]==1:return 3
   if mode=='conditions-bad-second'and reads[0]==2:return []
   if selection=='mutate-positive'and reads[0]==2:first.pop('positive',None)
   return first if reads[0]==1 else second
 guider=Guider();h=handler(dim=2,context_length=4,context_overlap=1,freenoise=enabled)
 options={'context_handler':h};sigmas=['sigmas'];callback=['callback']
 seeds={'seven':7,'zero':0,'negative':-1,'min':-2**63,'max':2**64-1,'bool':True,'float':1.5,'none':None,'list':[],'dict':{},'tensor':torch.tensor(7,dtype=torch.int64,device=device),'str':'7'}
 seed=seeds[seed_name];extra={'model_options':options,'seed':seed};kwargs={'own':['extra']}
 if mode=='extra-bad':extra=3
 if mode=='options-missing':del extra['model_options']
 if mode=='options-none':extra['model_options']=None
 if mode=='options-bad':extra['model_options']=3
 if mode=='handler-missing':del options['context_handler']
 if mode=='handler-none':options['context_handler']=None
 if mode=='handler-bad':options['context_handler']=3
 if mode=='seed-missing':del extra['seed']
 args=[guider,sigmas,extra,callback,noise];names=['guider','sigmas','extra_args','callback','noise']
 if call=='keywords':kwargs.update(zip(names,args));args=[]
 if call=='mixed':kwargs.update(zip(names[2:],args[2:]));args=args[:2]
 if call=='extra':args+=['extra',7]
 if call=='duplicate':kwargs['guider']=guider
 if call=='receiver':kwargs['executor']=3
 if call.startswith('missing-'):args=args[:int(call.removeprefix('missing-'))]
 def info(message):
  infos.append(message);events.append(dict(tag='info',message=message))
  if mode=='info-error-first'and len(infos)==1 or mode=='info-error-second'and len(infos)==2:raise ValueError('own sampler FreeNoise info failure')
  if mode=='info-mutate'and len(infos)==1:h.context_length+=1
 ns['logging']=NS(info=info)
 def executor(*passed,**kw):
  forward[0]+=1;events.append(dict(tag='executor',noise=encode(passed[4]),same=[passed[0]is guider,passed[1]is sigmas,passed[2]is extra,passed[3]is callback,passed[4]is noise],extra=encode(passed[5:]),keys=list(kw),own_same=kw.get('own')is kwargs['own']))
  if mode=='executor-error':raise ValueError('own sampler executor failure')
  return token
 code=ns['IndexListContextHandler']._apply_freenoise.__code__
 def trace(frame,event,arg):
  if frame.f_code is code and event=='call':
   local=frame.f_locals;chosen=local['conds'][0]
   events.append(dict(tag='freenoise',same_noise=local['noise']is noise,selected=value(chosen),positive=chosen is positive,negative=chosen is negative,seed=encode(local['seed'])))
  return trace
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()if device.startswith('cuda')else None
 previous=sys.gettrace();sys.settrace(trace)
 try:out=ns['_sampler_sample_wrapper'](executor,*args,**kwargs);outcome=dict(ok=True,same_return=out is token)
 except Exception as e:reject_resource_failure(e);outcome=dict(error=type(e).__name__,message=str(e))
 finally:sys.settrace(previous)
 unchanged=torch.equal(cpu,torch.get_rng_state())and(gpu is None or torch.equal(gpu,torch.cuda.get_rng_state()))
 return dict(params=dict(device=device,dtype=str(dtype),multi=multi,guides=guides,enabled=enabled,selection=selection,mode=mode,call=call,seed=seed_name),input=initial,events=events,**outcome,post=encode(noise),length=h.context_length,reads=reads[0],forwards=forward[0],global_rng_unchanged=unchanged)

cases=[];devices=os.environ.get('IMAGINE_CONTEXT_DEVICES','cpu,cuda:0').split(',')
with torch.no_grad():
 for device,dtype,multi,guides,enabled,selection in product(devices,[torch.float32,torch.bfloat16],[False,True],[False,True],[False,True],['both','negative','empty','positive-none','positive-empty','negative-none','different','mutate-positive']):cases.append(run_wrapper(device,dtype,multi,guides,enabled,selection))
 for device,dtype,mode in product(devices,[torch.float32,torch.bfloat16],['extra-bad','options-missing','options-none','options-bad','handler-missing','handler-none','handler-bad','seed-missing','conditions-error-first','conditions-error-second','conditions-bad-first','conditions-bad-second','info-error-first','info-error-second','info-mutate','executor-error']):cases.append(run_wrapper(device,dtype,True,True,mode=mode))
 for device,dtype,call in product(devices,[torch.float32,torch.bfloat16],['keywords','mixed','extra','duplicate','receiver','missing-0','missing-1','missing-2','missing-3','missing-4']):cases.append(run_wrapper(device,dtype,True,True,call=call))
 for device,dtype,multi,seed in product(devices,[torch.float32,torch.bfloat16],[False,True],['zero','negative','min','max','bool','float','none','list','dict','tensor','str']):cases.append(run_wrapper(device,dtype,multi,True,seed_name=seed))
assert not any(c.get('error')=='NameError'for c in cases),'Missing source dependency in own harness'
assert all(c.get('ok')for c in cases if c['params']['mode']=='observe'and c['params']['call']in ('positional','keywords','mixed','extra')and c['params']['seed']=='seven'),'Unexpected normal wrapper failure'
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,devices=devices,cases=cases),separators=(',',':'))+'\n');print('Captured',len(cases),'whole sampler wrapper cases including original FreeNoise, descriptor reads, seed/error order and forwarding',flush=True)
