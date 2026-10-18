#!/usr/bin/env python3
"""Original MOSS SDClip wrappers with bounded Qwen3.5 constructors."""
from pathlib import Path
loader=Path(__file__).with_name('reference_moss_text.py')
exec(compile(loader.read_text().split('\ngroups=[];torch.set_num_threads(2)',1)[0],str(loader),'exec'))
for name in ['comfy/text_encoders/moss_image.py','comfy/text_encoders/qwen_vl.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;sources[name]=dict(path=str(path),sha256=expected)
definitions('comfy/text_encoders/qwen_vl.py',['qwen2vl_mrope_position_ids'])
comfy.text_encoders.qwen_vl=NS(qwen2vl_mrope_position_ids=namespace['qwen2vl_mrope_position_ids'])
definitions('comfy/text_encoders/qwen35.py',['Qwen35','Qwen35ClipModel'])
comfy.text_encoders.qwen35=NS(Qwen35ClipModel=namespace['Qwen35ClipModel'])
definitions('comfy/text_encoders/moss_image.py',['MOSSImageTEModel'])
def tiny_init(self,config_dict,dtype,device,operations):
 # Replace constructor allocation only. Source wrapper and all forward methods
 # stay intact; even the real 248320-token vocabulary is kept for pad/image IDs.
 torch.nn.Module.__init__(self)
 self.model,cfg,self.factories=make_fixture_model(device,quantized,hidden=64,vocab=248320)
 self.cfg=cfg;self.num_layers=4;self.mtp=None;self.dtype=dtype
namespace['Qwen35'].__init__=tiny_init
tokens=json.loads((project/'inference/fixtures/moss-tokenizer.json').read_text())['cases']
def prompt(text,options={}):
 c=next(c for c in tokens if c['operation']=='prompt'and c['text']==text and c['options']==options)
 return [[tuple(t[:2])for t in batch]for batch in c['batches']]
groups=[];torch.set_num_threads(2)
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for quantized in [False,True]:
   te=namespace['MOSSImageTEModel'](device=device,dtype=torch.bfloat16,model_options=dict(custom_operations=operations))
   clip=te.qwen35_9b;llm=clip.transformer.model;cfg=clip.transformer.cfg
   block_indices={id(block):i for i,block in enumerate(llm.layers)}
   group=dict(device=device,quantized=quantized,config=cfg,factories=clip.transformer.factories,
       state={'model.'+n:stamp(t.detach())for n,t in llm.named_parameters()},calls=[])
   base=prompt('hello world');weighted=[[tuple(t)for t in base[0]]];weighted[0][3]=(weighted[0][3][0],1.25)
   work=[('normal',base),('empty',prompt('')),('unicode',prompt('Tomas beskriver en blå fjäril.')),
       ('left-and-eos-pad',[[(248044,1.),(248044,1.),(31,1.),(248044,1.),(71,1.)]]),
       ('weighted',weighted),('two-sections',[list(base[0]),list(base[0])]),
       ('explicit-thinking',prompt('hello world',dict(thinking=False)))]
   for label,pairs in work:
    operation_state.count=0;operation_state.interrupt_at=None;namespace['interrupt_processing']=False;events.clear()
    out=te.encode_token_weights(dict(qwen35_9b=pairs))
    group['calls'].append(dict(label=label,tokens=pairs,output=encode(out),events=events.copy(),operations=operation_state.count))
   groups.append(group);print('Captured MOSS conditioning',device,quantized,len(group['calls']),flush=True)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cpu_threads=2,groups=groups,
    provenance='Original MOSS/Qwen3.5/SDClip conditioning with initialized 64-wide four-layer language constructor and real vocabulary; no trained weights or image conditioning'),separators=(',',':'))+'\n')
print('Captured',sum(len(g['calls'])for g in groups),'MOSS conditioning calls',flush=True)
