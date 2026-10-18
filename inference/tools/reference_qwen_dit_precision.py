#!/usr/bin/env python3
"""Additional FP16/FP32 complete small DiT forwards, using original classes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_dit.py')
text=loader.read_text()
preamble,body=text.split('\nmodels=[]',1)
exec(compile(preamble,str(loader),'exec'))
body=body.split('\ncontrols=[]',1)[0]
old="for variant in ['bf16','int8_convrot','turbo_bf16','turbo_int8_convrot']:"
assert body.count(old)==1
body=body.replace(old,"for variant in ['fp16','fp32']:\n   dtype=torch.float16 if variant=='fp16'else torch.float32")
body=body.replace('torch.bfloat16','dtype')
exec(compile('models=[]'+body,str(loader),'exec'))
a.output.write_text(json.dumps(dict(schema=1,provenance='Original complete DiT with initialized small FP16/FP32 parameters; no trained weights or full model allocation',sources=sources,models=models,controls=[]),separators=(',',':'))+'\n')
print('Captured',len(models),'precision groups and',sum(len(g['calls'])for g in models),'complete forwards')
