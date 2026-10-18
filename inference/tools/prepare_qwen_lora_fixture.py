#!/usr/bin/env python3
"""Create a bounded synthetic text-encoder LoRA; never allocate model weights."""
import argparse,hashlib,json,math,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.site))
import torch
from safetensors.torch import save_file
project=Path(__file__).resolve().parents[2]
schema=json.loads((project/'inference/resources/qwen-image21-schema.json').read_text())
key='model.layers.0.self_attn.q_proj.weight'
shape=next(p['shape']for p in schema['models']['text']['parameters']if p['name']==key)
rank=2;assert math.prod(shape)<100_000_000
stem='text_encoders.qwen3vl_8b.transformer.'+key.removesuffix('.weight')
state={}
for suffix,sizes,offset in [('lora_B.weight',[shape[0],rank],1),('lora_A.weight',[rank,shape[1]],3)]:
    state[f'{stem}.{suffix}']=((torch.arange(math.prod(sizes),dtype=torch.float32)+offset).sin()*.015).reshape(sizes).to(torch.float16)
a.output.parent.mkdir(parents=True,exist_ok=True)
save_file(state,str(a.output),metadata={'purpose':'isolated bounded synthetic Qwen3-VL LoRA probe; no trained or production data'})
report=dict(schema=1,algorithm='CPU Float32 sin(arange + offset) * .015 -> FP16; rank 2',target=key,target_shape=shape,rank=rank,bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
