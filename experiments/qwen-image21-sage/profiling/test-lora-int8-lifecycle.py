"""Build/run native LoRA lifecycle checks with generated synthetic safetensors."""
import argparse
import json
from pathlib import Path
import shlex
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'.imagine/lora-merge'
SOURCE=OUT/'source'
BUILD=OUT/'build'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',action='store_true')
a=p.parse_args()

def save(path,items):
    header={};data=bytearray()
    for name,dtype,shape,value in items:
        start=len(data);data.extend(value)
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[start,len(data)])
    raw=json.dumps(header,separators=(',',':')).encode()
    raw+=b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+data)

key='model.diffusion_model.transformer_blocks.0.attn.to_q'
q=bytes(((i*17+3)%127-63)&255 for i in range(128*512))
scales=struct.pack('<128f',*((i%7+1)/4096 for i in range(128)))
config=json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)).encode()
save(OUT/'synthetic-model.safetensors',[(key+'.weight','I8',[128,512],q),
    (key+'.weight_scale','F32',[128],scales),(key+'.comfy_quant','U8',[len(config)],config)])
up=struct.pack('<'+str(128*32)+'e',*((i*3%17-8)/128 for i in range(128*32)))
down=struct.pack('<'+str(32*512)+'e',*((i*7%19-9)/128 for i in range(32*512)))
save(OUT/'synthetic-lora.safetensors',[(key+'.lora_up.weight','F16',[128,32],up),
    (key+'.lora_down.weight','F16',[32,512],down),(key+'.alpha','F32',[],struct.pack('<f',32))])
fused=key.replace('attn.to_q','img_mlp.gate_up')
save(OUT/'synthetic-fused-model.safetensors',[(fused+'.weight','I8',[256,512],q+q),
    (fused+'.weight_scale','F32',[256],scales+scales),(fused+'.comfy_quant','U8',[len(config)],config)])
items=[]
for half in ['gate_layer','proj']:
    target=fused.replace('gate_up',half)
    items.extend([(target+'.lora_up.weight','F16',[128,32],up),
        (target+'.lora_down.weight','F16',[32,512],down),(target+'.alpha','F32',[],struct.pack('<f',32))])
save(OUT/'synthetic-fused-lora.safetensors',items)
save(OUT/'synthetic-full-fused-lora.safetensors',[(fused+'.lora_up.weight','F16',[256,32],up+up),
    (fused+'.lora_down.weight','F16',[32,512],down),(fused+'.alpha','F32',[],struct.pack('<f',32))])
obj=OUT/'test-lifecycle.o';binary=OUT/'test-lifecycle'
subprocess.run(['g++','-std=c++17','-O2','-I'+str(SOURCE/'include'),'-I'+str(SOURCE/'src'),
                '-I'+str(SOURCE/'ggml/include'),'-c',str(Path(__file__).with_suffix('.cpp')),'-o',str(obj)],check=True)
link=subprocess.check_output(['ninja','-t','commands','bin/sd-server'],cwd=BUILD,text=True).splitlines()[-1]
assert link.startswith(': && ') and link.endswith(' && :')
cmd=[s for s in shlex.split(link[5:-5]) if not s.endswith('.o') and not s.startswith('-Wl,--dependency-file=')]
cmd[cmd.index('-o')+1]=str(binary);cmd.insert(1,str(obj))
subprocess.run(cmd,cwd=BUILD,check=True)
if not a.run:
    print('Built synthetic lifecycle check; GPU not used.')
else:
    free,util=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().split(','))
    assert free>=13000 and util<20,'GPU not idle; no service stopped'
    subprocess.run([str(binary),str(OUT/'synthetic-model.safetensors'),str(OUT/'synthetic-lora.safetensors'),
        str(OUT/'synthetic-fused-model.safetensors'),str(OUT/'synthetic-fused-lora.safetensors'),
        str(OUT/'synthetic-full-fused-lora.safetensors')],check=True)
