"""Loaded-layout graph plans and synthetic arithmetic checks; no diffusion."""
from pathlib import Path
import argparse,json,os,subprocess
p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);p.add_argument('--early-only',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];build=root/'runs/vae-dynamic-blocks-build';out=build/'checks';out.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('SD_','GGML_'))};env.update(json.loads((root/'baselines/1600x896-v1/baseline-environment.json').read_text()))
results=json.loads((out/'results.json').read_text()) if a.early_only else {}
def run(name,cmd):
 print('START',name,flush=True)
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT)
 results[name]=r.returncode;(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',name,r.returncode,flush=True)
vae=a.model_root/'vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors'
for w,h in ([] if a.early_only else [(1600,896),(4096,2304)]):
 for flag in [0,1]:run(f'probe-{w}-{flag}',[build/'measure-vae-dynamic-blocks',flag,vae,w,h])
for w,h,ci,co,up in [(100,56,1152,1152,1),(200,112,1152,1152,0),(200,112,1152,1152,1),(400,224,576,576,0),(800,448,576,288,0),(1600,896,288,144,0),(1600,896,144,144,0),(400,224,576,576,1),(800,448,288,288,1),(256,144,576,576,1),(512,288,288,288,1),(1024,576,288,144,0),(65,129,288,144,0)]:
 if a.early_only and ci != 1152:continue
 run(f'component-{w}-{h}-{ci}-{co}-{up}',[build/'test-vae-dynamic-blocks',w,h,ci,co,up])
print('COMPLETE',out,flush=True)
