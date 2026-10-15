"""CPU-only synthetic comparison against AIO's source algorithms, never user images."""
import argparse,importlib.util,json,pathlib,subprocess,sys,tempfile,types
import numpy as np
import torch
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--aio-reference',type=pathlib.Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
package=types.ModuleType('aio_numeric');package.__path__=[str(a.aio_reference/'services')];sys.modules['aio_numeric']=package
from aio_numeric.color_match import guided_chrominance_transfer
from aio_numeric.tone_match import match_reference_tone
root=pathlib.Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
 tmp=pathlib.Path(tmp);cases=[];expected=[]
 rng=np.random.default_rng(17)
 for i,(h,w,rh,rw) in enumerate([(64,96,64,96),(79,127,81,99),(520,768,256,384)]):
  source=rng.integers(0,256,(h,w,4),dtype=np.uint8);reference=rng.integers(0,256,(rh,rw,4),dtype=np.uint8)
  source[...,3]=255;reference[...,3]=255;reference[:10,:10,3]=0
  sp=tmp/f'source{i}.png';rp=tmp/f'reference{i}.png';Image.fromarray(source).save(sp);Image.fromarray(reference).save(rp)
  s=torch.from_numpy(source.astype(np.float32)/255)[None];r=torch.from_numpy(reference.astype(np.float32)/255)[None]
  for color,brightness,contrast in [(0,0,0),(1,0,0),(0,0,.5),(1,0,.5),(.8,.6,.7)]:
   out=guided_chrominance_transfer(s,r,color);out=match_reference_tone(out,r,brightness,contrast)
   expected.append(np.rint(out.numpy()[0]*255).astype(np.int16));cases.append(dict(source=str(sp),reference=str(rp),output=str(tmp/f'out{len(cases)}.png'),color=color,brightness=brightness,contrast=contrast))
 result=subprocess.run([str(root/'target/debug/examples/workflow-color')],input=''.join(json.dumps(c)+'\n' for c in cases),text=True,capture_output=True,check=True)
 differences=[]
 for case,want in zip(cases,expected):
  actual=np.array(Image.open(case['output']).convert('RGBA')).astype(np.int16);diff=np.abs(actual-want);differences.append(int(diff.max()));assert diff.max()<=1,(case,diff.max(),np.mean(diff))
 print(json.dumps({'comparisons':len(cases),'maximum_channel_error_8bit':max(differences),'tolerance':'one output quantization level; no image/content normalization'}))
