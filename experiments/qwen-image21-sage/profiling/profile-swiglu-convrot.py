"""Profile the full reference image with packed SwiGLU/ConvRot fusion off/on."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image

here=Path(__file__).resolve().parent
experiment=here.parent
out=Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-swiglu-profiles-'),dir=experiment/'runs'))
(out/Path(__file__).name).write_text(Path(__file__).read_text())
print(out,flush=True)
results=[]
for fused in [0,1]:
 env={**os.environ,'GGML_CUDA_FUSED_INT8_GATE_UP':'1','GGML_CUDA_FUSED_INT8_PROJECTION':'1','GGML_CUDA_CONVROT_THREADS':'512','GGML_CUDA_FUSED_SWIGLU_CONVROT':str(fused)}
 command=[sys.executable,str(here/'run.py'),'--binary',str(experiment/'runs/swiglu-convrot-build/sd-cli'),'--threads','11','--trace-library',str(experiment/'runs/activity-profiler/activity.so')]
 proc=subprocess.run(command,env=env,capture_output=True,text=True)
 (out/f'replay-{fused}.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
 item=json.loads(proc.stdout.splitlines()[-1]);run=Path(item['run'])
 proc=subprocess.run([sys.executable,str(here/'analyze.py'),str(run),'--fused-gate-up','--fused-projection'],capture_output=True,text=True)
 (out/f'analysis-{fused}.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
 profile=json.loads((run/'profile-summary.json').read_text());assert profile['boundary_crossing_events']==0
 image=Image.open(run/'image.png').convert('RGB')
 pixels=hashlib.sha256(image.tobytes()).hexdigest()
 validation={'size':image.size,'RGB_sha256':pixels,'bit_identical_reference':pixels=='253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'}
 (run/'image-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 assert image.size==(1600,896) and validation['bit_identical_reference']
 shapes=list(csv.DictReader((run/'kernel-shapes.csv').open()))
 convrot=[r for r in shapes if 'quantize_rowwise_i8_convrot_cuda' in r['name']]
 swiglu=[r for r in shapes if 'unary_gated_op_kernel' in r['name'] and 'op_silu' in r['name']]
 small=[r for r in convrot if '<512, false>' in r['name']]
 down=[r for r in convrot if f'<1024, {"true" if fused else "false"}>' in r['name']]
 assert sum(int(r['count']) for r in small)==7680
 assert sum(int(r['count']) for r in down)==2560
 assert sum(int(r['count']) for r in convrot)==10240
 assert sum(int(r['count']) for r in swiglu)==(0 if fused else 2560)
 assert all(r['block']=='1024,1,1' for r in down)
 item.update(fused=bool(fused),image=validation,convrot_4096_seconds=sum(float(r['seconds']) for r in small),convrot_down_seconds=sum(float(r['seconds']) for r in down),swiglu_seconds=sum(float(r['seconds']) for r in swiglu),profile=profile)
 results.append(item)
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({k:v for k,v in item.items() if k!='profile'}),flush=True)
print('DONE',out,flush=True)
