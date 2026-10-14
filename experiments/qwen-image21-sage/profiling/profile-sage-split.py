"""Profile one split-prefix mode using only the approved synthetic benchmark."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from PIL import Image
experiment=Path(__file__).resolve().parents[1]
build=experiment/'runs/sage-split-kv-build'
mode=int(sys.argv[1]);assert mode in (0,1,2)
binary=Path(sys.argv[2]).resolve()
env={**os.environ,**json.loads((experiment/'runs/20261013T203045Z-activity-q759cwrf/trace-environment.json').read_text()),'SD_SAGE_SPLIT_V':str(int(mode>=1)),'SD_SAGE_SPLIT_K':str(int(mode>=2)),'SD_QWEN21_BF16_GATE_UP':'0','PYTHONDONTWRITEBYTECODE':'1'}
for key in ('LD_PRELOAD','QWEN_ACTIVITY_TRACE'):env.pop(key,None)
cmd=[sys.executable,str(experiment/'profiling/run.py'),'--binary',str(binary),'--threads','11','--default-vram','--trace-library',str(experiment/'runs/activity-profiler/activity.so')]
proc=subprocess.run(cmd,env=env,capture_output=True,text=True)
log=build/f'profile-{mode}-{binary.name}.log';log.write_text(proc.stdout+proc.stderr);proc.check_returncode()
item=json.loads(proc.stdout.splitlines()[-1]);run=Path(item['run'])
digest=hashlib.sha256(Image.open(run/'image.png').convert('RGB').tobytes()).hexdigest()
assert digest=='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c',digest
a=subprocess.run([sys.executable,str(experiment/'profiling/analyze-sage-split.py'),str(run),str(mode)],capture_output=True,text=True)
(run/'sage-analysis.log').write_text(a.stdout+a.stderr);a.check_returncode()
item.update(mode=mode,binary=str(binary),image_sha256=digest)
(build/f'profile-{mode}-{binary.name}.json').write_text(json.dumps(item,indent=2)+'\n');print(json.dumps(item),flush=True)
