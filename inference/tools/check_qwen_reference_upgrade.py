#!/usr/bin/env python3
"""Three explicitly registered, trained Qwen upgrade controls; never ordinary tests."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['root','site','comfy','model-root','binary']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--prefix',required=True)
p.add_argument('--variants',nargs='+',choices=['int8','bf16','turbo'],default=['int8','bf16','turbo'])
p.add_argument('--original-first',action='store_true',help='Diagnostic execution order only; original policy and numerical settings remain unchanged')
p.add_argument('--replay-python-prefix',help='Separate diagnostic: replay only the saved original DiT placements, never change the automatic default')
a=p.parse_args()
assert re.fullmatch('[a-zA-Z0-9_-]+',a.prefix)
project=Path(__file__).resolve().parents[2];os.chdir(project)
reports=a.root/'runtime'/a.prefix;reports.mkdir(exist_ok=False)
binary=reports/'qwen21-driver';shutil.copyfile(a.binary,binary);binary.chmod(0o755)
binary_sha=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()
results=[]
for variant in a.variants:
 profile=project/'inference/profiles'/f'qwen21-cfg-er-sde-{variant}.json';data=json.loads(profile.read_text())
 available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
 required=2*max((a.model_root/v).stat().st_size for v in data['checkpoints'].values())+4*1024**3
 item=dict(variant=variant,binary_sha256=binary_sha,host_guard=dict(available=available,required=required,passed=available>=required),commands=[])
 if available<required:
  item['deferred']='host RAM guard';results.append(item);continue
 name=a.prefix+'-'+variant;rust=a.root/'runtime'/(name+'-rust');rust.mkdir();(rust/'memory-observation-enabled').touch()
 reference=a.root/'runtime'/(name+'-python')
 if a.replay_python_prefix:
  assert re.fullmatch('[a-zA-Z0-9_-]+',a.replay_python_prefix)
  reference=a.root/'runtime'/(a.replay_python_prefix+'-'+variant+'-python')
  original=json.loads((reference/'result.json').read_text());assert original['profile']==data
  fields=['phase','resident','loaded_bytes','buffer_bytes','lowvram_passes','model_lowvram','sampler_buffer']
  phases=[{k:phase[k]for k in fields}for phase in json.loads((reference/'memory-policy.json').read_text())if phase['phase']in ['dit_initial','dit_sampling']]
  assert [x['phase']for x in phases]==['dit_initial','dit_sampling']
  (rust/'reference-memory-plan.json').write_text(json.dumps(dict(profile=data,phases=phases),indent=2)+'\n')
  item['diagnostic_memory_plan']=True;item['original_reference']=str(reference)
 isolation=[sys.executable,'-B',str(project/'inference/tools/test_isolation.py'),
   '--root',str(a.root),'--site',str(a.site),'--binary',str(binary),
   '--aot-library',str(a.root/'triton-aot/libimagine_triton_fp8.so'),'--aot-fixture',str(a.root/'triton-aot/probe.json'),
   '--int8-library',str(a.root/'int8-aot/libimagine_int8_probe.so'),'--int8-fixture',str(a.root/'int8-aot/probe.json'),
   '--backend-fixture',str(a.root/'runtime/backend-contract.json'),'--aimdo-library',str(a.root/'aimdo/aimdo.so'),
   '--pillow-library',str(a.root/'pillow/libimagine_pillow.so'),'--qwen-driver-profile',str(profile),
   '--qwen-model-root',str(a.model_root),'--qwen-mode','run','--qwen-run-name',name+'-rust']
 commands=[('rust',isolation),('python',['bash','inference/tools/launch_reference.sh','inference/tools/reference_qwen_editing.py',
   '--site',str(a.site),'--comfy',str(a.comfy),'--model-root',str(a.model_root),'--own-root',str(a.root/'runtime'/(name+'-python')),
   '--profile',str(profile),'--memory-observation']),('comparison',['bash','inference/tools/launch_reference.sh','inference/tools/compare_qwen_first_image.py',
   '--site',str(a.site),'--rust',str(rust/'first-image'),'--python',str(reference),
   '--output',str(reports/f'{variant}-comparison.json')])]
 if a.original_first:
  assert not a.replay_python_prefix,'Original-first cannot be combined with saved placement replay'
  commands=[commands[1],commands[0],*commands[2:]]
 if not a.replay_python_prefix:
  commands.append(('automatic-acceptance',[sys.executable,'-B',str(project/'inference/tools/check_qwen_automatic_acceptance.py'),'--rust',str(rust/'first-image'),'--original',str(reference),'--comparison',str(reports/f'{variant}-comparison.json'),'--output',str(reports/f'{variant}-acceptance.json')]))
 for label,command in commands:
  if label=='python'and a.replay_python_prefix:continue
  print(variant,label,'started',flush=True)
  with(reports/f'{variant}-{label}.log').open('w')as out:run=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
  item['commands'].append(dict(label=label,returncode=run.returncode));print(variant,label,run.returncode,flush=True)
  if label=='rust'and run.returncode==0:shutil.copyfile(a.root/'runtime/qwen21-run-isolation.json',reports/f'{variant}-isolation.json')
  if run.returncode and label!='comparison':break
 results.append(item);(reports/'results.json').write_text(json.dumps(results,indent=2)+'\n')
(reports/'results.json').write_text(json.dumps(results,indent=2)+'\n')
sys.exit(int(any(x.get('deferred')or any(c['returncode']for c in x['commands'])for x in results)))
