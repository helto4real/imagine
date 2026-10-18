#!/usr/bin/env python3
"""Explicit local MOSS preflight or trained comparison, always sequential."""
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['root','site','comfy','model-root','binary']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--prefix',required=True);p.add_argument('--mode',choices=['preflight','run'],default='preflight')
p.add_argument('--profile',type=Path,help='Explicit registered MOSS development profile')
p.add_argument('--reference-run-name',help='Reuse one explicitly named own original run with identical profile/checkpoint hashes')
a=p.parse_args();assert re.fullmatch('[a-zA-Z0-9_-]+',a.prefix)
project=Path(__file__).resolve().parents[2]
report=a.root/'runtime'/a.prefix;report.mkdir(exist_ok=False)
binary=report/'moss-driver';shutil.copyfile(a.binary,binary);binary.chmod(0o755)
profile=a.profile or project/'inference/profiles/moss-first-int8.json';data=json.loads(profile.read_text())
available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
required=2*max((a.model_root/v).stat().st_size for v in data['checkpoints'].values())+4*1024**3
assert a.mode=='preflight'or available>=required,('Host RAM guard',available,required)
results=dict(schema=1,mode=a.mode,binary_sha256=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
 host_ram_guard=dict(available=available,required=required),commands=[])
rust=a.root/'runtime'/(a.prefix+'-rust')
isolation=[sys.executable,'-B',str(project/'inference/tools/test_isolation.py'),'--root',str(a.root),'--site',str(a.site),'--binary',str(binary),
 '--aot-library',str(a.root/'triton-aot/libimagine_triton_fp8.so'),'--aot-fixture',str(a.root/'triton-aot/probe.json'),
 '--int8-library',str(a.root/'int8-aot/libimagine_int8_probe.so'),'--int8-fixture',str(a.root/'int8-aot/probe.json'),
 '--backend-fixture',str(a.root/'runtime/backend-contract.json'),'--aimdo-library',str(a.root/'aimdo/aimdo.so'),
 '--driver-profile',str(profile),'--model-root',str(a.model_root),'--driver-mode',a.mode,'--run-name',a.prefix+'-rust']
commands=[('rust',isolation)]
if a.mode=='run':
 reference=a.root/'runtime'/(a.prefix+'-python')
 if a.reference_run_name:
  assert re.fullmatch('[a-zA-Z0-9_-]+',a.reference_run_name)
  reference=a.root/'runtime'/a.reference_run_name
  assert json.loads((reference/'result.json').read_text())['profile']==data
  results['original_run_reused']=str(reference)
 else:
  commands += [('python',['bash',str(project/'inference/tools/launch_reference.sh'),str(project/'inference/tools/reference_moss_first_image.py'),
   '--site',str(a.site),'--comfy',str(a.comfy),'--model-root',str(a.model_root),'--own-root',str(reference),'--profile',str(profile)])]
 commands += [('comparison',['bash',str(project/'inference/tools/launch_reference.sh'),str(project/'inference/tools/compare_moss_first_image.py'),
  '--site',str(a.site),'--rust',str(rust/'first-image'),'--python',str(reference),'--output',str(report/'comparison.json')])]
for label,command in commands:
 print('MOSS',label,'started',flush=True)
 with(report/(label+'.log')).open('w')as output:run=subprocess.run(command,cwd=project,stdout=output,stderr=subprocess.STDOUT)
 results['commands'].append(dict(label=label,returncode=run.returncode));(report/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print('MOSS',label,run.returncode,flush=True)
 if label=='rust'and run.returncode==0:shutil.copyfile(a.root/'runtime'/('moss-'+a.mode+'-isolation.json'),report/'isolation.json')
 if run.returncode:sys.exit(1)
