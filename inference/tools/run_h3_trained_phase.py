#!/usr/bin/env python3
"""Run one approved trained H3 phase with owned outputs and shared reservation."""
import argparse,json,re,shutil,signal,subprocess,time
from pathlib import Path
from h3_gpu_commands import CANONICAL_SOCKET,PYTHON_ROOT,native_command,original_command,reservation_command,sha
p=argparse.ArgumentParser(description=__doc__)
for n in ['root','site','binary','own-root','reservation','entry','output']:p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--engine',choices=['native','original'],required=True)
p.add_argument('--phase',choices=['text','reference','dit','video','audio'],required=True)
p.add_argument('--run-name',required=True);p.add_argument('--reference',action='store_true');p.add_argument('--load-only',action='store_true')
p.add_argument('--exercise',choices=['cancel-after-loaded','missing-input'],help='Explicit trained lifecycle probe; never inference acceptance')
a=p.parse_args();project=Path(__file__).resolve().parents[2]
assert re.fullmatch('[a-zA-Z0-9_-]+',a.run_name)
assert a.own_root.resolve().is_relative_to(project/'comfyui/h3-stage4-20261020')
assert not a.output.exists(),'Fresh retained evidence required'
a.output.mkdir(parents=True)
work=a.own_root/'trained'/a.run_name/a.engine;work.mkdir(parents=True,exist_ok=True)
assert not(work/a.phase).exists(),'Fresh phase output required'
if a.exercise:
 assert a.engine=='native'and not a.load_only
 if a.exercise=='missing-input':
  assert a.phase in ('video','audio')
  assert not(work/'dit').exists(),'Failure probe must have no own input latents'
profile_path=project/'inference/profiles/h3-short-av.json';profile=json.loads(profile_path.read_text())
lock_path=project/'design/h3-stage4-evidence/selected-model-sha256.json';lock=json.loads(lock_path.read_text());assert lock['result']=='passed'
role=dict(text='text',reference='video_vae',dit='dit',video='video_vae',audio='audio_vae')[a.phase]
entry=next(v for v in lock['components']if v['role']==role);checkpoint=Path(entry['path'])
model_root=Path('/mnt/media/gemensam/AI/ComfyUI/models');assert checkpoint==model_root/profile['checkpoints'][role]
assert checkpoint.stat().st_size==entry['file_bytes']and checkpoint.stat().st_mtime_ns==entry['mtime_ns']
record=dict(schema=1,result='running',scope='selected trained H3 component phase',engine=a.engine,phase=a.phase,reference=a.reference,load_only=a.load_only,exercise=a.exercise,output_root=str(work),profile_sha256=sha(profile_path),model_lock_sha256=sha(lock_path),selected_component=entry,controller_sha256=sha(a.reservation),entry_sha256=sha(a.entry))
def write():(a.output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
write()
if a.reference:
 # Own deterministic RGB fixture. Both engines read byte-identical source data;
 # image preprocessing and VAE encoding still run through their original paths.
 assets=a.own_root/'trained'/a.run_name/'assets';assets.mkdir(exist_ok=True)
 source=assets/'synthetic-reference.bin';meta=assets/'synthetic-reference.json'
 if not source.exists():
  import array,hashlib
  rgb=array.array('f')
  for y in range(256):
   for x in range(384):
    cube=112<=x<272 and 48<=y<208
    rgb.extend((.1,.25,.8)if cube else(.75,.7,.6))
  data=rgb.tobytes();source.write_bytes(data)
  meta.write_text(json.dumps(dict(shape=[1,256,384,3],dtype='float32',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)),indent=2)+'\n')
 for src in [source,meta]:
  dst=work/src.name
  if not dst.exists():shutil.copyfile(src,dst)
  assert sha(src)==sha(dst)
 record['reference_sha256']=sha(source);write()
read_only=[(profile_path,profile_path),(lock_path,lock_path),(checkpoint,checkpoint)]
if a.engine=='native':
 frozen=work/('worker-'+sha(a.binary));
 if not frozen.exists():shutil.copyfile(a.binary,frozen);frozen.chmod(0o500)
 record['worker_sha256']=sha(frozen);write()
 args=native_command(project,a.root,a.site,frozen,work)
 emitted=json.loads((work/'native-command.json').read_text())
 record['bridge_sha256']=emitted['bridge_sha256'];record['bridge_provenance_sha256']=emitted['bridge_provenance_sha256'];write()
 for source,target in read_only:args+=['--ro-bind',str(source),str(target)]
 args+=['--bind',str(work),str(work),'--', '/runtime/primitives','--profile',str(profile_path),'--model-lock',str(lock_path),'--model-root',str(model_root),'--own-root',str(work),'--phase',a.phase]
 if a.reference:args+=['--reference']
 if a.load_only:args+=['--load-only']
else:
 script=project/'inference/tools/reference_h3_trained.py'
 files=[script,project/'inference/tools/reference_h3_loading.py',project/'inference/tools/reference_paths.py']
 for name in ['h3-schema.json','h3-component-modules.json']:
  source=project/'inference/resources'/name;read_only.append((source,source))
 args=original_command(project,a.site,work,files,read_only)
 env=dict(IMAGINE_H3_PROFILE=str(profile_path),IMAGINE_H3_MODEL_LOCK=str(lock_path),IMAGINE_H3_MODEL_ROOT=str(model_root),IMAGINE_H3_PHASE=a.phase,IMAGINE_H3_REFERENCE=str(int(a.reference)),IMAGINE_H3_LOAD_ONLY=str(int(a.load_only)),IMAGINE_H3_SCHEMA=str(project/'inference/resources/h3-schema.json'),IMAGINE_H3_MODULES=str(project/'inference/resources/h3-component-modules.json'))
 for k,v in env.items():args+=['--setenv',k,v]
 args+=['--chdir',str(work),'--',str(PYTHON_ROOT/'bin/python'),'-S','-B',str(script),'--site',str(a.site),'--comfy','/home/thhel/git/ComfyUI','--own-root',str(work),'--output',str(work/a.phase/'phase.json')]
 record['oracle_source_sha256']={str(path.relative_to(project)):sha(path)for path in files};write()
manifest=work/(a.phase+'-reservation.json')
command=reservation_command(a.reservation,a.entry,CANONICAL_SOCKET,'h3-stage4-'+a.run_name+'-'+a.engine+'-'+a.phase,manifest,args)
(a.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
with(a.output/'worker.log').open('w')as log:
 process=subprocess.Popen(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
 cancel_sent=False
 while process.poll()is None:
  if a.exercise=='cancel-after-loaded'and not cancel_sent and(work/a.phase/'gpu-ready.json').exists():
   process.send_signal(signal.SIGTERM);cancel_sent=True;record['cancel_sent_after_gpu_ready']=True;write()
  time.sleep(.05)
 record['returncode']=process.wait()
if manifest.exists():record['reservation']=json.loads(manifest.read_text())
phase_json=work/a.phase/'phase.json'
if phase_json.exists():
 record['phase_report']=json.loads(phase_json.read_text());shutil.copyfile(phase_json,a.output/'phase.json')
assert record['reservation']['report']['namespace_exited']is True
assert record['reservation']['report']['worker']['finish_sent']is True
if a.exercise=='cancel-after-loaded':
 passed=cancel_sent and record['returncode']!=0 and record['reservation']['report']['worker']['state']=='cancelled'
elif a.exercise=='missing-input':
 report=record.get('phase_report',{})
 passed=record['returncode']!=0 and(work/a.phase/'gpu-ready.json').exists()and report.get('error')=='No such file or directory (os error 2)'and'cleanup_error'not in report and report['unloaded']['placement']['loaded_bytes']==0 and report['unloaded']['pinned_bytes']==0
else:passed=record['returncode']==0
record['result']='passed'if passed else'failed';write()
if not passed:raise SystemExit(record['returncode']or 1)
print('PASS reserved trained',a.engine,a.phase,a.exercise or('load-only'if a.load_only else'forward'),flush=True)
