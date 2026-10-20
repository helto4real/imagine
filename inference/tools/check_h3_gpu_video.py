#!/usr/bin/env python3
"""Reviewed tiny CUDA comparison under the canonical Ticket/PID-namespace runner."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__)
for n in ['root','site','binary','own-root','reservation','entry','socket','output']:p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--component',choices=['video','audio','dit','conditioning','sampling','quantized'],default='video')
p.add_argument('--reuse-oracle',type=Path,help='Reuse verified original results for unchanged synthetic inputs and oracle source')
a=p.parse_args();project=Path(__file__).resolve().parents[2]
assert a.socket==Path('/home/thhel/ChatGPT/imagine/.imagine/gpu.sock')
assert a.own_root.resolve().is_relative_to(project/'comfyui/h3-stage4-20261020')
assert not a.output.exists(),'Use a fresh evidence directory'
a.output.mkdir(parents=True);work=a.own_root/('gpu-'+a.component)/a.output.name;work.mkdir(parents=True)
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
frozen=work/'primitives';shutil.copyfile(a.binary,frozen);frozen.chmod(0o500)
record=dict(schema=1,result='running',scope='tiny synthetic CUDA H3 '+a.component,trained_weights=0,socket=str(a.socket),binary_sha256=sha(frozen),controller_sha256=sha(a.reservation),entry_sha256=sha(a.entry),runs=[])
def write(): (a.output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
write()
def reserved(label,args):
 # No shell and no implicit mounts: the exact argv is retained for review.
 args=[str(x)for x in args];assert '--'in args
 manifest=work/(label+'-reservation.json')
 command=[str(a.reservation),'--process-form','bwrap-pid-namespace','--namespace-entry',str(a.entry),'--socket',str(a.socket),'--job','h3-stage4-'+a.output.name+'-'+label,'--bytes','1','--timeout','600','--manifest',str(manifest),'--','/usr/bin/bwrap',*args]
 (a.output/(label+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
 with (a.output/(label+'.log')).open('w')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=project)
 entry=dict(phase=label,returncode=result.returncode)
 if manifest.exists():entry['reservation']=json.loads(manifest.read_text())
 record['runs'].append(entry)
 if result.returncode:record['result']='failed'
 write()
 if result.returncode:raise SystemExit(result.returncode)
 assert entry['reservation']['report']['namespace_exited']is True
 print('PASS reserved',label,flush=True)
ref=reference_dir(project);manifest=json.loads((ref/'source-manifest.json').read_text());mounts=[]
for e in manifest:
 if e['root']!='comfy':continue
 source=Path('/home/thhel/git/ComfyUI')/e['path'];assert sha(source)==e['sha256'],source
 mounts+=['--ro-bind',source,source]
python_root=Path('/home/thhel/.pyenv/versions/3.12.13')
for path in [ref/'source-manifest.json',ref/'reference-lock.json',project/'design/reference-current.json']:
 mounts+=['--ro-bind',path,path]
script='reference_h3_gpu_video.py'if a.component=='video'else'reference_h3_gpu_quantized.py'if a.component=='quantized'else'reference_h3_gpu_components.py'
for name in dict.fromkeys([script,'reference_h3_loading.py','reference_paths.py',*[f'reference_h3_{stage}.py'for stage in ['audio','dit','conditioning','sampling']]]):
 path=project/'inference/tools'/name;mounts+=['--ro-bind',path,path]
for name,target in [('h3-video.json','h3-video-input.json'),('h3-video-quant.json','h3-video-quant-input.json'),('h3-loading.json','h3-loading-input.json')]:
 path=project/'inference/fixtures'/name;mounts+=['--ro-bind',path,work/target]
record['inputs']={name:sha(project/'inference/fixtures'/name)for name in ['h3-video.json','h3-video-quant.json','h3-loading.json']}
record['reference_script_sha256']=sha(project/'inference/tools'/script);write()
devices=[]
for n in ['nvidia0','nvidiactl','nvidia-uvm','nvidia-uvm-tools']:
 path=Path('/dev')/n
 if path.exists():devices+=['--dev-bind',path,path]
args=['--unshare-all','--clearenv','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--ro-bind','/usr/lib','/usr/lib','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64','--ro-bind',python_root,python_root,'--ro-bind',a.site,a.site,'--bind',work,work,'--ro-bind','/sys','/sys',*mounts,*devices]
if a.component in ('conditioning','audio','quantized'):
 # Torch 2.13's source bmm-outer-product override invokes Triton's own driver
 # builder. Expose only the existing host compiler/header inputs it needs.
 args+=['--ro-bind','/usr/include','/usr/include']
 for tool in ['/usr/bin/cc','/usr/bin/as','/usr/bin/ld']:
  args+=['--ro-bind',tool,tool]
env=dict(IMAGINE_H3_COMPONENT=a.component,PYTHONHOME=str(python_root),PYTHONDONTWRITEBYTECODE='1',HOME=str(work),XDG_CACHE_HOME=str(work/'cache'),TORCH_HOME=str(work/'torch'),HF_HOME=str(work/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(work/'triton'),TRITON_LIBCUDA_PATH='/usr/lib',CC='/usr/bin/cc',CUDA_CACHE_PATH=str(work/'cuda'),TMPDIR='/tmp',LD_LIBRARY_PATH=f'{python_root}/lib:{a.site}/torch/lib:{a.site}/nvidia/cu13/lib:{a.site}/nvidia/cudnn/lib',PATH=f'{python_root}/bin:/usr/bin')
for k,v in env.items():args+=['--setenv',k,v]
oracle=work/(a.component+'-gpu.json')
args+=['--chdir',work,'--',python_root/'bin/python','-S','-B',project/'inference/tools'/script,'--site',a.site,'--comfy','/home/thhel/git/ComfyUI','--own-root',work,'--output',oracle]
if a.reuse_oracle is None:reserved('original',args)
else:
 previous=json.loads((a.reuse_oracle.parent/'selection.json').read_text())
 assert previous['inputs']==record['inputs']and previous['reference_script_sha256']==record['reference_script_sha256']
 assert previous['oracle_sha256']==sha(a.reuse_oracle)
 assert any(r['phase']=='original'and r['returncode']==0 and r['reservation']['report']['namespace_exited']for r in previous['runs'])
 shutil.copyfile(a.reuse_oracle,oracle);record['reused_original_selection']=str(a.reuse_oracle.parent/'selection.json')
record['oracle_sha256']=sha(oracle);shutil.copyfile(oracle,a.output/(a.component+'-gpu.json'));write()
emitted=work/'native-command.json'
command=[sys.executable,'-S','-B',str(project/'inference/tools/test_isolation.py'),'--root',str(a.root),'--site',str(a.site),'--binary',str(frozen),'--runtime-root',str(work/'native-runtime'),'--emit-command',str(emitted)]
if os.environ.get('IMAGINE_H3_KITCHEN_LIBRARY'):
 command+=['--kitchen-library',os.environ['IMAGINE_H3_KITCHEN_LIBRARY'],'--kitchen-provenance',os.environ['IMAGINE_H3_KITCHEN_PROVENANCE']]
for flag,path in [('aot-library','triton-aot/libimagine_triton_fp8.so'),('aot-fixture','triton-aot/probe.json'),('int8-library','int8-aot/libimagine_int8_probe.so'),('int8-fixture','int8-aot/probe.json'),('aimdo-library','aimdo/aimdo.so'),('pillow-library','pillow/libimagine_pillow.so')]:command+=['--'+flag,str(a.root/path)]
tests=dict(quantized='h3_quantized_tests::h3_tiny_cuda_selected_quantized_language_and_dit_match_original',video='h3_video_tests::h3_tiny_cuda_video_matches_original_selected_policy_without_python_in_native_worker',audio='h3_audio_tests::h3_tiny_cuda_audio_matches_original_encoder_causal_attention_and_pcm',dit='h3_dit_tests::h3_tiny_cuda_dit_matches_original_all_dense_forwards',conditioning='h3_conditioning_tests::h3_tiny_cuda_conditioning_matches_original_raw_text_vision_deepstack',sampling='h3_sampling_tests::h3_tiny_cuda_sampling_matches_original_complete_av_chain')
command+=['--backend-fixture',str(project/'inference/fixtures/kitchen-0238-backend-contract.json'),'--test-filter',tests[a.component]]
subprocess.run(command,cwd=project,check=True)
emitted_record=json.loads(emitted.read_text());record['bridge_sha256']=emitted_record['bridge_sha256'];record['bridge_provenance_sha256']=emitted_record['bridge_provenance_sha256'];write()
args=emitted_record['command'][1:]
# First occurrence is the ro-bind target; command position follows --chdir.
index=args.index('--chdir')+2
args[index:index]=['--ro-bind',str(oracle),'/runtime/h3-video-gpu.json','--setenv','IMAGINE_H3_GPU_FIXTURE','/runtime/h3-video-gpu.json','--']
args+=['--ignored']
reserved('native',args)
assert sha(frozen)==record['binary_sha256']and sha(oracle)==record['oracle_sha256']
native_log=(a.output/'native.log').read_text();assert 'test result: ok. 1 passed; 0 failed;'in native_log
record['result']='passed';write()
print('PASS tiny CUDA',a.component,'operations; no trained-model acceptance',flush=True)
