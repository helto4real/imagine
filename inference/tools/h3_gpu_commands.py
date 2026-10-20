"""Development-only reviewed command composition. No GPU work at import time."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from reference_paths import reference_dir
PYTHON_ROOT=Path('/home/thhel/.pyenv/versions/3.12.13')
CANONICAL_SOCKET=Path('/home/thhel/ChatGPT/imagine/.imagine/gpu.sock')
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def original_command(project,site,work,code_files,read_only_files):
 ref=reference_dir(project);mounts=[]
 for item in json.loads((ref/'source-manifest.json').read_text()):
  if item['root']!='comfy':continue
  source=Path('/home/thhel/git/ComfyUI')/item['path'];assert sha(source)==item['sha256'],source
  mounts+=['--ro-bind',str(source),str(source)]
 paths=[ref/'source-manifest.json',ref/'reference-lock.json',project/'design/reference-current.json',*code_files]
 for path in paths:mounts+=['--ro-bind',str(path),str(path)]
 for source,target in read_only_files:mounts+=['--ro-bind',str(source),str(target)]
 args=['--unshare-all','--clearenv','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--ro-bind','/usr/lib','/usr/lib','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64','--ro-bind',str(PYTHON_ROOT),str(PYTHON_ROOT),'--ro-bind',str(site),str(site),'--bind',str(work),str(work),'--ro-bind','/sys','/sys','--ro-bind','/usr/include','/usr/include',*mounts]
 for tool in ['/usr/bin/cc','/usr/bin/as','/usr/bin/ld']:args+=['--ro-bind',tool,tool]
 for name in ['nvidia0','nvidiactl','nvidia-uvm','nvidia-uvm-tools']:
  path=Path('/dev')/name
  if path.exists():args+=['--dev-bind',str(path),str(path)]
 env=dict(PYTHONHOME=str(PYTHON_ROOT),PYTHONDONTWRITEBYTECODE='1',HOME=str(work),XDG_CACHE_HOME=str(work/'cache'),TORCH_HOME=str(work/'torch'),HF_HOME=str(work/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(work/'triton'),TRITON_LIBCUDA_PATH='/usr/lib',CC='/usr/bin/cc',CUDA_CACHE_PATH=str(work/'cuda'),TMPDIR='/tmp',LD_LIBRARY_PATH=f'{PYTHON_ROOT}/lib:{site}/torch/lib:{site}/nvidia/cu13/lib:{site}/nvidia/cudnn/lib',PATH=f'{PYTHON_ROOT}/bin:/usr/bin')
 for k,v in env.items():args+=['--setenv',k,v]
 return args
def native_command(project,root,site,binary,work):
 emitted=work/'native-command.json'
 command=[sys.executable,'-S','-B',str(project/'inference/tools/test_isolation.py'),'--root',str(root),'--site',str(site),'--binary',str(binary),'--runtime-root',str(work/'native-runtime'),'--emit-command',str(emitted)]
 if os.environ.get('IMAGINE_H3_KITCHEN_LIBRARY'):
  command+=['--kitchen-library',os.environ['IMAGINE_H3_KITCHEN_LIBRARY'],'--kitchen-provenance',os.environ['IMAGINE_H3_KITCHEN_PROVENANCE']]
 for flag,path in [('aot-library','triton-aot/libimagine_triton_fp8.so'),('aot-fixture','triton-aot/probe.json'),('int8-library','int8-aot/libimagine_int8_probe.so'),('int8-fixture','int8-aot/probe.json'),('aimdo-library','aimdo/aimdo.so'),('pillow-library','pillow/libimagine_pillow.so')]:command+=['--'+flag,str(root/path)]
 command+=['--backend-fixture',str(project/'inference/fixtures/kitchen-0238-backend-contract.json')]
 subprocess.run(command,cwd=project,check=True)
 args=json.loads(emitted.read_text())['command'][1:]
 index=args.index('--chdir')+2
 # The caller supplies an explicit worker argument vector, after reviewed mounts.
 return args[:index]
def reservation_command(controller,entry,socket,job,manifest,args,timeout=1800):
 assert socket==CANONICAL_SOCKET and socket.is_absolute()
 assert '--'in args
 return [str(controller),'--process-form','bwrap-pid-namespace','--namespace-entry',str(entry),'--socket',str(socket),'--job',job,'--bytes','1','--timeout',str(timeout),'--manifest',str(manifest),'--','/usr/bin/bwrap',*[str(x)for x in args]]
