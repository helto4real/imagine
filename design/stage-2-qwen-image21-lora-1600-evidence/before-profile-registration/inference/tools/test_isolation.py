#!/usr/bin/env python3
"""Expose only selected ELF files and synthetic fixtures to a Python-free test.

Host-side development launcher. No interpreter, ComfyUI tree, Python package,
ordinary runtime directory is mounted. Primitive mode mounts no weights;
explicit Qwen-driver mode adds its three authorized models and selected,
hash-checked reference tensor fixtures when the registered profile requests them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--aot-library',type=Path,required=True)
parser.add_argument('--aot-fixture',type=Path,required=True)
parser.add_argument('--int8-library',type=Path,required=True)
parser.add_argument('--int8-fixture',type=Path,required=True)
parser.add_argument('--backend-fixture',type=Path,required=True)
parser.add_argument('--test-filter',help='Run one exact test; require one executed success')
parser.add_argument('--pillow-library',type=Path,help='Verified original-source Pillow buffer ABI; no interpreter')
parser.add_argument('--aimdo-library',type=Path,help='Explicit isolated copy of the frozen AIMDO ELF')
parser.add_argument('--qwen-driver-profile',type=Path,help='Explicit frozen first-image development profile; switches from primitive tests to the standalone driver')
parser.add_argument('--qwen-model-root',type=Path,help='Authorized local model root; mount only the three profile checkpoint files read-only')
parser.add_argument('--qwen-mode',choices=['preflight','load-check','run'],default='preflight')
parser.add_argument('--qwen-run-name',default='qwen21-first-image',help='Fresh isolated output directory name; no paths')
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
driver_mounts=[];driver_args=None;model_files=[];reference_files=[]
if args.qwen_driver_profile is not None:
    if args.test_filter or args.qwen_model_root is None or args.aimdo_library is None:
        parser.error('Driver mode requires explicit model root/AIMDO and cannot use a primitive test filter')
    profile=json.loads(args.qwen_driver_profile.read_text())
    assert profile in [json.loads((project/'inference/profiles'/name).read_text())for name in ['qwen21-first-int8.json','qwen21-1600x896-int8.json','qwen21-1600x896-40steps-int8.json','qwen21-1600x896-40steps-bf16.json','qwen21-1600x896-40steps-bf16-offload.json','qwen21-1600x896-8steps-turbo-offload.json','qwen21-editing-int8.json','qwen21-editing-bf16.json','qwen21-editing-turbo.json','qwen21-editing-int8-two-references.json','qwen21-editing-int8-keep-vision.json','qwen21-portrait-1mpx-int8.json','qwen21-editing-portrait-1mpx-int8.json','qwen21-editing-portrait-1mpx-int8-seed3061383787.json','qwen21-lora-int8.json','qwen21-lora-bf16.json','qwen21-lora-turbo.json','qwen21-lora-text-int8.json','qwen21-lora-stack-int8.json']],'Explicit registered comparison profile required'
    for rel in profile['checkpoints'].values():
        relative=Path(rel);assert not relative.is_absolute()and'..'not in relative.parts
        source=args.qwen_model_root/relative;assert source.is_file(),source
        target='/models/'+relative.as_posix();driver_mounts+=['--ro-bind',str(source),target]
        model_files.append(dict(source=str(source),target=target,bytes=source.stat().st_size))
    for spec in profile.get('lora',[]):
        relative=Path(spec['path']);assert not relative.is_absolute() and '..' not in relative.parts
        if 'fixture_file' in spec:
            fixture=Path(spec['fixture_file']);assert not fixture.is_absolute() and '..' not in fixture.parts
            source=project/fixture;assert source.resolve().is_relative_to((project/'inference/fixtures').resolve())
        else:source=args.qwen_model_root/relative
        assert source.is_file()
        assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==spec['sha256']
        target='/models/'+relative.as_posix();driver_mounts+=['--ro-bind',str(source),target]
        model_files.append(dict(source=str(source),target=target,bytes=source.stat().st_size,sha256=spec['sha256'],role='lora'))
    for index,spec in enumerate(profile.get('references',[])):
        if 'tensor_file' not in spec:continue
        relative=Path(spec['tensor_file']);assert not relative.is_absolute() and '..' not in relative.parts
        source=project/relative
        assert source.resolve().is_relative_to((project/'inference/fixtures').resolve())
        assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==spec['tensor_sha256']
        target=f'/runtime/qwen-reference-{index}.f32'
        driver_mounts+=['--ro-bind',str(source),target,'--setenv',f'IMAGINE_QWEN_REFERENCE_{index}',target]
        reference_files.append(dict(source=str(source),target=target,bytes=source.stat().st_size,sha256=spec['tensor_sha256']))
    assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.qwen_run_name)
    work=args.root/'runtime'/args.qwen_run_name;work.mkdir(parents=True,exist_ok=True)
    driver_mounts+=['--ro-bind',str(args.qwen_driver_profile),'/runtime/qwen21-profile.json','--bind',str(work),'/qwen-work']
    driver_args=['--model-root','/models','--profile','/runtime/qwen21-profile.json','--own-root','/qwen-work']
    if args.qwen_mode=='load-check':driver_args+=['--load-check']
    if args.qwen_mode=='run':driver_args+=['--run']
elif args.qwen_model_root is not None or args.qwen_mode!='preflight':
    parser.error('Qwen root/mode require --qwen-driver-profile')
lock=json.loads((project/'design/reference-local/native-linkage-lock.json').read_text())
objects={Path(x['path']):x for x in lock['distribution_objects']}
files=[Path(p) for p in lock['initial_target_linkage_audit']['package_object_closure'] if '/torch/lib/' in p or '/nvidia/' in p]
# cuDNN, NVRTC builtins and Torch/cuSOLVER linear algebra are dlopen dependencies, invisible to
# the initial DT_NEEDED closure. Expose only the already locked ELF objects.
lazy_files=sorted(path for path in objects if path.parent==args.site/'nvidia/cudnn/lib'
                  or path.parent==args.site/'nvidia/cu13/lib' and (path.name.startswith('libnvrtc-builtins') or path.name=='libcusolver.so.12')
                  or path==args.site/'torch/lib/libtorch_cuda_linalg.so')
files=list(dict.fromkeys([*files,*lazy_files]))
bridge=args.root/'kitchen-manylinux/libimagine_kitchen.so'
if args.aimdo_library is not None:
    expected=objects[args.site/'comfy_aimdo/aimdo.so']['sha256']
    assert hashlib.file_digest(args.aimdo_library.open('rb'),'sha256').hexdigest()==expected
fixture=args.root/'runtime/primitives.json'
triton_driver=args.root/'triton-driver/libimagine_triton_driver.so'
triton_cubins=args.root/'triton-frontend/standalone'
triton_mounts=[]
triton_cubin_hashes={}
for path in sorted(triton_cubins.glob('*.cubin')):
    triton_mounts+=['--ro-bind',str(path),'/runtime/triton-cubins/'+path.name]
    triton_cubin_hashes[path.name]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
assert len(triton_cubin_hashes)==24, 'Complete Rust-fronted typed compiler outputs required'
compiler_profile_path=args.root/'runtime/triton-compiler-profile.json'
compiler_profile=json.loads(compiler_profile_path.read_text())
compiler_binaries=[];compiler_files={};comfy_files={}
comfy_root=Path('/home/thhel/git/ComfyUI')
for tool in compiler_profile['tools'].values():
    path=Path(tool['path']);h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert h==tool['sha256']
    undefined=subprocess.check_output(['nm','-D','--undefined-only',str(path)],text=True)
    assert not re.search(r'\b(?:Py[A-Z]|_Py)',undefined),path
    compiler_binaries.append(path);compiler_files[str(path)]=h
    triton_mounts+=['--ro-bind',str(path),str(path)]
    if path.is_relative_to(comfy_root):comfy_files[str(path)]=h
toolchain=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())
for library in compiler_profile['external_libraries']:
    path=Path(library['path']);h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert h==library['sha256']
    assert h==toolchain['reference_cuda_codegen_inputs'][str(path.relative_to(args.site))]['sha256']
    triton_mounts+=['--ro-bind',str(path),str(path)]
    compiler_files[str(path)]=h
    if path.is_relative_to(comfy_root):comfy_files[str(path)]=h
source=compiler_profile['source_file'];source_copy=Path(source['copy']);source_path=Path(source['path'])
assert source_path==args.site/'comfy_kitchen/backends/triton/quantization.py'
source_hash=hashlib.file_digest(source_copy.open('rb'),'sha256').hexdigest();assert source_hash==source['sha256']
source_manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
assert source_hash==next(e['sha256']for e in source_manifest if e['root']=='site'and e['path']=='comfy_kitchen/backends/triton/quantization.py')
assert source_copy.stat().st_size==source['size']and source_copy.stat().st_mtime_ns==source['mtime_ns']
triton_mounts+=['--ro-bind',str(source_copy),str(source_copy),'--ro-bind',str(source_copy),str(source_path)]
comfy_files[str(source_path)]=source_hash
for source in compiler_profile['extra_source_files']:
    path=Path(source['path']);copy=Path(source['copy']);relative=str(path.relative_to(args.site));assert relative in ['triton/language/standard.py','comfy_kitchen/backends/triton/adaln.py','comfy_kitchen/backends/triton/rope.py','comfy_kitchen/backends/triton/rms_rope.py','comfy_kitchen/backends/triton/w4a8_int8.py','comfy_kitchen/backends/triton/na.py']
    h=hashlib.file_digest(copy.open('rb'),'sha256').hexdigest();assert h==source['sha256']==next(e['sha256']for e in source_manifest if e['root']=='site'and e['path']==relative)
    assert copy.stat().st_size==source['size']and copy.stat().st_mtime_ns==source['mtime_ns']
    triton_mounts+=['--ro-bind',str(copy),str(copy),'--ro-bind',str(copy),str(path)];comfy_files[str(path)]=h
# The frozen cuobjdump delegates disassembly to the same existing local tool
# found on the original development PATH. Mount/hash only this ELF dependency.
disassembly=json.loads((project/'design/stage-1-disassembly-dependency-lock.json').read_text())['tool']
path=Path(disassembly['path']);h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
assert h==disassembly['sha256']
compiler_binaries.append(path);compiler_files[str(path)]=h
triton_mounts+=['--ro-bind',str(path),str(path),'--setenv','PATH',str(path.parent)]
triton_mounts+=['--ro-bind',str(compiler_profile_path),'/runtime/triton-compiler-profile.json','--setenv','IMAGINE_TRITON_COMPILER_PROFILE','/runtime/triton-compiler-profile.json','--setenv','IMAGINE_ISOLATED_COMFY_FILES',json.dumps(comfy_files)]

env=dict(os.environ,LD_LIBRARY_PATH=str(args.site/'torch/lib'))
system=set()
for binary in [args.binary,bridge,triton_driver,*compiler_binaries,Path('/usr/lib/libcuda.so.1'),*lazy_files,*([args.aimdo_library]if args.aimdo_library else[]),*([args.pillow_library]if args.pillow_library else[])]:
    output=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
    if 'not found' in output:raise RuntimeError(output)
    for path in re.findall(r'(?:=>\s*)?(/[^\s]+)',output):
        path=Path(path)
        if str(path).startswith('/usr/lib'):system.add(path)
mounts=[]
records=[]
if args.aimdo_library is not None:
    mounts+=['--ro-bind',str(args.aimdo_library),'/runtime/aimdo.so','--setenv','IMAGINE_AIMDO_LIBRARY','/runtime/aimdo.so']
    records.append(dict(target='/runtime/aimdo.so',sha256=expected))
if args.pillow_library is not None:
    provenance=json.loads((args.pillow_library.parent/'provenance.json').read_text())
    assert provenance['upstream']==json.loads((project/'inference/resources/pillow-source-lock.json').read_text())
    assert not provenance['python_runtime_required']
    assert hashlib.sha256(args.pillow_library.read_bytes()).hexdigest()==provenance['library_sha256']
    for name,h in provenance['adapter_sha256'].items():assert hashlib.sha256((project/name).read_bytes()).hexdigest()==h,name
    mounts+=['--ro-bind',str(args.pillow_library),'/runtime/libimagine_pillow.so','--setenv','IMAGINE_PILLOW_LIBRARY','/runtime/libimagine_pillow.so']
    records.append(dict(target='/runtime/libimagine_pillow.so',sha256=provenance['library_sha256']))
for path in files:
    if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=objects[path]['sha256']:
        raise RuntimeError(f'Frozen native source changed: {path}')
    target='/runtime/'+str(path.relative_to(args.site))
    mounts+=['--ro-bind',str(path),target]
    records.append({'target':target,'sha256':objects[path]['sha256']})
for path in sorted(system|{Path('/usr/lib/libcuda.so.1')}):
    if 'python' in str(path).lower():raise RuntimeError('Python dependency detected')
    mounts+=['--ro-bind',str(path),str(path)]
devices=[]
for name in ['nvidia0','nvidiactl','nvidia-uvm','nvidia-uvm-tools']:
    path=Path('/dev')/name
    if path.exists():devices+=['--dev-bind',str(path),str(path)]
command=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
    '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/bin','--dir','/usr/bin',
    '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
    '--ro-bind','/sys','/sys',*mounts,*devices,
    '--ro-bind',str(bridge),'/runtime/libimagine_kitchen.so',
    '--ro-bind',str(triton_driver),'/runtime/libimagine_triton_driver.so',*triton_mounts,
    '--setenv','IMAGINE_TRITON_DRIVER_LIBRARY','/runtime/libimagine_triton_driver.so',
    '--setenv','IMAGINE_TRITON_FRONTEND_CUBINS','/runtime/triton-cubins',
    '--ro-bind',str(args.binary),'/runtime/primitives',
    '--ro-bind',str(fixture),'/runtime/primitives.json',
    '--ro-bind',str(args.aot_library),'/runtime/libimagine_triton_fp8.so',
    '--ro-bind',str(args.aot_fixture),'/runtime/triton-probe.json',
    '--ro-bind',str(args.int8_library),'/runtime/libimagine_int8_probe.so',
    '--ro-bind',str(args.int8_fixture),'/runtime/int8-probe.json',
    '--ro-bind',str(args.backend_fixture),'/runtime/backend-contract.json',
    '--setenv','LD_LIBRARY_PATH','/runtime/torch/lib:/runtime/nvidia/cudnn/lib:/runtime/nvidia/cu13/lib',
    '--setenv','IMAGINE_KITCHEN_LIBRARY','/runtime/libimagine_kitchen.so',
    '--setenv','IMAGINE_PRIMITIVE_FIXTURES','/runtime/primitives.json',
    '--setenv','IMAGINE_EXPECT_ISOLATION','1',
    '--setenv','IMAGINE_TRITON_PROBE_LIBRARY','/runtime/libimagine_triton_fp8.so',
    '--setenv','IMAGINE_TRITON_PROBE_FIXTURE','/runtime/triton-probe.json',
    '--setenv','IMAGINE_INT8_PROBE_LIBRARY','/runtime/libimagine_int8_probe.so',
    '--setenv','IMAGINE_INT8_PROBE_FIXTURE','/runtime/int8-probe.json',
    '--setenv','IMAGINE_BACKEND_CONTRACT_FIXTURE','/runtime/backend-contract.json',
    '--setenv','CUDA_CACHE_PATH','/tmp/cuda',
    *driver_mounts,'--chdir','/tmp','/runtime/primitives',*(driver_args if driver_args is not None else ['--test-threads=1'])]
if args.test_filter:
    command += [args.test_filter,'--exact']
    completed=subprocess.run(command,capture_output=True,text=True)
    print(completed.stdout,end='')
    print(completed.stderr,end='',file=sys.stderr)
    completed.check_returncode()
    if 'test result: ok. 1 passed; 0 failed;' not in completed.stdout:
        raise RuntimeError('Filtered isolation must execute exactly one passing test')
else:
    subprocess.run(command,check=True)
report={'schema':1,'result':'passed','interpreter_available':False,'comfyui_tree_available':False,'comfyui_installation_available':False,'compiler_only_files_at_reference_paths':comfy_files,'runtime_compiler_elf_files':compiler_files,'compiler_profile_sha256':hashlib.file_digest(compiler_profile_path.open('rb'),'sha256').hexdigest(),
    'network_namespace':'isolated','test_filter':args.test_filter,'package_elf_files':records,
    'driver_mode':args.qwen_mode if driver_args is not None else None,'authorized_model_files':model_files,'authorized_reference_files':reference_files,
    'system_elf_files':[str(p) for p in sorted(system)],
    'system_elf_hashes':{str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(system)},
    'lazy_package_elf_files':['/runtime/'+str(p.relative_to(args.site)) for p in lazy_files],
    'triton_driver_sha256':hashlib.file_digest(triton_driver.open('rb'),'sha256').hexdigest(),
    'triton_frontend_cubin_sha256':triton_cubin_hashes,
    'bridge_sha256':hashlib.file_digest(bridge.open('rb'),'sha256').hexdigest(),
    'binary_sha256':hashlib.file_digest(args.binary.open('rb'),'sha256').hexdigest(),
    'aot_binary_sha256':hashlib.file_digest(args.aot_library.open('rb'),'sha256').hexdigest(),
    'aot_fixture_sha256':hashlib.file_digest(args.aot_fixture.open('rb'),'sha256').hexdigest(),
    'int8_binary_sha256':hashlib.file_digest(args.int8_library.open('rb'),'sha256').hexdigest(),
    'int8_fixture_sha256':hashlib.file_digest(args.int8_fixture.open('rb'),'sha256').hexdigest(),
    'backend_fixture_sha256':hashlib.file_digest(args.backend_fixture.open('rb'),'sha256').hexdigest(),
    'fixture_sha256':hashlib.file_digest(fixture.open('rb'),'sha256').hexdigest()}
report_name='isolation-filtered-result.json' if args.test_filter else 'isolation-result.json'
if driver_args is not None:report_name='qwen21-'+args.qwen_mode+'-isolation.json'
(args.root/'runtime'/report_name).write_text(json.dumps(report,indent=2)+'\n')
