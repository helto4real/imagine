#!/usr/bin/env python3
"""Compare original compiler outputs in an empty Python-free namespace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--site',type=Path,required=True)
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
lock=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())
reference=args.root/'triton-pipeline'
output=reference/'standalone';output.mkdir(exist_ok=True)
oracle=json.loads((reference/'reference.json').read_text())
tools={'triton-opt':args.root/'triton-standalone/make/bin/triton-opt',
       'mlir-translate':args.root/'llvm'/lock['llvm']['distribution']/'bin/mlir-translate',
       'llvm-driver':args.root/'triton-standalone/make/bin/imagine-triton-llvm',
       'ptxas':args.site/'triton/backends/nvidia/bin/ptxas'}

def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
assert sha(tools['ptxas'])==lock['reference_cuda_codegen_inputs']['triton/backends/nvidia/bin/ptxas']['sha256']
mounts=[];libraries=set()
for name,binary in tools.items():
    mounts+=['--ro-bind',str(binary),'/bin/'+name]
    undefined=subprocess.check_output(['nm','-D','--undefined-only',str(binary)],text=True)
    assert not re.search(r'\b(?:Py[A-Z]|_Py)',undefined),name
    linkage=subprocess.check_output(['ldd',str(binary)],text=True)
    assert 'not found' not in linkage,linkage
    libraries.update(Path(x) for x in re.findall(r'(?:=>\s*)?(/[^\s]+)',linkage))
for path in sorted(libraries):
    assert 'python' not in str(path).lower(),path
    mounts+=['--ro-bind',str(path),str(path)]
for item in oracle['compiler_source_files']:
    path=reference/item['copy']
    assert sha(path)==item['sha256']
    assert path.stat().st_size==item['size'] and path.stat().st_mtime_ns==item['mtime_ns']
    # Original ptxas stats the .file directive's source for DWARF time/size.
    # A verified standalone copy is compiler data, not a Python runtime/tree.
    mounts+=['--ro-bind',str(path),item['reference_path']]
base=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
      '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/bin','--dir','/usr/bin',
      '--dir','/usr/lib','--dir','/usr/lib64',
      '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',*mounts,
      '--ro-bind',str(reference),'/reference','--bind',str(output),'/output','--chdir','/tmp']
report={'schema':1,'result':'in progress','scope':'six optimized-TTIR FP8 compiler specializations; frontend/general JIT not covered',
        'interpreter_available':False,'python_library_available':False,'comfyui_tree_available':False,
        'network_namespace':'isolated','gpu_devices_available':False,
        'compiler_source_files':oracle['compiler_source_files'],
        'binary_sha256':{name:sha(path) for name,path in tools.items()},
        'system_elf_files':{str(p):sha(p) for p in sorted(libraries)},'cases':[]}

def run(command):subprocess.run([*base,*command],check=True)
def compare(name,expected,actual,text=True):
    a=expected.read_bytes();b=actual.read_bytes()
    if text:a=a.rstrip(b'\n');b=b.rstrip(b'\n')
    if a!=b:raise AssertionError(f'{name}: standalone output differs from original; expected {sha(expected)}, actual {sha(actual)}')

try:
    for case in oracle['cases']:
        index=case['index'];prefix=f'case-{index}';stem=output/prefix
        for name,expected in case['files'].items():assert sha(reference/name)==expected,name
        run(['/bin/triton-opt',f'/reference/{prefix}.ttir','--pass-pipeline='+case['ttgir_pipeline'],
             '--mlir-print-debuginfo','--mlir-use-nameloc-as-prefix','-o',f'/output/{prefix}.ttgir'])
        compare('TTGIR',reference/f'{prefix}.expected.ttgir',stem.with_suffix('.ttgir'))
        run(['/bin/triton-opt',f'/output/{prefix}.ttgir','--pass-pipeline='+case['llir_pipeline'],
             '--mlir-print-debuginfo','--mlir-use-nameloc-as-prefix','-o',f'/output/{prefix}.llvm.mlir'])
        compare('LLVM MLIR',reference/f'{prefix}.expected.llvm.mlir',stem.with_suffix('.llvm.mlir'))
        run(['/bin/mlir-translate','--mlir-to-llvmir',f'/output/{prefix}.llvm.mlir','-o',f'/output/{prefix}.unoptimized.ll'])
        ptx_version=re.search(r'^\.version (\d+)\.(\d+)',(reference/f'{prefix}.expected.ptx').read_text(),re.M)
        assert ptx_version
        features='+ptx'+ptx_version[1]+ptx_version[2]
        version=ptx_version[1]+ptx_version[2]
        run(['/bin/llvm-driver','optimize',f'/output/{prefix}.unoptimized.ll',f'/output/{prefix}.llir','sm_89',features,version])
        compare('optimized LLVM IR',reference/f'{prefix}.expected.llir',stem.with_suffix('.llir'))
        run(['/bin/llvm-driver','ptx',f'/output/{prefix}.llir',f'/output/{prefix}.ptx','sm_89',features,version])
        compare('PTX',reference/f'{prefix}.expected.ptx',stem.with_suffix('.ptx'))
        run(['/bin/ptxas','-lineinfo','-v','--regAllocOptLevel=2','--gpu-name=sm_89',
             f'/output/{prefix}.ptx','-o',f'/output/{prefix}.cubin'])
        compare('cubin',reference/f'{prefix}.expected.cubin',stem.with_suffix('.cubin'),text=False)
        report['cases'].append({'index':index,'ttgir':'exact','llvm_mlir':'exact','optimized_llir':'exact','ptx':'exact','cubin':'byte-exact',
                                'output_sha256':{p.name:sha(p) for p in sorted(output.glob(prefix+'.*'))}})
        print(f'Compiler case {index}: exact TTGIR, LLVM MLIR, optimized LLVM IR, PTX and cubin',flush=True)
    report['result']='passed'
except Exception as error:
    report['result']='failed';report['error']=str(error)
    raise
finally:
    (args.root/'runtime/triton-standalone-isolation.json').write_text(json.dumps(report,indent=2)+'\n')
