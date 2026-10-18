#!/usr/bin/env python3
"""Rust typed frontend and original optimizer in a Python-free namespace."""
import argparse,hashlib,json,re,subprocess,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--reference-directory',type=Path);p.add_argument('--compile-selected',action='store_true');p.add_argument('--raw-frontend',action='store_true');p.add_argument('--family',choices=['fp8','rowwise','adaln','rope','rms-rope','mxfp8','nvfp4','dequant-nvfp4','int8','w4a8','na'],default='fp8');a=p.parse_args()
family_stem='triton-frontend'if a.family=='fp8'else f'triton-{a.family}-frontend'
reference=a.reference_directory or a.root/family_stem;oracle=json.loads((reference/'reference.json').read_text());output=reference/'standalone';output.mkdir(exist_ok=True)
tools={'frontend':a.root/'target/debug/examples/triton_fp8_frontend','triton-opt':a.root/'triton-standalone/make/bin/triton-opt'}
if a.compile_selected:
    lock=json.loads((Path(__file__).resolve().parents[2]/'design/stage-1-triton-toolchain-lock.json').read_text())
    tools.update({'mlir-translate':a.root/'llvm'/lock['llvm']['distribution']/'bin/mlir-translate','llvm-driver':a.root/'triton-standalone/make/bin/imagine-triton-llvm','ptxas':Path('/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas')})
    assert hashlib.sha256(tools['ptxas'].read_bytes()).hexdigest()==lock['reference_cuda_codegen_inputs']['triton/backends/nvidia/bin/ptxas']['sha256']
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
mounts=[];libs=set()
for name,binary in tools.items():
    mounts+=['--ro-bind',str(binary),'/bin/'+name]
    undefined=subprocess.check_output(['nm','-D','--undefined-only',str(binary)],text=True);assert not re.search(r'\b(?:Py[A-Z]|_Py)',undefined)
    linked=subprocess.check_output(['ldd',str(binary)],text=True);assert'not found'not in linked
    libs.update(Path(x)for x in re.findall(r'(?:=>\s*)?(/[^\s]+)',linked))
for lib in sorted(libs):
    assert'python'not in str(lib).lower();mounts+=['--ro-bind',str(lib),str(lib)]
for source in oracle['compiler_source_files']:
    path=reference/source['copy'];assert sha(path)==source['sha256'] and path.stat().st_mtime_ns==source['mtime_ns'] and path.stat().st_size==source['size'];mounts+=['--ro-bind',str(path),source['reference_path']]
for library in oracle.get('external_libraries',[]):
    path=Path(library['path']);assert sha(path)==library['sha256'];mounts+=['--ro-bind',str(path),str(path)]
base=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/bin','--dir','/usr/bin','--dir','/usr/lib','--dir','/usr/lib64','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',*mounts,'--bind',str(output),'/output','--chdir','/tmp']
report=dict(schema=1,status='in progress',scope=oracle['scope'],interpreter_available=False,python_library_available=False,comfy_tree_available=False,gpu_available=False,raw_frontend=a.raw_frontend,network_namespace='isolated',binaries={n:sha(b)for n,b in tools.items()},cases=[],compiled=[])
try:
    for case in oracle['cases'][:a.limit]:
        i=case['index'];expected=reference/f'case-{i}.expected.ttir';assert sha(expected)==case['expected_sha256']
        request=dict(case,raw=a.raw_frontend)
        raw=subprocess.check_output([*base,'/bin/frontend'],input=json.dumps(request).encode());(output/f'case-{i}.raw.ttir').write_bytes(raw)
        if a.raw_frontend:
            expected_source=reference/f'case-{i}.expected.source';assert sha(expected_source)==case['raw_sha256']
            subprocess.run([*base,'/bin/triton-opt',f'/output/case-{i}.raw.ttir','--mlir-print-debuginfo','--mlir-use-nameloc-as-prefix','-o',f'/output/case-{i}.source'],check=True)
            source=output/f'case-{i}.source'
            # The CLI prints one additional outer newline; the original Python
            # module.__str__ serializes exactly one final newline into cache.
            source.write_bytes(source.read_bytes().rstrip(b'\n')+b'\n')
            assert source.read_bytes()==expected_source.read_bytes(),f'Original raw source IR differs: case {i}'

        subprocess.run([*base,'/bin/triton-opt',f'/output/case-{i}.raw.ttir','--pass-pipeline='+(oracle['profiles'][str(case['config_index'])]['ttir_pipeline']if a.family=='int8'else oracle['ttir_pipeline']),'--mlir-print-debuginfo','--mlir-use-nameloc-as-prefix','-o',f'/output/case-{i}.ttir'],check=True)
        candidate=output/f'case-{i}.ttir'
        assert expected.read_bytes().rstrip(b'\n')==candidate.read_bytes().rstrip(b'\n'),f'Original optimized TTIR differs: case {i}'
        report['cases'].append(dict(index=i,source='byte-exact'if a.raw_frontend else'not part of this comparison',optimized_ttir='exact',sha256=sha(candidate)))
    if a.compile_selected:
        def run(cmd):subprocess.run([*base,*cmd],check=True)
        def compare(index,ext):
            candidate=output/f'case-{index}.{ext}';expected=reference/f'case-{index}.expected.{ext}'
            x=expected.read_bytes();y=candidate.read_bytes()
            if ext!='cubin':x=x.rstrip(b'\n');y=y.rstrip(b'\n')
            assert x==y,f'Original compiler {ext} differs: {index}'
        for compiled in oracle['compiled']:
            i=compiled['index']
            for name,expected in compiled['files'].items():assert sha(reference/name)==expected
            # Custom textual scf.for syntax omits region-argument locations.
            # Preserve the original in-memory graph between native processes;
            # render the original pretty text separately for exact cache checks.
            run(['/bin/triton-opt',f'/output/case-{i}.raw.ttir','--pass-pipeline='+(oracle['profiles'][str(compiled['config_index'])]['ttir_pipeline']if a.family=='int8'else oracle['ttir_pipeline']),'--emit-bytecode','-o',f'/output/case-{i}.ttir.mlirbc'])
            previous=f'/output/case-{i}.ttir.mlirbc'
            for ext,pipeline in [('ttgir',compiled['ttgir_pipeline']),('llvm.mlir',compiled['llir_pipeline'])]:
                current=f'/output/case-{i}.{ext}.mlirbc'
                run(['/bin/triton-opt',previous,'--pass-pipeline='+pipeline,'--emit-bytecode','-o',current])
                run(['/bin/triton-opt',current,'--mlir-print-debuginfo','--mlir-use-nameloc-as-prefix','-o',f'/output/case-{i}.{ext}']);compare(i,ext)
                previous=current
            run(['/bin/mlir-translate','--mlir-to-llvmir',previous,'-o',f'/output/case-{i}.unoptimized.ll'])
            run(['/bin/llvm-driver','optimize',f'/output/case-{i}.unoptimized.ll',f'/output/case-{i}.llir','sm_89','+ptx87','87',*[x['path']for x in oracle.get('external_libraries',[])]]);compare(i,'llir')
            run(['/bin/llvm-driver','ptx',f'/output/case-{i}.llir',f'/output/case-{i}.ptx','sm_89','+ptx87','87']);compare(i,'ptx')
            expected_error=compiled.get('cubin_error')
            if expected_error is not None:
                assert expected_error['class_name']=='PTXASError'
                original=expected_error['message'];command=shlex.split(original.split('Repro command: ',1)[1].strip());assert command[-2]=='-o'and command[-1]==command[-3]+'.o'
                # Use exactly the captured original temporary input name inside
                # our private tmpfs, so even the native assembler diagnostic is
                # compared without erasing or normalizing any message bytes.
                proc=subprocess.run([*base,'--ro-bind',str(output/f'case-{i}.ptx'),command[-3],'/bin/ptxas',*command[1:]],capture_output=True,text=True)
                assert proc.returncode!=0
                reason='Internal Triton PTX codegen error'if proc.returncode==255 else'`ptxas` raised SIGSEGV'if proc.returncode==139 else f'`ptxas` failed with error code {proc.returncode}'
                actual=f'PTXAS error: {reason}\n`ptxas` stderr:\n{proc.stderr}\nRepro command: {" ".join(command)}\n'
                assert actual==original,f'Original complete PTXAS diagnostic differs: {i}'
                report['compiled'].append(dict(index=i,ttgir='exact',llvm_mlir='exact',optimized_llir='exact',ptx='exact',cubin_error='byte-exact',class_name='PTXASError',message_sha256=hashlib.sha256(actual.encode()).hexdigest(),returncode=proc.returncode))
            else:
                run(['/bin/ptxas','-lineinfo','-v','--regAllocOptLevel=2','--gpu-name=sm_89',f'/output/case-{i}.ptx','-o',f'/output/case-{i}.cubin']);compare(i,'cubin')
                report['compiled'].append(dict(index=i,ttgir='exact',llvm_mlir='exact',optimized_llir='exact',ptx='exact',complete_cubin='byte-exact',cubin_sha256=sha(output/f'case-{i}.cubin')))
        assert len(report['compiled'])==oracle.get('expected_compiled_count', {'fp8':24,'rowwise':3,'adaln':6*len(oracle.get('epsilon_types',['fp32'])),'rope':72,'rms-rope':(6 if oracle.get('scalar_slice')else 54)*len(oracle.get('epsilon_types',['fp32'])),'mxfp8':9,'nvfp4':18,'dequant-nvfp4':6,'int8':24*len(oracle.get('output_types',['fp32','fp16','bf16'])),'w4a8':4,'na':6 if oracle.get('dynamic_kernels')else 30}[a.family])
    for original in oracle.get('source_errors',[]):
        completed=subprocess.run([*base,'/bin/frontend'],input=json.dumps(original['request']).encode(),capture_output=True)
        assert completed.returncode==2
        actual=json.loads(completed.stdout)
        assert actual==dict(error=original['class_name'],message=original['message']),f"Original compiler diagnostic differs: {original['request']}"
    report['source_error_cases']=len(oracle.get('source_errors',[]))
    report['status']='passed';print(f"{len(report['cases'])} exact Rust frontend/TTIR comparisons in Python-free isolation",flush=True)
except Exception as e:
    report.update(status='failed',error=str(e));raise
finally:(a.root/'runtime'/f'{a.reference_directory.name if a.reference_directory else family_stem}-isolation.json').write_text(json.dumps(report,indent=2)+'\n')
