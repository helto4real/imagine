#!/usr/bin/env python3
"""Original external-library policy on synthetic LLVM modules, CPU only."""
import argparse,hashlib,json,sys
from pathlib import Path

def capture(site,output,project):
    lock=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())
    source=output.parent/'sources/triton-candidate'
    for name in ['python/src/llvm.cc','third_party/nvidia/triton_nvidia.cc']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==lock['triton']['source_file_sha256'][name]
    sys.path.insert(0,str(site))
    from triton._C.libtriton import llvm,nvidia,ir
    llvm.init_targets();output.mkdir(parents=True,exist_ok=True)
    libdevice=site/'triton/backends/nvidia/lib/libdevice.10.bc'
    assert hashlib.sha256(libdevice.read_bytes()).hexdigest()==lock['reference_cuda_codegen_inputs']['triton/backends/nvidia/lib/libdevice.10.bc']['sha256']
    libraries=[]
    def library(name,text):
        path=output/name;path.write_text(text);libraries.append(path);return str(path)
    foo=library('foo.ll','declare float @probe_bar(float)\ndefine float @probe_foo(float %x) {\n %v = call float @probe_bar(float %x)\n ret float %v\n}\ndefine float @unused_foo(float %x) {\n ret float %x\n}\n')
    bar=library('bar.ll','define float @probe_bar(float %x) {\n %v = fmul float %x, 2.0\n ret float %v\n}\ndefine float @unused_bar(float %x) {\n ret float %x\n}\n')
    bad=library('invalid.ll','not LLVM IR\n');missing=str(output/'missing.ll')
    cases=[]
    scenarios=[(['__nv_exp'],[str(libdevice)],None),(['__nv_logf','__nv_sinf'],[str(libdevice)],None),(['__nv_tanhf','__nv_nearbyintf'],[str(libdevice)],None),(['llvm.sqrt.f32'],[str(libdevice)],None),(['llvm.sqrt.f32'],[missing],None),(['probe_foo'],[foo,bar],None),(['probe_foo'],[bar,foo],None),(['probe_foo'],[],None),(['probe_foo'],[bad],None),(['probe_foo'],[missing],None),(['llvm.sqrt.f32'],[missing],'global_smem'),(['llvm.sqrt.f32'],[str(libdevice)],'outside')]
    for names,paths,global_name in scenarios:
        index=len(cases);ty='f64'if names==['__nv_exp']else'f32'
        declarations='\n'.join(f'llvm.func @{name}({ty}) -> {ty}'for name in names)
        body=[];previous='%x'
        for i,name in enumerate(names):body.append(f' %v{i} = llvm.call @{name}({previous}) : ({ty}) -> {ty}');previous=f'%v{i}'
        global_decl=f'llvm.mlir.global external @{global_name}() : f32\n'if global_name else''
        mlir='module {\n'+global_decl+declarations+f'\nllvm.func @probe(%x: {ty}) -> {ty} {{\n'+'\n'.join(body)+f'\n llvm.return {previous} : {ty}\n}}\n}}\n'
        path=output/f'case-{index}.mlir';path.write_text(mlir)
        ictx=ir.context();ir.load_dialects(ictx);nvidia.load_dialects(ictx)
        source=ir.parse_mlir_module(str(path),ictx);context=llvm.context();module=llvm.to_module(source,context)
        input_path=output/f'case-{index}.input.ll';input_path.write_text(str(module))
        nvidia.set_short_ptr();llvm.attach_datalayout(module,'nvptx64-nvidia-cuda','sm_89','+ptx87');nvidia.set_nvvm_reflect_ftz(module)
        has=nvidia.has_extern_deps(module)
        case=dict(index=index,names=names,global_name=global_name,has_external_dependencies=has,libraries=paths,input=input_path.name,input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
        try:
            if paths and has:llvm.link_extern_libs(module,paths)
            llvm.optimize_module(module,llvm.OPTIMIZE_O3)
            expected=output/f'case-{index}.expected.ll';expected.write_text(str(module));case.update(expected=expected.name,expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest())
        except Exception as error:case.update(error=type(error).__name__,message=str(error))
        cases.append(case)
    result=dict(schema=1,gpu_work=False,cases=cases,libraries={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [libdevice,*libraries]},source_function_hashes={name:lock['triton']['source_file_sha256'][name]for name in ['python/src/llvm.cc','third_party/nvidia/triton_nvidia.cc']})
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n');print(f'{len(cases)} original external-library compiler cases; no GPU work',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    capture(a.site,a.output,Path(__file__).resolve().parents[2])
