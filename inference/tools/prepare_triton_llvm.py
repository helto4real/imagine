#!/usr/bin/env python3
"""Rebind exact original LLVM helper bodies, removing only Python bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
lock=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())

def read(name):
    p=args.source/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==lock['triton']['source_file_sha256'][name]
    return p.read_text()

def body(source,name):
    match=re.search(r'm\.def\(\s*"'+name+r'"\s*,\s*\[\]\([^)]*\)\s*(?:->\s*bool\s*)?\{',source)
    assert match,name
    start=match.end();depth=1
    tokens=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[{}]',re.S)
    for token in tokens.finditer(source,start):
        if token.group()=='{':depth+=1
        elif token.group()=='}':
            depth-=1
            if depth==0:return source[start:token.start()]
    raise AssertionError(name)

source=read('python/src/llvm.cc')
prefix=source.split('using ret = py::return_value_policy;',1)[0]
prefix=re.sub(r'^#include <pybind11/[^>]+>\n','',prefix,flags=re.M)
prefix=prefix.replace('namespace py = pybind11;','')
assert 'py::' not in prefix and 'pybind11' not in prefix
output=prefix
output+='\nvoid imagine_init_targets() {'+body(source,'init_targets')+'}\n'
output+='\nvoid imagine_attach_datalayout(llvm::Module *mod, const std::string triple, const std::string proc, const std::string features) {'+body(source,'attach_datalayout')+'}\n'
output+='\nvoid imagine_optimize_module(llvm::Module *mod) {\nconst auto opt=llvm::OptimizationLevel::O3; const std::string arch,features; const std::vector<std::string> flags; const bool enable_fp_fusion=false;\n'+body(source,'optimize_module')+'}\n'
output+='\nvoid imagine_link_extern_libs(llvm::Module *dstMod, const std::vector<std::string> &paths) {'+body(source,'link_extern_libs')+'}\n'
nvidia=read('third_party/nvidia/triton_nvidia.cc')
output+='\nvoid imagine_set_short_ptr() {'+body(nvidia,'set_short_ptr')+'}\n'
output+='\nvoid imagine_set_nvvm_reflect_ftz(llvm::Module *mod) {'+body(nvidia,'set_nvvm_reflect_ftz')+'}\n'
output+='\nbool imagine_has_extern_deps(llvm::Module *dstMod) {'+body(nvidia,'has_extern_deps')+'}\n'
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'original_triton_llvm.inc').write_text(output)
report={'schema':1,'transform':'original helper prefix and lambda bodies; remove Python binding includes/types; default optimize_module arguments unchanged',
        'source_functions':['init_targets','attach_datalayout','optimize_module','link_extern_libs','set_short_ptr','set_nvvm_reflect_ftz','has_extern_deps','translateLLVMIRToASM'],
        'generated_sha256':hashlib.sha256(output.encode()).hexdigest(),'algorithms_or_flags_modified':False}
(args.output/'llvm-binding-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('Prepared original Triton LLVM helper bodies without Python bindings.')
